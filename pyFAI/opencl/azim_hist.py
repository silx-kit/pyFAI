# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#                            D. Karkoulis (dimitris.karkoulis@gmail.com)
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  .
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  .
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

from __future__ import absolute_import, print_function, with_statement, division

"""
C++ less implementation of Dimitris' code based on PyOpenCL

TODO and trick from dimitris still missing:
  * dark-current subtraction is still missing
  * In fact you might want to consider doing the conversion on the GPU when
    possible. Think about it, you have a uint16 to float which for large arrays
    was slow.. You load on the graphic card a uint16 (2x transfer speed) and
    you convert to float inside so it should be blazing fast.


"""
__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__date__ = "16/05/2019"
__copyright__ = "2012, ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import os
import logging
import threading
from collections import OrderedDict
import numpy

from . import ocl, pyopencl
if pyopencl is not None:
    mf = pyopencl.mem_flags
else:
    raise ImportError("pyopencl is not installed")

from . import allocate_cl_buffers, release_cl_buffers, kernel_workgroup_size
from . import concatenate_cl_kernel, get_x87_volatile_option, processing
from ..containers import Integrate1dtpl, Integrate2dtpl
EventDescription = processing.EventDescription
OpenclProcessing = processing.OpenclProcessing
BufferDescription = processing.BufferDescription
from ..utils import calc_checksum
logger = logging.getLogger(__name__)


class Integrator1d(object):
    """
    Attempt to implements ocl_azim using pyopencl
    """
    BLOCK_SIZE = 128

    def __init__(self, filename=None):
        """

        :param filename: file in which profiling information are saved
        """
        self.BLOCK_SIZE = 128
        self.tdim = (self.BLOCK_SIZE,)
        self.wdim_bins = None
        self.wdim_data = None
        self._tth_min = self._tth_max = self.tth_min = self.tth_max = None
        self.nBins = -1
        self.nData = -1
        self.platformid = -1
        self.deviceid = -1
        self.useFp64 = False
        self.devicetype = "gpu"
        self.filename = filename
        self.tth_out = None
        if "write" in dir(filename):
            self.logfile = filename
        elif filename:
            self.logfile = open(self.filename, "a")
        else:
            self.logfile = None
        self.lock = threading.Semaphore()
        # Those are pointer to memory on the GPU (or None if uninitialized
        self._cl_mem = {"tth": None,
                        "tth_delta": None,
                        "image": None,
                        "solidangle": None,
                        "dark": None,
                        "mask": None,
                        "histogram": None,
                        "uhistogram": None,
                        "weights": None,
                        "uweights": None,
                        "span_ranges": None,
                        "tth_min_max": None,
                        "tth_range": None,
                        "dummyval": None,
                        "dummyval_delta": None}
        self._cl_kernel_args = {"uimemset2": [],
                                "create_histo_binarray": [],
                                "imemset": [],
                                "ui2f2": [],
                                "get_spans": [],
                                "group_spans": [],
                                "solidangle_correction": [],
                                "dummyval_correction": []}
        self._cl_program = None
        self._ctx = None
        self._queue = None
        self.do_solidangle = None
        self.do_dummy = None
        self.do_mask = None
        self.do_dark = None
        self.useTthRange = None

    def __dealloc__(self):
        self.tth_out = None
        self._queue.finish()
        self._free_buffers()
        self._free_kernels()
        self._cl_program = None
        self._queue = None
        self._ctx = None

    def __repr__(self):
        return os.linesep.join(
            [("PyOpenCL implementation of ocl_xrpd1d.ocl_xrpd1D_fullsplit"
              " C++ class. Logging in %s") % self.filename,
             ("device: %s, platform %s device %s"
              " 64bits:%s image size: %s histogram size: %s") %
             (self.devicetype, self.platformid, self.deviceid,
              self.useFp64, self.nData, self.nBins)])

    def log(self, **kwarg):
        """
        log in a file all opencl events
        """
        if self.logfile:
            for key, event in kwarg.items():
                # if event is an event
                event.wait()
                self.logfile.write(
                    " %s: %.3fms\t" %
                    (key, (1e-6 * (event.profile.END - event.profile.START))))
            self.logfile.write(os.linesep)
            self.logfile.flush()

    def _allocate_buffers(self):
        """
        Allocate OpenCL buffers required for a specific configuration

        allocate_CL_buffers() is a private method and is called by
        configure().  Given the size of the image and the number of
        the bins, all the required OpenCL buffers are allocated.

        The method performs a basic check to see if the memory
        required by the configuration is smaller than the total global
        memory of the device. However, there is no built-in way in
        OpenCL to check the real available memory.

        In the case allocate_CL_buffers fails while allocating
        buffers, it will automatically deallocate the buffers that did
        not fail and leave the flag hasBuffers to 0.

        Note that an OpenCL context also requires some memory, as well
        as Event and other OpenCL functionalities which cannot and are
        not taken into account here.

        The memory required by a context varies depending on the
        device. Typical for GTX580 is 65Mb but for a 9300m is ~15Mb

        In addition, a GPU will always have at least 3-5Mb of memory
        in use.

        Unfortunately, OpenCL does NOT have a built-in way to check
        the actual free memory on a device, only the total memory.
        """
        utype = None
        if self.useFp64:
            utype = numpy.int64
        else:
            utype = numpy.int32

        buffers = [
            ("tth", mf.READ_ONLY, numpy.float32, self.nData),
            ("tth_delta", mf.READ_ONLY, numpy.float32, self.nData),
            ("tth_min_max", mf.READ_ONLY, numpy.float32, 2),
            ("tth_range", mf.READ_ONLY, numpy.float32, 2),
            ("mask", mf.READ_ONLY, numpy.int32, self.nData),
            ("image", mf.READ_ONLY, numpy.float32, self.nData),
            ("solidangle", mf.READ_ONLY, numpy.float32, self.nData),
            ("dark", mf.READ_ONLY, numpy.float32, self.nData),
            ("histogram", mf.READ_WRITE, numpy.float32, self.nBins),
            ("uhistogram", mf.READ_WRITE, utype, self.nBins),
            ("weights", mf.READ_WRITE, numpy.float32, self.nBins),
            ("uweights", mf.READ_WRITE, utype, self.nBins),
            ("span_ranges", mf.READ_WRITE, numpy.float32, self.nData),
            ("dummyval", mf.READ_ONLY, numpy.float32, 1),
            ("dummyval_delta", mf.READ_ONLY, numpy.float32, 1),
        ]

        if self.nData < self.BLOCK_SIZE:
            raise RuntimeError(("Fatal error in allocate_CL_buffers."
                                " nData (%d) must be >= BLOCK_SIZE (%d)\n"),
                               self.nData, self.BLOCK_SIZE)

        # TODO it seems that the device should be a member parameter
        # like in the most recent code
        device = ocl.platforms[self.platformid].devices[self.deviceid]
        self._cl_mem = allocate_cl_buffers(buffers, device, self._ctx)

    def _free_buffers(self):
        """
        free all memory allocated on the device
        """
        self._cl_mem = release_cl_buffers(self._cl_mem)

    def _free_kernels(self):
        """
        free all kernels
        """
        for kernel in self._cl_kernel_args:
            self._cl_kernel_args[kernel] = []
        self._cl_program = None

    def _compile_kernels(self, kernel_file=None):
        """
        Compile the kernel

        :param kernel_file: filename of the kernel (to test other kernels)
        """
        kernel_file = kernel_file or "pyfai:openCL/ocl_azim_kernel_2.cl"
        kernel_src = concatenate_cl_kernel([kernel_file])

        template_options = "-D BLOCK_SIZE=%i  -D BINS=%i -D NN=%i"
        compile_options = template_options % (self.BLOCK_SIZE, self.nBins, self.nData)

        if self.useFp64:
            compile_options += " -D ENABLE_FP64"

        try:
            default_compiler_options = self.get_compiler_options(x87_volatile=True)
        except AttributeError:  # Silx version too old
            logger.warning("Please upgrade to silx v0.10+")
            default_compiler_options = get_x87_volatile_option(self._ctx)

        if default_compiler_options:
            compile_options += " " + default_compiler_options

        try:
            self._cl_program = pyopencl.Program(self._ctx, kernel_src)
            self._cl_program.build(options=compile_options)
        except pyopencl.MemoryError as error:
            raise MemoryError(error)

    def _set_kernel_arguments(self):
        """Tie arguments of OpenCL kernel-functions to the actual kernels

        set_kernel_arguments() is a private method, called by configure().
        It uses the dictionary _cl_kernel_args.

        Note that by default, since TthRange is disabled, the
        integration kernels have tth_min_max tied to the tthRange
        argument slot.

        When setRange is called it replaces that argument with
        tthRange low and upper bounds. When unsetRange is called, the
        argument slot is reset to tth_min_max.
        """
        self._cl_kernel_args["create_histo_binarray"] = \
            [self._cl_mem[i] for i in ("tth", "tth_delta", "uweights",
                                       "tth_min_max", "image", "uhistogram",
                                       "span_ranges", "mask", "tth_min_max")]
        self._cl_kernel_args["get_spans"] = \
            [self._cl_mem[i] for i in ["tth", "tth_delta",
                                       "tth_min_max", "span_ranges"]]
        self._cl_kernel_args["solidangle_correction"] = \
            [self._cl_mem["image"], self._cl_mem["solidangle"]]
        self._cl_kernel_args["dummyval_correction"] = \
            [self._cl_mem["image"], self._cl_mem["dummyval"],
             self._cl_mem["dummyval_delta"]]
        self._cl_kernel_args["uimemset2"] = \
            [self._cl_mem["uweights"], self._cl_mem["uhistogram"]]
        self._cl_kernel_args["imemset"] = [self._cl_mem["mask"], ]
        self._cl_kernel_args["ui2f2"] = \
            [self._cl_mem[i] for i in ("uweights", "uhistogram",
                                       "weights", "histogram")]

    def _calc_tth_out(self, lower, upper):
        """
        Calculate the bin-center position in 2theta
        """
        self.tth_min = numpy.float32(lower)
        self.tth_max = numpy.float32(upper)
        delta = (upper - lower) / (self.nBins)
        self.tth_out = numpy.linspace(lower + 0.5 * delta,
                                      upper - 0.5 * delta,
                                      self.nBins)

    def getConfiguration(self, Nimage, Nbins, useFp64=None):
        """getConfiguration gets the description of the integrations
        to be performed and keeps an internal copy

        :param Nimage: number of pixel in image
        :param Nbins: number of bins in regrouped histogram
        :param useFp64: use double precision. By default the same as init!
        """
        if Nimage < 1 or Nbins < 1:
            raise RuntimeError(("getConfiguration with Nimage=%s and"
                                " Nbins=%s makes no sense") % (Nimage, Nbins))
        if useFp64 is not None:
            self.useFp64 = bool(useFp64)
        self.nBins = Nbins
        self.nData = Nimage
        self.wdim_data = (Nimage + self.BLOCK_SIZE - 1) & \
            ~ (self.BLOCK_SIZE - 1),
        self.wdim_bins = (Nbins + self.BLOCK_SIZE - 1) & \
            ~ (self.BLOCK_SIZE - 1),

    def configure(self, kernel=None):
        """
        The method configure() allocates the OpenCL resources required
        and compiled the OpenCL kernels.  An active context must exist
        before a call to configure() and getConfiguration() must have
        been called at least once. Since the compiled OpenCL kernels
        carry some information on the integration parameters, a change
        to any of the parameters of getConfiguration() requires a
        subsequent call to configure() for them to take effect.

        If a configuration exists and configure() is called, the
        configuration is cleaned up first to avoid OpenCL memory leaks

        :param kernel_path: is the path to the actual kernel
        """
        if self.nBins < 1 or self.nData < 1:
            raise RuntimeError(("configure() with Nimage=%s and"
                                " Nbins=%s makes no sense") %
                               (self.nData, self.nBins))
        if not self._ctx:
            raise RuntimeError("You may not call config() at this point."
                               " There is no Active context."
                               " (Hint: run init())")

        # If configure is recalled, force cleanup of OpenCL resources
        # to avoid accidental leaks
        self.clean(True)
        with self.lock:
            self._allocate_buffers()
            self._compile_kernels(kernel)
            self._set_kernel_arguments()
            # We need to initialise the Mask to 0
            imemset = self._cl_program.imemset(self._queue, self.wdim_data,
                                               self.tdim, self._cl_mem["mask"])
        if self.logfile:
            self.log(memset_mask=imemset)

    def loadTth(self, tth, dtth, tth_min=None, tth_max=None):
        """
        Load the 2th arrays along with the min and max value.

        loadTth maybe be recalled at any time of the execution in
        order to update the 2th arrays.

        loadTth is required and must be called at least once after a
        configure()
        """

        if not self._ctx:
            raise RuntimeError("You may not call loadTth() at this point."
                               " There is no Active context."
                               " (Hint: run init())")
        if not self._cl_mem["tth"]:
            raise RuntimeError("You may not call loadTth() at this point,"
                               " OpenCL is not configured"
                               " (Hint: run configure())")

        ctth = numpy.ascontiguousarray(tth.ravel(), dtype=numpy.float32)
        cdtth = numpy.ascontiguousarray(dtth.ravel(), dtype=numpy.float32)
        with self.lock:
            self._tth_max = (ctth + cdtth).max() * \
                (1.0 + numpy.finfo(numpy.float32).eps)
            self._tth_min = max(0.0, (ctth - cdtth).min())
            if tth_min is None:
                tth_min = self._tth_min

            if tth_max is None:
                tth_max = self._tth_max
            copy_tth = pyopencl.enqueue_copy(self._queue,
                                             self._cl_mem["tth"], ctth)
            copy_dtth = pyopencl.enqueue_copy(self._queue,
                                              self._cl_mem["tth_delta"], cdtth)
            pyopencl.enqueue_copy(self._queue, self._cl_mem["tth_min_max"],
                                  numpy.array((self._tth_min, self._tth_max),
                                              dtype=numpy.float32))
            logger.debug("kernel get_spans sizes: \t%s %s",
                         self.wdim_data, self.tdim)
            get_spans = \
                self._cl_program.get_spans(self._queue, self.wdim_data,
                                           self.tdim,
                                           *self._cl_kernel_args["get_spans"])
            # Group 2th span ranges group_spans(__global float *span_range)
            logger.debug("kernel group_spans sizes: \t%s %s",
                         self.wdim_data, self.tdim)
            group_spans = \
                self._cl_program.group_spans(self._queue,
                                             self.wdim_data,
                                             self.tdim,
                                             self._cl_mem["span_ranges"])
            self._calc_tth_out(tth_min, tth_max)
        if self.logfile:
            self.log(copy_2th=copy_tth, copy_delta2th=copy_dtth,
                     get_spans=get_spans, group_spans=group_spans)

    def setSolidAngle(self, solidAngle):
        """
        Enables SolidAngle correction and uploads the suitable array
        to the OpenCL device.

        By default the program will assume no solidangle correction
        unless setSolidAngle() is called.  From then on, all
        integrations will be corrected via the SolidAngle array.

        If the SolidAngle array needs to be changes, one may just call
        setSolidAngle() again with that array

        :param solidAngle: the solid angle of the given pixel
        :type solidAngle: ndarray
        """
        if not self._ctx:
            raise RuntimeError("You may not call Integrator1d.setSolidAngle()"
                               " at this point. There is no Active context."
                               " (Hint: run init())")
        cSolidANgle = numpy.ascontiguousarray(solidAngle.ravel(),
                                              dtype=numpy.float32)
        with self.lock:
            self.do_solidangle = True
            copy_solidangle = pyopencl.enqueue_copy(self._queue,
                                                    self._cl_mem["solidangle"],
                                                    cSolidANgle)
        if self.logfile:
            self.log(copy_solidangle=copy_solidangle)

    def unsetSolidAngle(self):
        """
        Instructs the program to not perform solidangle correction from now on.

        SolidAngle correction may be turned back on at any point
        """
        with self.lock:
            self.do_solidangle = False

    def setMask(self, mask):
        """
        Enables the use of a Mask during integration. The Mask can be
        updated by recalling setMask at any point.

        The Mask must be a PyFAI Mask. Pixels with 0 are masked
        out. TODO: check and invert!

        :param mask: numpy.ndarray of integer.
        """
        if not self._ctx:
            raise RuntimeError("You may not call"
                               " Integrator1d.setDummyValue(dummy,delta_dummy)"
                               " at this point. There is no Active context."
                               " (Hint: run init())")
        cMask = numpy.ascontiguousarray(mask.ravel(), dtype=numpy.int32)
        with self.lock:
            self.do_mask = True
            copy_mask = pyopencl.enqueue_copy(self._queue,
                                              self._cl_mem["mask"], cMask)
        if self.logfile:
            self.log(copy_mask=copy_mask)

    def unsetMask(self):
        """
        Disables the use of a Mask from that point.
        It may be re-enabled at any point via setMask
        """
        with self.lock:
            self.do_mask = False

    def setDummyValue(self, dummy, delta_dummy):
        """
        Enables dummy value functionality and uploads the value to the
        OpenCL device.

        Image values that are similar to the dummy value are set to 0.

        :param dummy: value in image of missing values (masked pixels?)
        :param delta_dummy: precision for dummy values
        """
        if not self._ctx:
            raise RuntimeError("You may not call"
                               " Integrator1d.setDummyValue(dummy,delta_dummy)"
                               " at this point. There is no Active context."
                               " (Hint: run init())")
        else:
            with self.lock:
                self.do_dummy = True
                pyopencl.enqueue_copy(self._queue, self._cl_mem["dummyval"],
                                      numpy.array((dummy,),
                                                  dtype=numpy.float32))
                pyopencl.enqueue_copy(self._queue,
                                      self._cl_mem["dummyval_delta"],
                                      numpy.array((delta_dummy,),
                                                  dtype=numpy.float32))

    def unsetDummyValue(self):
        """Disable a dummy value.
        May be re-enabled at any time by setDummyValue
        """
        with self.lock:
            self.do_dummy = False

    def setRange(self, lowerBound, upperBound):
        """
        Instructs the program to use a user - defined range for 2th
        values

        setRange is optional. By default the integration will use the
        tth_min and tth_max given by loadTth() as integration
        range. When setRange is called it sets a new integration range
        without affecting the 2th array. All values outside that range
        will then be discarded when interpolating.  Currently, if the
        interval of 2th (2th + -d2th) is not all inside the range
        specified, it is discarded. The bins of the histogram are
        RESCALED to the defined range and not the original tth_max -
        tth_min range.

        setRange can be called at any point and as many times required
        after a valid configuration is created.

        :param lowerBound: lower bound of the integration range
        :type lowerBound: float
        :param upperBound: upper bound of the integration range
        :type upperBound: float
        """
        if self._ctx is None:
            raise RuntimeError("You may not call setRange() at this point."
                               " There is no Active context."
                               " (Hint: run init())")
        if not (self.nData > 1 and self._cl_mem["tth_range"]):
            raise RuntimeError("You may not call setRange() at this point,"
                               " the required buffers are not allocated"
                               " (Hint: run config())")

        with self.lock:
            self.useTthRange = True
            copy_2thrange = \
                pyopencl.enqueue_copy(self._queue, self._cl_mem["tth_range"],
                                      numpy.array((lowerBound, upperBound),
                                                  dtype=numpy.float32))
            self._cl_kernel_args["create_histo_binarray"][8] = \
                self._cl_mem["tth_range"]
            self._cl_kernel_args["get_spans"][2] = self._cl_mem["tth_range"]

        if self.logfile:
            self.log(copy_2thrange=copy_2thrange)

    def unsetRange(self):
        """
        Disable the use of a user-defined 2th range and revert to
        tth_min,tth_max range

        unsetRange instructs the program to revert to its default
        integration range. If the method is called when no
        user-defined range had been previously specified, no action
        will be performed
        """

        with self.lock:
            if self.useTthRange:
                self._calc_tth_out(self._tth_min, self._tth_max)
            self.useTthRange = False
            self._cl_kernel_args["create_histo_binarray"][8] = \
                self._cl_mem["tth_min_max"]
            self._cl_kernel_args["get_spans"][2] = self._cl_mem["tth_min_max"]

    def execute(self, image):
        """
        Perform a 1D azimuthal integration

        execute() may be called only after an OpenCL device is
        configured and a Tth array has been loaded (at least once) It
        takes the input image and based on the configuration provided
        earlier it performs the 1D integration.  Notice that if the
        provided image is bigger than N then only N points will be
        taked into account, while if the image is smaller than N the
        result may be catastrophic.  set/unset and loadTth methods
        have a direct impact on the execute() method.  All the rest of
        the methods will require at least a new configuration via
        configure().

        Takes an image, integrate and return the histogram and weights

        :param image: image to be processed as a numpy array
        :return: tth_out, histogram, bins

        TODO: to improve performances, the image should be casted to
        float32 in an optimal way: currently using numpy machinery but
        would be better if done in OpenCL
        """
        assert image.size == self.nData
        if not self._ctx:
            raise RuntimeError("You may not call execute() at this point."
                               " There is no Active context."
                               " (Hint: run init())")
        if not self._cl_mem["histogram"]:
            raise RuntimeError("You may not call execute() at this point,"
                               " kernels are not configured"
                               " (Hint: run configure())")
        if not self._tth_max:
            raise RuntimeError("You may not call execute() at this point."
                               " There is no 2th array loaded."
                               " (Hint: run loadTth())")

        with self.lock:
            copy_img = pyopencl.enqueue_copy(
                self._queue,
                self._cl_mem["image"],
                numpy.ascontiguousarray(image.ravel(),
                                        dtype=numpy.float32))
            logger.debug("kernel uimemset2 sizes: \t%s %s",
                         self.wdim_bins, self.tdim)
            memset = \
                self._cl_program.uimemset2(self._queue, self.wdim_bins,
                                           self.tdim,
                                           *self._cl_kernel_args["uimemset2"])

            if self.do_dummy:
                logger.debug("kernel dummyval_correction sizes: \t%s %s",
                             self.wdim_data, self.tdim)
                dummy = self._cl_program.dummyval_correction(
                    self._queue, self.wdim_data, self.tdim,
                    self._cl_kernel_args["dummyval_correction"])

            if self.do_solidangle:
                sa = self._cl_program.solidangle_correction(
                    self._queue, self.wdim_data, self.tdim,
                    *self._cl_kernel_args["solidangle_correction"])
            logger.debug("kernel create_histo_binarray sizes: \t%s %s",
                         self.wdim_data, self.tdim)
            integrate = self._cl_program.create_histo_binarray(
                self._queue, self.wdim_data, self.tdim,
                *self._cl_kernel_args["create_histo_binarray"])
            # convert to float
            convert = self._cl_program.ui2f2(self._queue, self.wdim_data,
                                             self.tdim,
                                             *self._cl_kernel_args["ui2f2"])
            histogram = numpy.empty(self.nBins, dtype=numpy.float32)
            bins = numpy.empty(self.nBins, dtype=numpy.float32)
            copy_hist = pyopencl.enqueue_copy(self._queue, histogram,
                                              self._cl_mem["histogram"])
            copy_bins = pyopencl.enqueue_copy(self._queue, bins,
                                              self._cl_mem["weights"])

            if self.logfile:
                self.log(copy_in=copy_img, memset2=memset)
                if self.do_dummy:
                    self.log(dummy_corr=dummy)
                if self.do_solidangle:
                    self.log(solid_angle=sa)
                self.log(integrate=integrate, convert_uint2float=convert,
                         copy_hist=copy_hist, copy_bins=copy_bins)
            copy_bins.wait()
        return self.tth_out, histogram, bins

    def init(self, devicetype="GPU", useFp64=True,
             platformid=None, deviceid=None):
        """Initial configuration: Choose a device and initiate a
        context.  Devicetypes can be GPU, gpu, CPU, cpu, DEF, ACC,
        ALL. Suggested are GPU,CPU. For each setting to work there
        must be such an OpenCL device and properly installed. E.g.: If
        Nvidia driver is installed, GPU will succeed but CPU will
        fail. The AMD SDK kit (AMD APP) is required for CPU via
        OpenCL.

        :param devicetype: string in ["cpu","gpu", "all", "acc"]
        :param useFp64: boolean specifying if double precision will be used
        :param platformid: integer
        :param devid: integer
        """
        if self._ctx is None:
            self._ctx = ocl.create_context(devicetype, useFp64,
                                           platformid, deviceid)
            device = self._ctx.devices[0]

            self.devicetype = pyopencl.device_type.to_string(device.type)
            if (self.devicetype == "CPU")\
                    and (device.platform.vendor == "Apple"):
                logger.warning("This is a workaround for Apple's OpenCL"
                               " on CPU: enforce BLOCK_SIZE=1")
                self.BLOCK_SIZE = 1
                self.tdim = (self.BLOCK_SIZE,)

                if self.nBins:
                    self.wdim_bins = (self.nBins + self.BLOCK_SIZE - 1) & \
                        ~ (self.BLOCK_SIZE - 1),
                if self.nData:
                    self.wdim_data = (self.nData + self.BLOCK_SIZE - 1) & \
                        ~ (self.BLOCK_SIZE - 1),

            self.useFp64 = "fp64" in device.extensions
            platforms = pyopencl.get_platforms()
            self.platformid = platforms.index(device.platform)
            devices = platforms[self.platformid].get_devices()
            self.deviceid = devices.index(device)
            if self.filename:
                self._queue = pyopencl.CommandQueue(
                    self._ctx,
                    properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
            else:
                self._queue = pyopencl.CommandQueue(self._ctx)
        else:
            logger.warning("Recycling existing context ..."
                           " if you want to get start from scratch,"
                           " use clean()")

    def clean(self, preserve_context=False):
        """
        Free OpenCL related resources allocated by the library.

        clean() is used to reinitiate the library back in a vanilla
        state.  It may be asked to preserve the context created by
        init or completely clean up OpenCL. Guard/Status flags that
        are set will be reset.

        :param preserve_context: preserves or destroys all OpenCL resources
        :type preserve_context: bool
        """

        with self.lock:
            self._free_buffers()
            self._free_kernels()
            if not preserve_context:
                self._queue = None
                self._ctx = None

    def get_status(self):
        """return a dictionnary with the status of the integrator: for
        compatibilty with former implementation"""
        out = {'dummy': bool(self.do_dummy),
               'mask': bool(self.do_mask),
               'dark': bool(self.do_dark),
               "solid_angle": bool(self.do_solidangle),
               "pos1": False,
               'pos0': (self._tth_max is not None),
               'compiled': (self._cl_program is not None),
               'size': self.nData,
               'context': (self._ctx is not None)}
        return out


class OCL_Histogram1d(OpenclProcessing):
    """Class in charge of performing histogram calculation in OpenCL using
    atomic_add

    It also performs the preprocessing using the preproc kernel
    """
    BLOCK_SIZE = 32
    buffers = [BufferDescription("output4", 4, numpy.float32, mf.READ_WRITE),
               BufferDescription("position", 1, numpy.float32, mf.READ_ONLY),
               BufferDescription("image_raw", 1, numpy.float32, mf.READ_ONLY),
               BufferDescription("image", 1, numpy.float32, mf.READ_WRITE),
               BufferDescription("variance", 1, numpy.float32, mf.READ_WRITE),
               BufferDescription("dark", 1, numpy.float32, mf.READ_WRITE),
               BufferDescription("dark_variance", 1, numpy.float32, mf.READ_ONLY),
               BufferDescription("flat", 1, numpy.float32, mf.READ_ONLY),
               BufferDescription("polarization", 1, numpy.float32, mf.READ_ONLY),
               BufferDescription("solidangle", 1, numpy.float32, mf.READ_ONLY),
               BufferDescription("absorption", 1, numpy.float32, mf.READ_ONLY),
               BufferDescription("mask", 1, numpy.int8, mf.READ_ONLY),
               ]
    kernel_files = ["pyfai:openCL/kahan.cl",
                    "pyfai:openCL/preprocess.cl",
                    "pyfai:openCL/ocl_histo.cl"
                    ]
    mapping = {numpy.int8: "s8_to_float",
               numpy.uint8: "u8_to_float",
               numpy.int16: "s16_to_float",
               numpy.uint16: "u16_to_float",
               numpy.uint32: "u32_to_float",
               numpy.int32: "s32_to_float"
               }

    def __init__(self, position, bins, checksum=None, empty=None, unit=None,
                 ctx=None, devicetype="all", platformid=None, deviceid=None,
                 block_size=None, profile=False):
        """
        :param position: array with the radial position of every single pixel. Same as image size
        :param bins: number of bins on which to histogram
        :param checksum: pre-calculated checksum of the position array to prevent re-calculating it :)
        :param empty: value to be assigned to bins without contribution from any pixel
        :param unit: just a place_holder for the units of position.
        :param ctx: actual working context, left to None for automatic
                    initialization from device type or platformid/deviceid
        :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
        :param platformid: integer with the platform_identifier, as given by clinfo
        :param deviceid: Integer with the device identifier, as given by clinfo
        :param block_size: preferred workgroup size, may vary depending on the outpcome of the compilation
        :param profile: switch on profiling to be able to profile at the kernel level,
                        store profiling elements (makes code slightly slower)
        """
        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
                                  platformid=platformid, deviceid=deviceid,
                                  block_size=block_size, profile=profile)
        if "cl_khr_int64_base_atomics" not in self.ctx.devices[0].extensions:
            logger.warning("Apparently 64-bit atomics are missing on device %s, "
                           "possibly falling back on 32-bit atomics (loss of precision)"
                           " but it can be present and not declared as Nvidia does",
                           self.ctx.devices[0].name)
        self.unit = unit
        self.bins = numpy.uint32(bins)
        self.size = numpy.uint32(position.size)
        self.empty = numpy.float32(empty) if empty is not None else numpy.float32(0.0)
        self.mini = numpy.float32(numpy.min(position))
        self.maxi = numpy.float32(numpy.max(position) * (1.0 + numpy.finfo(numpy.float32).eps))

        if not checksum:
            checksum = calc_checksum(position)
        self.position = position
        self.on_device = {"position": checksum,
                          "dark": None,
                          "flat": None,
                          "polarization": None,
                          "solidangle": None,
                          "absorption": None}

        if block_size is None:
            block_size = self.BLOCK_SIZE

        self.BLOCK_SIZE = min(block_size, self.device.max_work_group_size)
        self.workgroup_size = {}
        self.wdim_bins = (self.bins + self.BLOCK_SIZE - 1) & ~(self.BLOCK_SIZE - 1),
        self.wdim_data = (self.size + self.BLOCK_SIZE - 1) & ~(self.BLOCK_SIZE - 1),

        self.buffers = [BufferDescription(i.name, i.size * self.size, i.dtype, i.flags)
                        for i in self.__class__.buffers]

        self.buffers += [BufferDescription("histo_sig", self.bins * 2, numpy.float32, mf.READ_WRITE),
                         BufferDescription("histo_var", self.bins * 2, numpy.float32, mf.READ_WRITE),
                         BufferDescription("histo_nrm", self.bins * 2, numpy.float32, mf.READ_WRITE),
                         BufferDescription("histo_cnt", self.bins, numpy.uint32, mf.READ_WRITE),
                         BufferDescription("error", self.bins, numpy.float32, mf.WRITE_ONLY),
                         BufferDescription("intensity", self.bins, numpy.float32, mf.WRITE_ONLY),
                         ]
        try:
            self.set_profiling(profile)
            self.allocate_buffers()
            self.compile_kernels()
            self.set_kernel_arguments()
        except (pyopencl.MemoryError, pyopencl.LogicError) as error:
            raise MemoryError(error)
        self.send_buffer(position, "position", checksum)

    def __copy__(self):
        """Shallow copy of the object

        :return: copy of the object
        """
        return self.__class__(self.position,
                              self.bins,
                              checksum=self.on_device.get("position"),
                              empty=self.empty,
                              ctx=self.ctx,
                              block_size=self.block_size,
                              profile=self.profile)

    def __deepcopy__(self, memo=None):
        """deep copy of the object

        :return: deepcopy of the object
        """
        if memo is None:
            memo = {}
        position = self.position.copy()
        memo[id(self.position)] = position
        new_obj = self.__class__(position, self.bins,
                                 checksum=self.on_device.get("data"),
                                 empty=self.empty,
                                 ctx=self.ctx,
                                 block_size=self.block_size,
                                 profile=self.profile)
        memo[id(self)] = new_obj
        return new_obj

    def compile_kernels(self, kernel_file=None):
        """Call the OpenCL compiler

        :param kernel_file: path to the kernel (by default use the one in the resources directory)
        """
        # concatenate all needed source files into a single openCL module
        kernel_file = kernel_file or self.kernel_files[-1]
        kernels = self.kernel_files[:-1] + [kernel_file]

        try:
            default_compiler_options = self.get_compiler_options(x87_volatile=True)
        except AttributeError:  # Silx version too old
            logger.warning("Please upgrade to silx v0.10+")
            default_compiler_options = get_x87_volatile_option(self.ctx)
        compile_options = "-D NBINS=%i  -D NIMAGE=%i -D WORKGROUP_SIZE=%i" % \
                          (self.bins, self.size, self.BLOCK_SIZE)
        if default_compiler_options:
            compile_options += " " + default_compiler_options
        try:
            OpenclProcessing.compile_kernels(self, kernels, compile_options)
        except Exception as error:
            # This error may be related to issue #1219. Provides an ugly work around.
            if "cl_khr_int64_base_atomics" in self.ctx.devices[0].extensions:
                # Maybe this extension actually does not existe!
                OpenclProcessing.compile_kernels(self, ["pyfai:openCL/deactivate_atomic64.cl"] + kernels, compile_options)
                logger.warning("Your OpenCL compiler wrongly claims it support 64-bit atomics. Degrading to 32 bits atomics!")
            else:
                logger.error("Failed to compile kernel ! Check the compiler. %s", error)

        for kernel_name, kernel in self.kernels.get_kernels().items():
            wg = kernel_workgroup_size(self.program, kernel)
            self.workgroup_size[kernel_name] = (min(wg, self.BLOCK_SIZE),)  # this is a tuple

    def set_kernel_arguments(self):
        """Tie arguments of OpenCL kernel-functions to the actual kernels

        """
        self.cl_kernel_args["corrections4"] = OrderedDict((("image", self.cl_mem["image"]),
                                                           ("variance", self.cl_mem["variance"]),
                                                           ("do_dark", numpy.int8(0)),
                                                           ("dark", self.cl_mem["dark"]),
                                                           ("do_dark_variance", numpy.int8(0)),
                                                           ("dark_variance", self.cl_mem["dark_variance"]),
                                                           ("do_flat", numpy.int8(0)),
                                                           ("flat", self.cl_mem["flat"]),
                                                           ("do_solidangle", numpy.int8(0)),
                                                           ("solidangle", self.cl_mem["solidangle"]),
                                                           ("do_polarization", numpy.int8(0)),
                                                           ("polarization", self.cl_mem["polarization"]),
                                                           ("do_absorption", numpy.int8(0)),
                                                           ("absorption", self.cl_mem["absorption"]),
                                                           ("do_mask", numpy.int8(0)),
                                                           ("mask", self.cl_mem["mask"]),
                                                           ("do_dummy", numpy.int8(0)),
                                                           ("dummy", numpy.float32(0)),
                                                           ("delta_dummy", numpy.float32(0)),
                                                           ("normalization_factor", numpy.float32(1.0)),
                                                           ("preproc4", self.cl_mem["output4"])))

        self.cl_kernel_args["histogram_1d_preproc"] = OrderedDict((("position", self.cl_mem["position"]),
                                                                   ("preproc4", self.cl_mem["output4"]),
                                                                   ("histo_sig", self.cl_mem["histo_sig"]),
                                                                   ("histo_var", self.cl_mem["histo_var"]),
                                                                   ("histo_nrm", self.cl_mem["histo_nrm"]),
                                                                   ("histo_cnt", self.cl_mem["histo_cnt"]),
                                                                   ("size", self.size),
                                                                   ("bins", self.bins),
                                                                   ("mini", self.mini),
                                                                   ("maxi", self.maxi)))

        self.cl_kernel_args["histogram_postproc"] = OrderedDict((("histo_sig", self.cl_mem["histo_sig"]),
                                                                 ("histo_var", self.cl_mem["histo_var"]),
                                                                 ("histo_nrm", self.cl_mem["histo_nrm"]),
                                                                 ("histo_cnt", self.cl_mem["histo_cnt"]),
                                                                 ("bins", self.bins),
                                                                 ("empty", self.empty),
                                                                 ("intensity", self.cl_mem["intensity"]),
                                                                 ("error", self.cl_mem["error"])))

        self.cl_kernel_args["memset_histograms"] = OrderedDict((("histo_sig", self.cl_mem["histo_sig"]),
                                                                ("histo_var", self.cl_mem["histo_var"]),
                                                                ("histo_nrm", self.cl_mem["histo_nrm"]),
                                                                ("histo_cnt", self.cl_mem["histo_cnt"]),
                                                                ("bins", self.bins)))
        self.cl_kernel_args["u8_to_float"] = OrderedDict(((i, self.cl_mem[i]) for i in ("image_raw", "image")))
        self.cl_kernel_args["s8_to_float"] = OrderedDict(((i, self.cl_mem[i]) for i in ("image_raw", "image")))
        self.cl_kernel_args["u16_to_float"] = OrderedDict(((i, self.cl_mem[i]) for i in ("image_raw", "image")))
        self.cl_kernel_args["s16_to_float"] = OrderedDict(((i, self.cl_mem[i]) for i in ("image_raw", "image")))
        self.cl_kernel_args["u32_to_float"] = OrderedDict(((i, self.cl_mem[i]) for i in ("image_raw", "image")))
        self.cl_kernel_args["s32_to_float"] = OrderedDict(((i, self.cl_mem[i]) for i in ("image_raw", "image")))

    def send_buffer(self, data, dest, checksum=None):
        """Send a numpy array to the device, including the cast on the device if possible

        :param data: numpy array with data
        :param dest: name of the buffer as registered in the class
        """

        dest_type = numpy.dtype([i.dtype for i in self.buffers if i.name == dest][0])
        events = []
        if isinstance(data, pyopencl.array.Array):
            if (data.dtype == dest_type):
                copy_image = pyopencl.enqueue_copy(self.queue, self.cl_mem[dest], data.data)
                events.append(EventDescription("copy D->D %s" % dest, copy_image))
            else:
                copy_image = pyopencl.enqueue_copy(self.queue, self.cl_mem["image_raw"], data.data)
                kernel_name = self.mapping[data.dtype.type]
                kernel = self.kernels.get_kernel(kernel_name)
                cast_to_float = kernel(self.queue, (self.size,), None, self.cl_mem["image_raw"], self.cl_mem[dest])
                events += [EventDescription("copy raw D->D " + dest, copy_image),
                           EventDescription("cast " + kernel_name, cast_to_float)]
        else:
            # Assume it is a numpy array
            if (data.dtype == dest_type) or (data.dtype.itemsize > dest_type.itemsize):
                copy_image = pyopencl.enqueue_copy(self.queue, self.cl_mem[dest], numpy.ascontiguousarray(data, dest_type))
                events.append(EventDescription("copy H->D %s" % dest, copy_image))
            else:
                copy_image = pyopencl.enqueue_copy(self.queue, self.cl_mem["image_raw"], numpy.ascontiguousarray(data))
                kernel_name = self.mapping[data.dtype.type]
                kernel = self.kernels.get_kernel(kernel_name)
                cast_to_float = kernel(self.queue, (self.size,), None, self.cl_mem["image_raw"], self.cl_mem[dest])
                events += [EventDescription("copy raw H->D " + dest, copy_image),
                           EventDescription("cast " + kernel_name, cast_to_float)]
        if self.profile:
            self.events += events
        if checksum is not None:
            self.on_device[dest] = checksum

    def integrate(self, data, dark=None,
                  dummy=None, delta_dummy=None,
                  variance=None, dark_variance=None,
                  flat=None, solidangle=None, polarization=None, absorption=None,
                  dark_checksum=None, flat_checksum=None, solidangle_checksum=None,
                  polarization_checksum=None, absorption_checksum=None,
                  preprocess_only=False, safe=True,
                  normalization_factor=1.0, bin_range=None,
                  histo_signal=None, histo_variance=None,
                  histo_normalization=None, histo_count=None,
                  intensity=None, error=None):
        """
        Performing azimuthal integration, the preprocessing is:

        .. math::

            Signal= (data - dark)
            Variance= (variance + dark_variance)
            Normalization= (normalization_factor * flat * solidangle * polarization * absorption)
            Count= 1 per valid pixel

        Integration is performed using the histograms (based on atomic adds

        :param dark: array of same shape as data for pre-processing
        :param dummy: value for invalid data
        :param delta_dummy: precesion for dummy assessement
        :param variance: array of same shape as data for pre-processing
        :param dark_variance: array of same shape as data for pre-processing
        :param flat: array of same shape as data for pre-processing
        :param solidangle: array of same shape as data for pre-processing
        :param polarization: array of same shape as data for pre-processing
        :param dark_checksum: CRC32 checksum of the given array
        :param flat_checksum: CRC32 checksum of the given array
        :param solidangle_checksum: CRC32 checksum of the given array
        :param polarization_checksum: CRC32 checksum of the given array
        :param safe: if True (default) compares arrays on GPU according to their checksum, unless, use the buffer location is used
        :param preprocess_only: return the dark subtracted; flat field & solidangle & polarization corrected image, else
        :param normalization_factor: divide raw signal by this value
        :param bin_range: provide lower and upper bound for position
        :param histo_signal: destination array or pyopencl array for sum of signals
        :param histo_normalization: destination array or pyopencl array for sum of normalization
        :param histo_count: destination array or pyopencl array for counting pixels
        :param intensity: destination PyOpenCL array for integrated intensity
        :param error: destination PyOpenCL array for standart deviation
        :return: bin_positions, averaged data, histogram of signal, histogram of variance, histogram of normalization, count of pixels
        """
        events = []
        with self.sem:
            self.send_buffer(data, "image")
            memset = self.kernels.memset_histograms(self.queue, self.wdim_bins, self.workgroup_size["memset_histograms"],
                                                    *list(self.cl_kernel_args["memset_histograms"].values()))
            events.append(EventDescription("memset_histograms", memset))
            kw_correction = self.cl_kernel_args["corrections4"]
            kw_histogram = self.cl_kernel_args["histogram_1d_preproc"]

            if variance is None:
                kw_correction["variance"] = self.cl_mem["image"]
            else:
                self.send_buffer(variance, "variance")
                kw_correction["variance"] = self.cl_mem["variance"]

            if dummy is not None:
                do_dummy = numpy.int8(1)
                dummy = numpy.float32(dummy)
                if delta_dummy is None:
                    delta_dummy = numpy.float32(0.0)
                else:
                    delta_dummy = numpy.float32(abs(delta_dummy))
            else:
                do_dummy = numpy.int8(0)
                dummy = numpy.float32(self.empty)
                delta_dummy = numpy.float32(0.0)

            kw_correction["do_dummy"] = do_dummy
            kw_correction["dummy"] = dummy
            kw_correction["delta_dummy"] = delta_dummy
            kw_correction["normalization_factor"] = numpy.float32(normalization_factor)

            if dark is not None:
                do_dark = numpy.int8(1)
                # TODO: what is do_checksum=False and image not on device ...
                if not dark_checksum:
                    dark_checksum = calc_checksum(dark, safe)
                if dark_checksum != self.on_device["dark"]:
                    self.send_buffer(dark, "dark", dark_checksum)
            else:
                do_dark = numpy.int8(0)
            kw_correction["do_dark"] = do_dark

            if dark_variance is None:
                kw_correction["do_dark_variance"] = numpy.int8(0)
            else:
                kw_correction["do_dark_variance"] = numpy.int8(1)

            if flat is not None:
                do_flat = numpy.int8(1)
                if not flat_checksum:
                    flat_checksum = calc_checksum(flat, safe)
                if self.on_device["flat"] != flat_checksum:
                    self.send_buffer(flat, "flat", flat_checksum)
            else:
                do_flat = numpy.int8(0)
            kw_correction["do_flat"] = do_flat

            if solidangle is not None:
                do_solidangle = numpy.int8(1)
                if not solidangle_checksum:
                    solidangle_checksum = calc_checksum(solidangle, safe)
                if solidangle_checksum != self.on_device["solidangle"]:
                    self.send_buffer(solidangle, "solidangle", solidangle_checksum)
            else:
                do_solidangle = numpy.int8(0)
            kw_correction["do_solidangle"] = do_solidangle

            if polarization is not None:
                do_polarization = numpy.int8(1)
                if not polarization_checksum:
                    polarization_checksum = calc_checksum(polarization, safe)
                if polarization_checksum != self.on_device["polarization"]:
                    self.send_buffer(polarization, "polarization", polarization_checksum)
            else:
                do_polarization = numpy.int8(0)
            kw_correction["do_polarization"] = do_polarization

            if absorption is not None:
                do_absorption = numpy.int8(1)
                if not absorption_checksum:
                    absorption_checksum = calc_checksum(absorption, safe)
                if absorption_checksum != self.on_device["absorption"]:
                    self.send_buffer(absorption, "absorption", absorption_checksum)
            else:
                do_absorption = numpy.int8(0)
            kw_correction["do_absorption"] = do_absorption

            ev = self.kernels.corrections4(self.queue, self.wdim_data, self.workgroup_size["corrections4"],
                                           *list(kw_correction.values()))
            events.append(EventDescription("corrections4", ev))

            if preprocess_only:
                image = numpy.empty(data.shape + (4,), dtype=numpy.float32)
                ev = pyopencl.enqueue_copy(self.queue, image, self.cl_mem["output"])
                events.append(EventDescription("copy D->H image", ev))
                if self.profile:
                    self.events += events
                ev.wait()
                return image

            if bin_range:
                mini = numpy.float32(min(bin_range))
                maxi = numpy.float32(max(bin_range) * (1.0 + numpy.finfo(numpy.float32).eps))
            else:
                mini = self.mini
                maxi = self.maxi
            kw_histogram["mini"] = mini
            kw_histogram["maxi"] = maxi

            histogram = self.kernels.histogram_1d_preproc(self.queue, self.wdim_data, self.workgroup_size["histogram_1d_preproc"],
                                                          *kw_histogram.values())
            events.append(EventDescription("histogram_1d_preproc", histogram))

            postproc = self.kernels.histogram_postproc(self.queue, self.wdim_bins, self.workgroup_size["histogram_postproc"],
                                                       *self.cl_kernel_args["histogram_postproc"].values())
            events.append(EventDescription("histogram_postproc", postproc))

            if histo_signal is None:
                histo_signal = numpy.empty((self.bins, 2), dtype=numpy.float32)
            else:
                histo_signal = histo_signal.data
            if histo_normalization is None:
                histo_normalization = numpy.empty((self.bins, 2), dtype=numpy.float32)
            else:
                histo_normalization = histo_normalization.data
            if histo_variance is None:
                histo_variance = numpy.empty((self.bins, 2), dtype=numpy.float32)
            else:
                histo_variance = histo_normalization.data
            if histo_count is None:
                histo_count = numpy.empty(self.bins, dtype=numpy.uint32)
            else:
                histo_count = histo_count.data
            if intensity is None:
                intensity = numpy.empty(self.bins, dtype=numpy.float32)
            else:
                intensity = intensity.data
            if error is None:
                error = numpy.empty(self.bins, dtype=numpy.float32)
            else:
                error = error.data

            ev = pyopencl.enqueue_copy(self.queue, histo_signal, self.cl_mem["histo_sig"])
            events.append(EventDescription("copy D->H histo_sig", ev))
            ev = pyopencl.enqueue_copy(self.queue, histo_variance, self.cl_mem["histo_var"])
            events.append(EventDescription("copy D->H histo_variance", ev))
            ev = pyopencl.enqueue_copy(self.queue, histo_normalization, self.cl_mem["histo_nrm"])
            events.append(EventDescription("copy D->H histo_normalization", ev))
            ev = pyopencl.enqueue_copy(self.queue, histo_count, self.cl_mem["histo_cnt"])
            events.append(EventDescription("copy D->H histo_count", ev))
            ev = pyopencl.enqueue_copy(self.queue, intensity, self.cl_mem["intensity"])
            events.append(EventDescription("copy D->H intensity", ev))
            ev = pyopencl.enqueue_copy(self.queue, error, self.cl_mem["error"])
            events.append(EventDescription("copy D->H error", ev))
            delta = (maxi - mini) / self.bins
            positions = numpy.linspace(mini + 0.5 * delta, maxi - 0.5 * delta, self.bins)
            ev.wait()

        if self.profile:
            self.events += events

        return Integrate1dtpl(positions, intensity, error, histo_signal, histo_variance, histo_normalization, histo_count)

    # Name of the default "process" method
    __call__ = integrate


class OCL_Histogram2d(OCL_Histogram1d):
    """Class in charge of performing histogram calculation in OpenCL using
    atomic_add

    It also performs the preprocessing using the preproc kernel
    """
    BLOCK_SIZE = 32
    buffers = [BufferDescription("output4", 4, numpy.float32, mf.READ_WRITE),
               BufferDescription("radial", 1, numpy.float32, mf.READ_ONLY),
               BufferDescription("azimuthal", 1, numpy.float32, mf.READ_ONLY),
               BufferDescription("image_raw", 1, numpy.float32, mf.READ_ONLY),
               BufferDescription("image", 1, numpy.float32, mf.READ_WRITE),
               BufferDescription("variance", 1, numpy.float32, mf.READ_WRITE),
               BufferDescription("dark", 1, numpy.float32, mf.READ_WRITE),
               BufferDescription("dark_variance", 1, numpy.float32, mf.READ_ONLY),
               BufferDescription("flat", 1, numpy.float32, mf.READ_ONLY),
               BufferDescription("polarization", 1, numpy.float32, mf.READ_ONLY),
               BufferDescription("solidangle", 1, numpy.float32, mf.READ_ONLY),
               BufferDescription("absorption", 1, numpy.float32, mf.READ_ONLY),
               BufferDescription("mask", 1, numpy.int8, mf.READ_ONLY),
               ]
    kernel_files = ["pyfai:openCL/kahan.cl",
                    "pyfai:openCL/preprocess.cl",
                    "pyfai:openCL/ocl_histo.cl"
                    ]
    mapping = {numpy.int8: "s8_to_float",
               numpy.uint8: "u8_to_float",
               numpy.int16: "s16_to_float",
               numpy.uint16: "u16_to_float",
               numpy.uint32: "u32_to_float",
               numpy.int32: "s32_to_float"
               }

    def __init__(self, radial, azimuthal,
                 bins_radial, bins_azimuthal,
                 checksum_radial=None, checksum_azimuthal=None,
                 empty=None, unit=None,
                 ctx=None, devicetype="all", platformid=None, deviceid=None,
                 block_size=None, profile=False):
        """
        :param radial: array with the radial position of every single pixel. Same as image size
        :param azimuthal: array with the azimuthal position of every single pixel. Same as image size
        :param bins_radial: number of bins on which to histogram is calculated in radial direction
        :param bins_azimuthal: number of bins on which to histogram is calculated in azimuthal direction
        :param checksum_radial: pre-calculated checksum of the position array to prevent re-calculating it :)
        :param checksum_azimuthal: pre-calculated checksum of the position array to prevent re-calculating it :)

        :param empty: value to be assigned to bins without contribution from any pixel
        :param ctx: actual working context, left to None for automatic
                    initialization from device type or platformid/deviceid
        :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
        :param platformid: integer with the platform_identifier, as given by clinfo
        :param deviceid: Integer with the device identifier, as given by clinfo
        :param block_size: preferred workgroup size, may vary depending on the outpcome of the compilation
        :param profile: switch on profiling to be able to profile at the kernel level,
                        store profiling elements (makes code slightly slower)
        """
        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
                                  platformid=platformid, deviceid=deviceid,
                                  block_size=block_size, profile=profile)
        self.unit = unit
        self.bins_radial = numpy.uint32(bins_radial)
        self.bins_azimuthal = numpy.uint32(bins_azimuthal)
        self.bins = numpy.uint32(bins_radial * bins_azimuthal)
        self.size = numpy.uint32(radial.size)
        self.empty = numpy.float32(empty) if empty is not None else numpy.float32(0.0)

        self.mini_rad = numpy.float32(numpy.min(radial))
        self.maxi_rad = numpy.float32(numpy.max(radial) * (1.0 + numpy.finfo(numpy.float32).eps))
        self.mini_azim = numpy.float32(numpy.min(azimuthal))
        self.maxi_azim = numpy.float32(numpy.max(azimuthal) * (1.0 + numpy.finfo(numpy.float32).eps))
        self.maxi = self.mini = None

        if not checksum_radial:
            checksum_radial = calc_checksum(radial)
        if not checksum_azimuthal:
            checksum_azimuthal = calc_checksum(azimuthal)
        self.on_device = {"radial": checksum_radial,
                          "azimuthal": checksum_azimuthal,
                          "dark": None,
                          "flat": None,
                          "polarization": None,
                          "solidangle": None,
                          "absorption": None}

        if block_size is None:
            block_size = self.BLOCK_SIZE

        self.BLOCK_SIZE = min(block_size, self.device.max_work_group_size)
        self.workgroup_size = {}
        self.wdim_bins = (self.bins + self.BLOCK_SIZE - 1) & ~(self.BLOCK_SIZE - 1),
        self.wdim_data = (self.size + self.BLOCK_SIZE - 1) & ~(self.BLOCK_SIZE - 1),

        self.buffers = [BufferDescription(i.name, i.size * self.size, i.dtype, i.flags)
                        for i in self.__class__.buffers]

        self.buffers += [BufferDescription("histo_sig", self.bins * 2, numpy.float32, mf.READ_WRITE),
                         BufferDescription("histo_var", self.bins * 2, numpy.float32, mf.READ_WRITE),
                         BufferDescription("histo_nrm", self.bins * 2, numpy.float32, mf.READ_WRITE),
                         BufferDescription("histo_cnt", self.bins, numpy.uint32, mf.READ_WRITE),
                         BufferDescription("error", self.bins, numpy.float32, mf.WRITE_ONLY),
                         BufferDescription("intensity", self.bins, numpy.float32, mf.WRITE_ONLY),
                         BufferDescription("position", 1, numpy.float32, mf.WRITE_ONLY)]
        try:
            self.set_profiling(profile)
            self.allocate_buffers()
            self.compile_kernels()
            self.set_kernel_arguments()
        except (pyopencl.MemoryError, pyopencl.LogicError) as error:
            raise MemoryError(error)
        self.radial = radial
        self.azimuthal = azimuthal
        self.send_buffer(radial, "radial", checksum_radial)
        self.send_buffer(azimuthal, "azimuthal", checksum_azimuthal)

    def __copy__(self):
        """Shallow copy of the object

        :return: copy of the object
        """
        return self.__class__(self.radial, self.azimuthal,
                              self.bins_radial, self.bins_azimuthal,
                              checksum_radial=self.on_device.get("radial"),
                              checksum_azimuthal=self.on_device.get("azimuthal"),
                              empty=self.empty,
                              ctx=self.ctx,
                              block_size=self.block_size,
                              profile=self.profile)

    def __deepcopy__(self, memo=None):
        """deep copy of the object

        :return: deepcopy of the object
        """
        if memo is None:
            memo = {}
        radial = self.radial.copy()
        azimuthal = self.azimuthal.copy()
        memo[id(self.radial)] = radial
        memo[id(self.azimuthal)] = azimuthal
        new_obj = self.__class__(radial, azimuthal,
                                 self.bins_radial, self.bins_azimuthal,
                                 checksum_radial=self.on_device.get("radial"),
                                 checksum_azimuthal=self.on_device.get("azimuthal"),
                                 empty=self.empty,
                                 ctx=self.ctx,
                                 block_size=self.block_size,
                                 profile=self.profile)
        memo[id(self)] = new_obj
        return new_obj

    def set_kernel_arguments(self):
        """Tie arguments of OpenCL kernel-functions to the actual kernels

        """
        OCL_Histogram1d.set_kernel_arguments(self)
        self.cl_kernel_args["histogram_2d_preproc"] = OrderedDict((("radial", self.cl_mem["radial"]),
                                                                   ("azimuthal", self.cl_mem["azimuthal"]),
                                                                   ("preproc4", self.cl_mem["output4"]),
                                                                   ("histo_sig", self.cl_mem["histo_sig"]),
                                                                   ("histo_var", self.cl_mem["histo_var"]),
                                                                   ("histo_nrm", self.cl_mem["histo_nrm"]),
                                                                   ("histo_cnt", self.cl_mem["histo_cnt"]),
                                                                   ("size", self.size),
                                                                   ("bins_radial", self.bins_radial),
                                                                   ("bins_azimuthal", self.bins_azimuthal),
                                                                   ("mini_rad", self.mini_rad),
                                                                   ("maxi_rad", self.maxi_rad),
                                                                   ("mini_azim", self.mini_azim),
                                                                   ("maxi_azim", self.maxi_azim)))

    def integrate(self, data, dark=None,
                  dummy=None, delta_dummy=None,
                  variance=None, dark_variance=None,
                  flat=None, solidangle=None, polarization=None, absorption=None,
                  dark_checksum=None, flat_checksum=None, solidangle_checksum=None,
                  polarization_checksum=None, absorption_checksum=None,
                  preprocess_only=False, safe=True,
                  normalization_factor=1.0,
                  radial_range=None, azimuthal_range=None,
                  histo_signal=None, histo_variance=None,
                  histo_normalization=None, histo_count=None,
                  intensity=None, error=None):
        """
        Performing azimuthal integration, the preprocessing is:

        .. math::

            Signal= (data - dark)
            Variance= (variance + dark_variance)
            Normalization= (normalization_factor * flat * solidangle * polarization * absorption)
            Count= 1 per valid pixel

        Integration is performed using the histograms (based on atomic adds

        :param dark: array of same shape as data for pre-processing
        :param dummy: value for invalid data
        :param delta_dummy: precesion for dummy assessement
        :param variance: array of same shape as data for pre-processing
        :param dark_variance: array of same shape as data for pre-processing
        :param flat: array of same shape as data for pre-processing
        :param solidangle: array of same shape as data for pre-processing
        :param polarization: array of same shape as data for pre-processing
        :param dark_checksum: CRC32 checksum of the given array
        :param flat_checksum: CRC32 checksum of the given array
        :param solidangle_checksum: CRC32 checksum of the given array
        :param polarization_checksum: CRC32 checksum of the given array
        :param safe: if True (default) compares arrays on GPU according to their checksum, unless, use the buffer location is used
        :param preprocess_only: return the dark subtracted; flat field & solidangle & polarization corrected image, else
        :param normalization_factor: divide raw signal by this value
        :param bin_range: provide lower and upper bound for position
        :param histo_signal: destination array or pyopencl array for sum of signals
        :param histo_normalization: destination array or pyopencl array for sum of normalization
        :param histo_count: destination array or pyopencl array for counting pixels
        :param intensity: destination PyOpenCL array for integrated intensity
        :param error: destination PyOpenCL array for standart deviation
        :return: bin_positions, averaged data, histogram of signal, histogram of variance, histogram of normalization, count of pixels
        """
        events = []
        with self.sem:
            self.send_buffer(data, "image")
            memset = self.kernels.memset_histograms(self.queue, self.wdim_bins, self.workgroup_size["memset_histograms"],
                                                    *list(self.cl_kernel_args["memset_histograms"].values()))
            events.append(EventDescription("memset_histograms", memset))
            kw_correction = self.cl_kernel_args["corrections4"]
            kw_histogram = self.cl_kernel_args["histogram_2d_preproc"]

            if variance is None:
                kw_correction["variance"] = self.cl_mem["image"]
            else:
                self.send_buffer(variance, "variance")
                kw_correction["variance"] = self.cl_mem["variance"]

            if dummy is not None:
                do_dummy = numpy.int8(1)
                dummy = numpy.float32(dummy)
                if delta_dummy is None:
                    delta_dummy = numpy.float32(0.0)
                else:
                    delta_dummy = numpy.float32(abs(delta_dummy))
            else:
                do_dummy = numpy.int8(0)
                dummy = numpy.float32(self.empty)
                delta_dummy = numpy.float32(0.0)

            kw_correction["do_dummy"] = do_dummy
            kw_correction["dummy"] = dummy
            kw_correction["delta_dummy"] = delta_dummy
            kw_correction["normalization_factor"] = numpy.float32(normalization_factor)

            if dark is not None:
                do_dark = numpy.int8(1)
                # TODO: what is do_checksum=False and image not on device ...
                if not dark_checksum:
                    dark_checksum = calc_checksum(dark, safe)
                if dark_checksum != self.on_device["dark"]:
                    self.send_buffer(dark, "dark", dark_checksum)
            else:
                do_dark = numpy.int8(0)
            kw_correction["do_dark"] = do_dark

            if dark_variance is None:
                kw_correction["do_dark_variance"] = numpy.int8(0)
            else:
                kw_correction["do_dark_variance"] = numpy.int8(1)

            if flat is not None:
                do_flat = numpy.int8(1)
                if not flat_checksum:
                    flat_checksum = calc_checksum(flat, safe)
                if self.on_device["flat"] != flat_checksum:
                    self.send_buffer(flat, "flat", flat_checksum)
            else:
                do_flat = numpy.int8(0)
            kw_correction["do_flat"] = do_flat

            if solidangle is not None:
                do_solidangle = numpy.int8(1)
                if not solidangle_checksum:
                    solidangle_checksum = calc_checksum(solidangle, safe)
                if solidangle_checksum != self.on_device["solidangle"]:
                    self.send_buffer(solidangle, "solidangle", solidangle_checksum)
            else:
                do_solidangle = numpy.int8(0)
            kw_correction["do_solidangle"] = do_solidangle

            if polarization is not None:
                do_polarization = numpy.int8(1)
                if not polarization_checksum:
                    polarization_checksum = calc_checksum(polarization, safe)
                if polarization_checksum != self.on_device["polarization"]:
                    self.send_buffer(polarization, "polarization", polarization_checksum)
            else:
                do_polarization = numpy.int8(0)
            kw_correction["do_polarization"] = do_polarization

            if absorption is not None:
                do_absorption = numpy.int8(1)
                if not absorption_checksum:
                    absorption_checksum = calc_checksum(absorption, safe)
                if absorption_checksum != self.on_device["absorption"]:
                    self.send_buffer(absorption, "absorption", absorption_checksum)
            else:
                do_absorption = numpy.int8(0)
            kw_correction["do_absorption"] = do_absorption

            ev = self.kernels.corrections4(self.queue, self.wdim_data, self.workgroup_size["corrections4"],
                                           *list(kw_correction.values()))
            events.append(EventDescription("corrections4", ev))

            if preprocess_only:
                image = numpy.empty(data.shape + (4,), dtype=numpy.float32)
                ev = pyopencl.enqueue_copy(self.queue, image, self.cl_mem["output"])
                events.append(EventDescription("copy D->H image", ev))
                if self.profile:
                    self.events += events
                ev.wait()
                return image

            if radial_range:
                mini_rad = numpy.float32(min(radial_range))
                maxi_rad = numpy.float32(max(radial_range) * (1.0 + numpy.finfo(numpy.float32).eps))
            else:
                mini_rad = self.mini_rad
                maxi_rad = self.maxi_rad
            kw_histogram["mini_rad"] = mini_rad
            kw_histogram["maxi_rad"] = maxi_rad

            if azimuthal_range:
                mini_azim = numpy.float32(min(azimuthal_range))
                maxi_azim = numpy.float32(max(azimuthal_range) * (1.0 + numpy.finfo(numpy.float32).eps))
            else:
                mini_azim = self.mini_azim
                maxi_azim = self.maxi_azim
            kw_histogram["mini_azim"] = mini_azim
            kw_histogram["maxi_azim"] = maxi_azim

            histogram = self.kernels.histogram_2d_preproc(self.queue, self.wdim_data, self.workgroup_size["histogram_2d_preproc"],
                                                          *kw_histogram.values())
            events.append(EventDescription("histogram_2d_preproc", histogram))

            postproc = self.kernels.histogram_postproc(self.queue, self.wdim_bins, self.workgroup_size["histogram_postproc"],
                                                       *self.cl_kernel_args["histogram_postproc"].values())
            events.append(EventDescription("histogram_postproc", postproc))

            if histo_signal is None:
                histo_signal = numpy.empty((self.bins_radial, self.bins_azimuthal, 2), dtype=numpy.float32)
            else:
                histo_signal = histo_signal.data
            if histo_normalization is None:
                histo_normalization = numpy.empty((self.bins_radial, self.bins_azimuthal, 2), dtype=numpy.float32)
            else:
                histo_normalization = histo_normalization.data
            if histo_variance is None:
                histo_variance = numpy.empty((self.bins_radial, self.bins_azimuthal, 2), dtype=numpy.float32)
            else:
                histo_variance = histo_normalization.data
            if histo_count is None:
                histo_count = numpy.empty((self.bins_radial, self.bins_azimuthal), dtype=numpy.uint32)
            else:
                histo_count = histo_count.data
            if intensity is None:
                intensity = numpy.empty((self.bins_radial, self.bins_azimuthal), dtype=numpy.float32)
            else:
                intensity = intensity.data
            if error is None:
                error = numpy.empty((self.bins_radial, self.bins_azimuthal), dtype=numpy.float32)
            else:
                error = error.data

            ev = pyopencl.enqueue_copy(self.queue, histo_signal, self.cl_mem["histo_sig"])
            events.append(EventDescription("copy D->H histo_sig", ev))
            ev = pyopencl.enqueue_copy(self.queue, histo_variance, self.cl_mem["histo_var"])
            events.append(EventDescription("copy D->H histo_variance", ev))
            ev = pyopencl.enqueue_copy(self.queue, histo_normalization, self.cl_mem["histo_nrm"])
            events.append(EventDescription("copy D->H histo_normalization", ev))
            ev = pyopencl.enqueue_copy(self.queue, histo_count, self.cl_mem["histo_cnt"])
            events.append(EventDescription("copy D->H histo_count", ev))
            ev = pyopencl.enqueue_copy(self.queue, intensity, self.cl_mem["intensity"])
            events.append(EventDescription("copy D->H intensity", ev))
            ev = pyopencl.enqueue_copy(self.queue, error, self.cl_mem["error"])
            events.append(EventDescription("copy D->H error", ev))

            delta_radial = (maxi_rad - mini_rad) / self.bins_radial
            delta_azimuthal = (maxi_azim - mini_azim) / self.bins_azimuthal

            pos_radial = numpy.linspace(mini_rad + 0.5 * delta_radial, maxi_rad - 0.5 * delta_radial, self.bins_radial)
            pos_azim = numpy.linspace(mini_azim + 0.5 * delta_azimuthal, maxi_azim - 0.5 * delta_azimuthal, self.bins_azimuthal)
            ev.wait()

        if self.profile:
            self.events += events

        return Integrate2dtpl(pos_radial, pos_azim, intensity, error, histo_signal, histo_variance, histo_normalization, histo_count)

    # Name of the default "process" method
    __call__ = integrate
