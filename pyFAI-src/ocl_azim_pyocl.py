# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://forge.epn-campus.eu/projects/azimuthal
#
#    File: "$Id$"
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#                            D. Karkoulis (dimitris.karkoulis@gmail.com)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

"""
C++ less implementation of Dimitris' code based on PyOpenCL 
"""
__author__ = "Jerome Kieffer"
__license__ = "GPLv3"
__date__ = "07/11/2012"
__copyright__ = "2012, ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import os, logging
import threading
import hashlib
import numpy
from opencl import ocl, pyopencl
if pyopencl:
    mf = pyopencl.mem_flags
else:
    raise ImportError("pyopencl is not installed")
logger = logging.getLogger("pyFAI.ocl_azim_pyocl")

class Integrator1d(object):
    """
    Attempt to implements ocl_azim using pyopencl
    """

    def __init__(self, filename=None):
        """

        @param filename: file in which profiling information are saved
        """
        self.BLOCK_SIZE = 128
        self.nBins = -1
        self.nData = -1
        self.platformid = -1
        self.deviceid = -1
        self.useFp64 = False
        self.devicetype = "gpu"
        self.filename = filename
        if filename:
            self.logfile = open(self.filename, a)
        else:
            self.logfile = None
        self.lock = threading.Semaphore()
        #Those are pointer to memory on the GPU (or None if uninitialized
        self._cl_mem = {"tth":None,
                        "tth_delta":None,
                        "image":None,
                        "solidangle":None,
                        "dark":None,
                        "mask":None,
                        "histogram":None,
                        "uhistogram":None,
                        "weights":None,
                        "uweights":None,
                        "span_ranges":None,
                        "tth_min_max":None,
                        "tth_range":None,
                        "dummyval":None,
                        "dummyval_delta":None,
                        }
#        self._cl_kernels = {"integrate":None,
#                            "uimemset2":None,
#                            "imemset":None,
#                            "ui2f2":None,
#                            "get_spans":None,
#                            "group_spans":None,
#                            "solidangle_correction":None,
#                            "dummyval_correction":None}
        self._cl_program = None
        self._ctx = None
        self._queue = None
        self.do_solidangle = None
        self.do_dummy = None
        self.do_mask = None
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
        return os.linesep.join(["Cython wrapper for ocl_xrpd1d.ocl_xrpd1D_fullsplit C++ class. Logging in %s" % self.filename,
                                "device: %s, platform %s device %s 64bits:%s image size: %s histogram size: %s" % (self._devicetype, self._platformid, self._deviceid, self._useFp64, self._nData, self._nBins),
                                ",\t ".join(["%s: %s" % (k, v) for k, v in self.get_status().items()])])

    def log(self, **kwarg):
        """
        log in a file all opencl events
        """
        if self.logfile:
            for key, event in kwarg.items():
#                if  event is an event
                event.wait()
                self.logfile.write(" %s: %.3ms\t" % (key, event,
                           (1e-6 * (event.profile.end - event.profile.start))))
            self.logfile.write(os.linesep)
            self.log.flush()

    def _allocate_buffers(self):
        """
        Allocate OpenCL buffers required for a specific configuration
        
        allocate_CL_buffers() is a private method and is called by configure().
        Given the size of the image and the number of the bins, all the required OpenCL buffers
        are allocated.
        The method performs a basic check to see if the memory required by the configuration is
        smaller than the total global memory of the device. However, there is no built-in way in OpenCL
        to check the real available memory.
        In the case allocate_CL_buffers fails while allocating buffers, it will automatically deallocate
        the buffers that did not fail and leave the flag hasBuffers to 0.
        
        Note that an OpenCL context also requires some memory, as well as Event and other OpenCL functionalities which cannot and
        are not taken into account here.
        The memory required by a context varies depending on the device. Typical for GTX580 is 65Mb but for a 9300m is ~15Mb
        In addition, a GPU will always have at least 3-5Mb of memory in use.
        Unfortunately, OpenCL does NOT have a built-in way to check the actual free memory on a device, only the total memory.
        """
        if self.nData < self.BLOCK_SIZE:
            raise RuntimeError("Fatal error in allocate_CL_buffers. nData (%d) must be >= BLOCK_SIZE (%d)\n", self.nData, self.BLOCK_SIZE)
        size_of_float = numpy.dtype(numpy.float32).itemsize
        size_of_double = numpy.dtype(numpy.float64).itemsize
        size_of_int = numpy.dtype(numpy.int32).itemsize
        size_of_long = numpy.dtype(numpy.int64).itemsize

        ualloc = (self.nData * size_of_float) * 7
        ualloc = (self.nBbins * size_of_float) * 2
        if self.useFp64:
            ualloc += (self.Nbins * numpy.dtype(numpy.int64).itemsize) * 2
        else:
            ualloc += (self.Nbins * numpy.dtype(numpy.int32).itemsize) * 2
        ualloc += 6 * size_of_float
        memory = ocl.platforms[self.platformid][self.deviceid].memory
        if ualloc >= memory:
            raise RuntimeError("Fatal error in allocate_CL_buffers. Not enough device memory for buffers (%lu requested, %lu available)" % (ualloc, memory))
        #now actually allocate:
        try:
            self._cl_mem["tth"] = pyopencl.Buffer(self._ctx, mf.READ_ONLY, size_of_float * self.nData)
            self._cl_mem["tth_delta"] = pyopencl.Buffer(self._ctx, mf.READ_ONLY, size_of_float * self.nData)
            self._cl_mem["mask"] = pyopencl.Buffer(self._ctx, mf.READ_ONLY, size_of_int * self.nData)
            self._cl_mem["image"] = pyopencl.Buffer(self._ctx, mf.READ_ONLY, size_of_float * self.nData)
            self._cl_mem["solidangle"] = pyopencl.Buffer(self._ctx, mf.READ_ONLY, size_of_float * self.nData)
            self._cl_mem["dark"] = pyopencl.Buffer(self._ctx, mf.READ_ONLY, size_of_float * self.nData)
            self._cl_mem["histogram"] = pyopencl.Buffer(self._ctx, mf.READ_WRITE, size_of_float * self.nBins)
            self._cl_mem["weights"] = pyopencl.Buffer(self._ctx, mf.READ_WRITE, size_of_float * self.nBins)
            if self.useFp64:
                self._cl_mem["uhistogram"] = pyopencl.Buffer(self._ctx, mf.READ_WRITE, size_of_long * self.nBins)
                self._cl_mem["uweights"] = pyopencl.Buffer(self._ctx, mf.READ_WRITE, size_of_long * self.nBins)
            else:
                self._cl_mem["uhistogram"] = pyopencl.Buffer(self._ctx, mf.READ_WRITE, size_of_int * self.nBins)
                self._cl_mem["uweights"] = pyopencl.Buffer(self._ctx, mf.READ_WRITE, size_of_int * self.nBins)
            self._cl_mem["span_ranges"] = pyopencl.Buffer(self._ctx, mf.READ_WRITE, size_of_float * self.nData)
            self._cl_mem["tth_min_max"] = pyopencl.Buffer(self._ctx, mf.READ_ONLY, size_of_float * 2)
            self._cl_mem["tth_range"] = pyopencl.Buffer(self._ctx, mf.READ_ONLY, size_of_float * 2)
            self._cl_mem["dummyval"] = pyopencl.Buffer(self._ctx, mf.READ_ONLY, size_of_float * 1)
            self._cl_mem["dummyval_delta"] = pyopencl.Buffer(self._ctx, mf.READ_ONLY, size_of_float * 1)
        except pyopencl.MemoryError as error:
            self._free_buffers()
            raise MemoryError(error)


    def _free_buffers(self):
        """
        free all memory allocated on the device
        """
        for buffer_name in self._cl_mem:
            if self._cl_mem[buffer] is not None:
                try:
                    self._cl_mem[buffer].release()
                    self._cl_mem[buffer] = None
                except LogicError:
                    logger.error("Error while freeing buffer %s" % buffer_name)

    def _free_kernels(self):
        """
        free all kernels
        """
#        for kernel in self._cl_kernels:
#            self._cl_kernels[kernel] = None
        self._cl_program = None

    def _calc_tth_out(self, lower, upper):
        """
        Calculate the bin-center position in 2theta
        """
        self.tth_min = numpy.float32(lower)
        self.tth_max = numpy.float32(upper)
        delta = (upper - lower) / numpy.float32(self._nBins)
        self.tth_out = numpy.arange(lower, upper, delta, dtype=numpy.float32)


    def getConfiguration(self, Nimage, Nbins, useFp64=None):
        """getConfiguration gets the description of the integrations to be performed and keeps an internal copy
        @param Nimage: number of pixel in image
        @param Nbins: number of bins in regrouped histogram
        @param useFp64: use double precision. By default the same as init!
        """
        if Nimage < 1 or Nbins < 1:
            raise RuntimeError("getConfiguration with Nimage=%s and Nbins=%s makes no sense" % (Nimage, Nbins))
        if useFp64 is not None:
            self.useFp64 = bool(useFp64)
        self.nBins = Nbins
        self.nData = Nimage


    def configure(self, kernel=None):
        """
        The method configure() allocates the OpenCL resources required and compiled the OpenCL kernels.
        An active context must exist before a call to configure() and getConfiguration() must have been
        called at least once. Since the compiled OpenCL kernels carry some information on the integration
        parameters, a change to any of the parameters of getConfiguration() requires a subsequent call to
        configure() for them to take effect.
        
        If a configuration exists and configure() is called, the configuration is cleaned up first to avoid
        OpenCL memory leaks        

        @param kernel_path: is the path to the actual kernel
        """
        if self.nBins < 1 or self.nData < 1:
            raise RuntimeError("configure() with Nimage=%s and Nbins=%s makes no sense" % (self.nData, self.nBins))
        if not self._ctx:
            raise RuntimeError("You may not call config() at this point. There is no Active context. (Hint: run init())")

        kernel_name = "ocl_azim_kernel_2.cl"
        if kernel is None:
            if os.path.isfile(kernel_name):
                kernel = kernel_name
            else:
                kernel = os.path.join(os.path.dirname(os.path.abspath(__file__)), kernel_name)
        else:
            kernel = str(kernel)
        source = open(kernel).read()
        #If configure is recalled, force cleanup of OpenCL resources to avoid accidental leaks
        self.clean(True)
        with self.lock:
            self._allocate_buffers()

        try:
            self._program = pyopencl.Program(self._ctx, source).build()
        except pyopencl.MemoryError as error:
            raise MemoryError(error)
        #We need to initialise the Mask to 0



    def loadTth(self, tth, dtth , tth_min=None, tth_max=None):
        """
        Load the 2th arrays along with the min and max value.

        loadTth maybe be recalled at any time of the execution in order to update
        the 2th arrays.

        loadTth is required and must be called at least once after a configure()
        """

        if not self._ctx:
            raise RuntimeError("You may not call loadTth() at this point. There is no Active context. (Hint: run init())")
        if not self._cl_mem["tth"]:
            raise RuntimeError("You may not call loadTth() at this point, OpenCL is not configured (Hint: run configure())")

        ctth = numpy.ascontiguousarray(tth.ravel(), dtype=numpy.float32)
        cdtth = numpy.ascontiguousarray(dtth.ravel(), dtype=numpy.float32)
        with self.lock:
            self._tth_max = (ctth + cdtth).max() * (1.0 + numpy.finfo(numpy.float32).eps)
            self._tth_min = max(0.0, (ctth - cdtth).min())
            if tth_min is None:
                tth_min = self._tth_min

            if tth_max is None:
                tth_max = self._tth_max
            copy_tth = pyopencl.enqueue_copy(sels._queue, self._cl_mem["tth"], ctth)
            copy_dtth = pyopencl.enqueue_copy(sels._queue, self._cl_mem["tth_delta"], cdtth)
            self._calc_tth_out(tth_min, tth_max)
        if self.log:
            copy_tth.wait()
            copy_dtth.wait()

    def setSolidAngle(self, solidAngle):
        """
        Enables SolidAngle correction and uploads the suitable array to the OpenCL device.

        By default the program will assume no solidangle correction unless setSolidAngle() is called.
        From then on, all integrations will be corrected via the SolidAngle array.

        If the SolidAngle array needs to be changes, one may just call setSolidAngle() again
        with that array

        @param solidAngle: numpy array representing the solid angle of the given pixel
        """
        cSolidANgle = numpy.ascontiguousarray(solidAngle.ravel(), dtype=numpy.float32)
        with self.lock:
            self.do_solidangle = True
            if self._cl_mem["solidangle"] is not None:
               self._cl_mem["solidangle"].release()
            self._cl_mem["solidangle"] = pyopencl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cSolidANgle)

    def unsetSolidAngle(self):
        """
        Instructs the program to not perform solidangle correction from now on.

        SolidAngle correction may be turned back on at any point
        """
        with self.lock:
            self.do_solidangle = False
            if self._cl_mem["solidangle"] is not None:
               self._cl_mem["solidangle"].release()
               self._cl_mem["solidangle"] = None

    def setMask(self, mask):
        """
        Enables the use of a Mask during integration. The Mask can be updated by
        recalling setMask at any point.

        The Mask must be a PyFAI Mask. Pixels with 0 are masked out. TODO: check and invert!
        @param mask: numpy.ndarray of integer.
        """
        cMask = numpy.ascontiguousarray(mask.ravel(), dtype=numpy.int32)
        with self.lock:
            self.do_mask = True
            if self._cl_mem["mask"] is not None:
               self._cl_mem["mask"].release()
            self._cl_mem["mask"] = pyopencl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cMask)

    def unsetMask(self):
        """
        Disables the use of a Mask from that point.
        It may be re-enabled at any point via setMask
        """
        with self.lock:
            self.do_mask = False
            if self._cl_mem["mask"] is not None:
               self._cl_mem["mask"].release()
               self._cl_mem["mask"] = None

    def setDummyValue(self, dummy, delta_dummy):
        """
        Enables dummy value functionality and uploads the value to the OpenCL device.

        Image values that are similar to the dummy value are set to 0.

        @param dummy: value in image of missing values (masked pixels?)
        @param delta_dummy: precision for dummy values
        """
        if not self._ctx:
            logger.error("You may not call Integrator1d.setDummyValue(dummy,delta_dummy) at this point. \
                            There is no Active context. (Hint: run init())")
            return
        else:
            with self.lock:
                self.do_dummy = True
                if self._cl_mem["dummyval"]:
                    self._cl_mem["dummyval"].release()
                self._cl_mem["dummyval"] = pyopencl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                                      hostbuf=numpy.array(dummy, dtype=numpy.float32))
                if self._cl_mem["dummyval_delta"]:
                    self._cl_mem["dummyval_delta"].release()
                self._cl_mem["dummyval_delta"] = pyopencl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                                      hostbuf=numpy.array(delta_dummy, dtype=numpy.float32))

    def unsetDummyValue(self):
        """Disable a dummy value.
        May be re-enabled at any time by setDummyValue
        """
        with self.lock:
            self.do_dummy = False
            if self._cl_mem["dummyval"]:
                self._cl_mem["dummyval"].release()
                self._cl_mem["dummyval"] = None
            if self._cl_mem["dummyval_delta"]:
                self._cl_mem["dummyval_delta"].release()
                self._cl_mem["dummyval_delta"] = None


    def setRange(self, lowerBound, upperBound):
        """
        Instructs the program to use a user - defined range for 2th values

        setRange is optional. By default the integration will use the tth_min and tth_max given by loadTth() as integration
        range. When setRange is called it sets a new integration range without affecting the 2th array. All values outside that
        range will then be discarded when interpolating.
        Currently, if the interval of 2th (2th + -d2th) is not all inside the range specified, it is discarded. The bins of the
        histogram are RESCALED to the defined range and not the original tth_max - tth_min range.

        setRange can be called at any point and as many times required after a valid configuration is created.

        @param lowerBound: A float value for the lower bound of the integration range
        @param upperBound: A float value for the upper bound of the integration range
        """
        if self._ctx is None:
            logger.error("You may not call setRange() at this point. There is no Active context. (Hint: run init())")

        tthrmm = numpy.array([lowerBound, upperBound], dtype=numpy.float32)
        with self.lock:
            self.useTthRange = True
            self._calc_tth_out(lowerBound, upperBound)
            if self._cl_mem["tth_min_max"]:
                self._cl_mem["tth_min_max"].release()
            self._cl_mem["tth_min_max"] = pyopencl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tthrmm)

    def unsetRange(self):
        """
        Disable the use of a user-defined 2th range and revert to tth_min,tth_max range
 
        unsetRange instructs the program to revert to its default integration range. If the method is called when
        no user-defined range had been previously specified, no action will be performed
        """

        with self.lock:
            if self.useTthRange:
                self._calc_tth_out(self._tth_min, self._tth_max)
            self.useTthRange = False
            if self._cl_mem["tth_min_max"]:
                self._cl_mem["tth_min_max"].release()
                self._cl_mem["tth_min_max"] = None

    def execute(self, image):
        """
        Perform a 1D azimuthal integration

        execute() may be called only after an OpenCL device is configured and a Tth array has been loaded (at least once)
        It takes the input image and based on the configuration provided earlier it performs the 1D integration.
        Notice that if the provided image is bigger than N then only N points will be taked into account, while
        if the image is smaller than N the result may be catastrophic.
        set/unset and loadTth methods have a direct impact on the execute() method.
        All the rest of the methods will require at least a new configuration via configure().

        Takes an image, integrate and return the histogram and weights


        @param image: image to be processed as a numpy array
        @return: tth_out, histogram, bins

        TODO: to improve performances, the image should be casted to float32 in an optimal way:
        currently using numpy machinery but would be better if done in OpenCL
        """
        assert image.size == self._nData
        if not self._ctx:
            raise RuntimeError("You may not call execute() at this point. There is no Active context. (Hint: run init())")
        if not self.self._cl_mem["histogram"]:
            raise RuntimeError("You may not call execute() at this point, kernels are not configured (Hint: run configure())")
        if not self._cl_mem["tth"]:
            raise RuntimeError("You may not call execute() at this point. There is no 2th array loaded. (Hint: run loadTth())")

        cimage = numpy.ascontiguousarray(image.ravel(), dtype=numpy.float32)


        histogram = numpy.empty(self._nBins, dtype=numpy.float32)
        bins = numpy.empty(self._nBins, dtype=numpy.float32)
        tth_out = self.tth_out.copy()
        pyopencl.enqueue_copy(self._queue, histogram, self._cl_mem["histogram"])
        pyopencl.enqueue_copy(self._queue, bins, self._cl_mem["?"])
        pyopencl.enqueue_barrier(self._queue).wait()
        return tth_out, histogram, bins

    def init(self, devicetype="GPU", useFp64=True, platformid=None, deviceid=None):
        """Initial configuration: Choose a device and initiate a context. Devicetypes can be GPU,gpu,CPU,cpu,DEF,ACC,ALL.
        Suggested are GPU,CPU. For each setting to work there must be such an OpenCL device and properly installed.
        E.g.: If Nvidia driver is installed, GPU will succeed but CPU will fail. The AMD SDK kit is required for CPU via OpenCL.
        @param devicetype: string in ["cpu","gpu", "all", "acc"]
        @param useFp64: boolean specifying if double precision will be used
        @param platformid: integer
        @param devid: integer
        """
        if self._ctx is None:
            self._ctx = self.ocl.create_context(devicetype, useFp64, platformid, deviceid)
            device = self._ctx.device[0]
            self.devicetype = pyopencl.device_type.to_string(device.type)
            self.useFp64 = "fp64" in device.extensions
            self.platformid = pyopencl.get_platforms().index(device.platform)
            self.deviceid = pyopencl.get_platforms()[self.platformid].get_devices().index(device)
            if self.filename:
                self._queue = pyopencl.CommandQueue(self._ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
            else:
                self._queue = pyopencl.CommandQueue(self._ctx)
        else:
            logger.warning("Recycling existing context ... if you want to get start from scratch, use clean()")


    def clean(self, preserve_context=False):
        """
        Free OpenCL related resources allocated by the library.
 
        clean() is used to reinitiate the library back in a vanilla state.
        It may be asked to preserve the context created by init or completely clean up OpenCL.
        Guard/Status flags that are set will be reset.
        
        @param preserve_context Flag that preserves the context (True) or destroys all OpenCL resources (False)
        """

        with self.lock:
            self._queue.finish()
            self._queue = None
            self.nBins = -1
            self.nData = -1
            self.platformid = -1
            self.deviceid = -1
            self.useFp64 = False
            self.devicetype = "gpu"
            self._free_buffers()
            self._free_kernels()
            if  not preserve_context:
                self._queue = None
                self._ctx = None
