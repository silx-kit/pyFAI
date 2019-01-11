# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
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


__author__ = "Jerome Kieffer"
__license__ = "MIT"
__date__ = "11/01/2019"
__copyright__ = "2012-2017, ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import logging
import numpy
from collections import OrderedDict
from . import pyopencl, get_x87_volatile_option
from ..utils import calc_checksum
if pyopencl:
    mf = pyopencl.mem_flags
else:
    raise ImportError("pyopencl is not installed")

from . import processing
EventDescription = processing.EventDescription
OpenclProcessing = processing.OpenclProcessing
BufferDescription = processing.BufferDescription

logger = logging.getLogger(__name__)


class OCL_LUT_Integrator(OpenclProcessing):
    """Class in charge of doing a sparse-matrix multiplication in OpenCL
    using the LUT representation of the matrix.
    It also performs the preprocessing using the preproc kernel
    """
    BLOCK_SIZE = 16
    buffers = [BufferDescription("output", 1, numpy.float32, mf.WRITE_ONLY),
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
                    "pyfai:openCL/memset.cl",
                    "pyfai:openCL/ocl_azim_LUT.cl"
                    ]
    mapping = {numpy.int8: "s8_to_float",
               numpy.uint8: "u8_to_float",
               numpy.int16: "s16_to_float",
               numpy.uint16: "u16_to_float",
               numpy.uint32: "u32_to_float",
               numpy.int32: "s32_to_float"
               }

    def __init__(self, lut, image_size, checksum=None, empty=None,
                 ctx=None, devicetype="all", platformid=None, deviceid=None,
                 block_size=None, profile=False):
        """Constructor of the OCL_LUT_Integrator class

        :param lut: array of int32 - float32 with shape (nbins, lut_size) with indexes and coefficients
        :param image_size: Expected image size: image.size
        :param checksum: pre-calculated checksum of the LUT to prevent re-calculating it :)
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
        self._lut = lut
        self.nbytes = lut.nbytes
        self.bins, self.lut_size = lut.shape
        self.size = image_size
        self.profile = None
        self.empty = empty or 0.0

        if not checksum:
            checksum = calc_checksum(self._lut)
        self.on_device = {"lut": checksum,
                          "dark": None,
                          "flat": None,
                          "polarization": None,
                          "solidangle": None,
                          "absorption": None}

        self.BLOCK_SIZE = min(self.BLOCK_SIZE, self.device.max_work_group_size)
        self.workgroup_size = self.BLOCK_SIZE,  # Note this is a tuple
        self.wdim_bins = (self.bins + self.BLOCK_SIZE - 1) & ~(self.BLOCK_SIZE - 1),
        self.wdim_data = (self.size + self.BLOCK_SIZE - 1) & ~(self.BLOCK_SIZE - 1),

        self.buffers = [BufferDescription(i.name, i.size * self.size, i.dtype, i.flags)
                        for i in self.__class__.buffers]
        self.buffers += [BufferDescription("lut", self.bins * self.lut_size,
                                           [("bins", numpy.float32),
                                            ("lut_size", numpy.int32)], mf.READ_ONLY),
                         BufferDescription("sum_data", self.bins, numpy.float32, mf.WRITE_ONLY),
                         BufferDescription("sum_count", self.bins, numpy.float32, mf.WRITE_ONLY),
                         BufferDescription("merged", self.bins, numpy.float32, mf.WRITE_ONLY),
                         ]
        self.allocate_buffers()
        self.compile_kernels()
        self.set_kernel_arguments()
        if self.device.type == "CPU":
            ev = pyopencl.enqueue_copy(self.queue, self.cl_mem["lut"], lut)
        else:
            ev = pyopencl.enqueue_copy(self.queue, self.cl_mem["lut"], lut.T.copy())
        if self.profile:
            self.events.append(EventDescription("copy LUT", ev))

    def __copy__(self):
        """Shallow copy of the object

        :return: copy of the object
        """
        return self.__class__(self._lut, self.size,
                              checksum=self.on_device.get("lut"),
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
        new_lut = self._lut.copy()
        memo[id(self._lut)] = new_lut
        new_obj = self.__class__(new_lut, self.size,
                                 checksum=self.on_device.get("lut"),
                                 empty=self.empty,
                                 ctx=self.ctx,
                                 block_size=self.block_size,
                                 profile=self.profile)
        memo[id(self)] = new_obj
        return new_obj

    def compile_kernels(self, kernel_file=None):
        """
        Call the OpenCL compiler
        :param kernel_file: path to the kernel (by default use the one in the resources directory)
        """
        # concatenate all needed source files into a single openCL module
        kernel_file = kernel_file or self.kernel_files[-1]
        kernels = self.kernel_files[:-1] + [kernel_file]

        compile_options = "-D NBINS=%i  -D NIMAGE=%i -D NLUT=%i -D ON_CPU=%i" % \
                          (self.bins, self.size, self.lut_size, int(self.device.type == "CPU"))

        try:
            default_compiler_options = self.get_compiler_options(x87_volatile=True)
        except AttributeError:  # Silx version too old
            logger.warning("Please upgrade to silx v0.10+")
            default_compiler_options = get_x87_volatile_option(self.ctx)

        if default_compiler_options:
            compile_options += " " + default_compiler_options
        OpenclProcessing.compile_kernels(self, kernels, compile_options)

    def set_kernel_arguments(self):
        """Tie arguments of OpenCL kernel-functions to the actual kernels

        set_kernel_arguments() is a private method, called by configure().
        It uses the dictionary _cl_kernel_args.
        Note that by default, since TthRange is disabled, the integration kernels have tth_min_max tied to the tthRange argument slot.
        When setRange is called it replaces that argument with tthRange low and upper bounds. When unsetRange is called, the argument slot
        is reset to tth_min_max.
        """
        self.cl_kernel_args["corrections"] = OrderedDict((("image", self.cl_mem["image"]),
                                                          ("do_dark", numpy.int8(0)),
                                                          ("dark", self.cl_mem["dark"]),
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
                                                          ("output", self.cl_mem["output"])))

        self.cl_kernel_args["lut_integrate"] = OrderedDict((("output", self.cl_mem["output"]),
                                                            ("lut", self.cl_mem["lut"]),
                                                            ("do_dummy", numpy.int8(0)),
                                                            ("dummy", numpy.float32(0)),
                                                            ("coef_power", numpy.int32(1)),
                                                            ("sum_data", self.cl_mem["sum_data"]),
                                                            ("sum_count", self.cl_mem["sum_count"]),
                                                            ("merged", self.cl_mem["merged"])))
        self.cl_kernel_args["memset_out"] = OrderedDict(((i, self.cl_mem[i]) for i in ("sum_data", "sum_count", "merged")))
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
        if (data.dtype == dest_type) or (data.dtype.itemsize > dest_type.itemsize):
            copy_image = pyopencl.enqueue_copy(self.queue, self.cl_mem[dest], numpy.ascontiguousarray(data, dest_type))
            events.append(EventDescription("copy %s" % dest, copy_image))
        else:
            copy_image = pyopencl.enqueue_copy(self.queue, self.cl_mem["image_raw"], numpy.ascontiguousarray(data))
            kernel = self.kernels.get_kernel(self.mapping[data.dtype.type])
            cast_to_float = kernel(self.queue, (self.size,), None, self.cl_mem["image_raw"], self.cl_mem[dest])
            events += [EventDescription("copy raw %s" % dest, copy_image), EventDescription("cast to float", cast_to_float)]
        if self.profile:
            self.events += events
        if checksum is not None:
            self.on_device[dest] = checksum

    def integrate(self, data, dummy=None, delta_dummy=None,
                  dark=None, flat=None, solidangle=None, polarization=None, absorption=None,
                  dark_checksum=None, flat_checksum=None, solidangle_checksum=None,
                  polarization_checksum=None, absorption_checksum=None,
                  preprocess_only=False, safe=True,
                  normalization_factor=1.0, coef_power=1,
                  out_merged=None, out_sum_data=None, out_sum_count=None):
        """
        Before performing azimuthal integration, the preprocessing is:

        .. math::

            data = (data - dark) / (flat * solidangle * polarization * absorption)

        Integration is performed using the LUT representation of the look-up table

        :param dark: array of same shape as data for pre-processing
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
        :param coef_power: set to 2 for variance propagation, leave to 1 for mean calculation
        :param out_merged: destination array or pyopencl array for averaged data
        :param out_sum_data: destination array or pyopencl array for sum of all data
        :param out_sum_count: destination array or pyopencl array for sum of the number of pixels

        :return: averaged data, weighted histogram, unweighted histogram
        """
        events = []
        with self.sem:
            self.send_buffer(data, "image")
            memset = self.kernels.memset_out(self.queue, self.wdim_bins, self.workgroup_size, *list(self.cl_kernel_args["memset_out"].values()))
            events.append(EventDescription("memset", memset))
            kw1 = self.cl_kernel_args["corrections"]
            kw2 = self.cl_kernel_args["lut_integrate"]
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

            kw1["do_dummy"] = do_dummy
            kw1["dummy"] = dummy
            kw1["delta_dummy"] = delta_dummy
            kw1["normalization_factor"] = numpy.float32(normalization_factor)
            kw2["do_dummy"] = do_dummy
            kw2["dummy"] = dummy
            kw2["coef_power"] = numpy.int32(coef_power)

            if dark is not None:
                do_dark = numpy.int8(1)
                if not dark_checksum:
                    dark_checksum = calc_checksum(dark, safe)
                if dark_checksum != self.on_device["dark"]:
                    self.send_buffer(dark, "dark", dark_checksum)
            else:
                do_dark = numpy.int8(0)
            kw1["do_dark"] = do_dark
            if flat is not None:
                do_flat = numpy.int8(1)
                if not flat_checksum:
                    flat_checksum = calc_checksum(flat, safe)
                if self.on_device["flat"] != flat_checksum:
                    self.send_buffer(flat, "flat", flat_checksum)
            else:
                do_flat = numpy.int8(0)
            kw1["do_flat"] = do_flat

            if solidangle is not None:
                do_solidangle = numpy.int8(1)
                if not solidangle_checksum:
                    solidangle_checksum = calc_checksum(solidangle, safe)
                if solidangle_checksum != self.on_device["solidangle"]:
                    self.send_buffer(solidangle, "solidangle", flat_checksum)
            else:
                do_solidangle = numpy.int8(0)
            kw1['do_solidangle'] = do_solidangle

            if polarization is not None:
                do_polarization = numpy.int8(1)
                if not polarization_checksum:
                    polarization_checksum = calc_checksum(polarization, safe)
                if polarization_checksum != self.on_device["polarization"]:
                    self.send_buffer(polarization, "polarization", polarization_checksum)
                    self.on_device["polarization"] = polarization_checksum
            else:
                do_polarization = numpy.int8(0)
            kw1["do_polarization"] = do_polarization

            if absorption is not None:
                do_absorption = numpy.int8(1)
                if not absorption_checksum:
                    absorption_checksum = calc_checksum(absorption, safe)
                if absorption_checksum != self.on_device["absorption"]:
                    self.send_buffer(absorption, "absorption", absorption_checksum)
            else:
                do_absorption = numpy.int8(0)
            kw1["do_absorption"] = do_absorption

            ev = self.kernels.corrections(self.queue, self.wdim_data, self.workgroup_size, *list(kw1.values()))
            events.append(EventDescription("corrections", ev))

            if preprocess_only:
                image = numpy.empty(data.shape, dtype=numpy.float32)
                ev = pyopencl.enqueue_copy(self.queue, image, self.cl_mem["output"])
                events.append(EventDescription("copy D->H image", ev))
                if self.profile:
                    self.events += events
                ev.wait()
                return image
            integrate = self.kernels.lut_integrate(self.queue, self.wdim_bins, self.workgroup_size, *list(kw2.values()))
            events.append(EventDescription("integrate", integrate))
            if out_merged is None:
                merged = numpy.empty(self.bins, dtype=numpy.float32)
            else:
                merged = out_merged.data
            if out_sum_count is None:
                sum_count = numpy.empty(self.bins, dtype=numpy.float32)
            else:
                sum_count = out_sum_count.data
            if out_sum_data is None:
                sum_data = numpy.empty(self.bins, dtype=numpy.float32)
            else:
                sum_data = out_sum_data.data
            ev = pyopencl.enqueue_copy(self.queue, merged, self.cl_mem["merged"])
            events.append(EventDescription("copy D->H merged", ev))
            ev = pyopencl.enqueue_copy(self.queue, sum_data, self.cl_mem["sum_data"])
            events.append(EventDescription("copy D->H sum_data", ev))
            ev = pyopencl.enqueue_copy(self.queue, sum_count, self.cl_mem["sum_count"])
            events.append(EventDescription("copy D->H sum_count", ev))
            ev.wait()
        if self.profile:
            self.events += events
        return merged, sum_data, sum_count
