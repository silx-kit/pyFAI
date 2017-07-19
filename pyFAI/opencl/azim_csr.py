# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit
#
#
#    Copyright (C) 2014-2017 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#                            Giannis Ashiotis
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

__authors__ = ["Jérôme Kieffer", "Giannis Ashiotis"]
__license__ = "MIT"
__date__ = "19/07/2017"
__copyright__ = "2014-2017, ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import logging
from collections import OrderedDict
import numpy
from .common import ocl, pyopencl, kernel_workgroup_size
from ..utils import calc_checksum

if pyopencl:
    mf = pyopencl.mem_flags
else:
    raise ImportError("pyopencl is not installed")

from .processing import EventDescription, OpenclProcessing, BufferDescription


logger = logging.getLogger("pyFAI.opencl.azim_csr")


class OCL_CSR_Integrator(OpenclProcessing):
    """Class in charge of doing a sparse-matrix multiplication in OpenCL
    using the CSR representation of the matrix.

    It also performs the preprocessing using the preproc kernel
    """
    BLOCK_SIZE = 32
    buffers = [
               BufferDescription("output", 1, numpy.float32, mf.WRITE_ONLY),
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
    kernel_files = ["kahan.cl", "preprocess.cl", "memset.cl", "ocl_azim_CSR.cl"]
    mapping = {numpy.int8: "s8_to_float",
               numpy.uint8: "u8_to_float",
               numpy.int16: "s16_to_float",
               numpy.uint16: "u16_to_float",
               numpy.uint32: "u32_to_float",
               numpy.int32: "s32_to_float"}

    def __init__(self, lut, image_size, checksum=None, empty=None,
                 ctx=None, devicetype="all", platformid=None, deviceid=None,
                 block_size=None, profile=False):
        """
        :param lut: 3-tuple of arrays
            data: coefficient of the matrix in a 1D vector of float32 - size of nnz
            indices: Column index position for the data (same size as data)
            indptr: row pointer indicates the start of a given row. len nbin+1
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

        self._data, self._indices, self._indptr = lut
        self.bins = self._indptr.size - 1
        self.nbytes = self._data.nbytes + self._indices.nbytes + self._indptr.nbytes
        if self._data.shape[0] != self._indices.shape[0]:
            raise RuntimeError("data.shape[0] != indices.shape[0]")
        self.data_size = self._data.shape[0]
        self.size = image_size
        self.empty = empty or 0.0

        if not checksum:
            checksum = calc_checksum(self._data)
        self.on_device = {"data": checksum,
                          "dark": None,
                          "flat": None,
                          "polarization": None,
                          "solidangle": None,
                          "absorption": None}

        if block_size is None:
            block_size = self.BLOCK_SIZE

        self.BLOCK_SIZE = min(block_size, self.device.max_work_group_size)
        self.workgroup_size = {}
        self.wdim_bins = (self.bins * self.BLOCK_SIZE),
        self.wdim_data = (self.size + self.BLOCK_SIZE - 1) & ~(self.BLOCK_SIZE - 1),

        self.buffers = [BufferDescription(i.name, i.size * self.size, i.dtype, i.flags)
                        for i in self.__class__.buffers]

        self.buffers += [BufferDescription("data", self.data_size, numpy.float32, mf.READ_ONLY),
                         BufferDescription("indices", self.data_size, numpy.int32, mf.READ_ONLY),
                         BufferDescription("indptr", (self.bins + 1), numpy.int32, mf.READ_ONLY),
                         BufferDescription("outData", self.bins, numpy.float32, mf.WRITE_ONLY),
                         BufferDescription("outCount", self.bins, numpy.float32, mf.WRITE_ONLY),
                         BufferDescription("outMerge", self.bins, numpy.float32, mf.WRITE_ONLY),
                         ]
        try:
            self.set_profiling(profile)
            self.allocate_buffers()
            self.compile_kernels()
            self.set_kernel_arguments()
        except (pyopencl.MemoryError, pyopencl.LogicError) as error:
            raise MemoryError(error)
        self.send_buffer(self._data, "data")
        self.send_buffer(self._indices, "indices")
        self.send_buffer(self._indptr, "indptr")

    def __copy__(self):
        """Shallow copy of the object

        :return: copy of the object
        """
        return self.__class__((self._data, self._indices, self._indptr),
                              self.size,
                              checksum=self.on_device.get("data"),
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
        new_csr = self._data.copy(), self._indices.copy(), self._indptr.copy()
        memo[id(self._data)] = new_csr[0]
        memo[id(self._indices)] = new_csr[1]
        memo[id(self._indptr)] = new_csr[2]
        new_obj = self.__class__(new_csr, self.size,
                                 checksum=self.on_device.get("data"),
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

        compile_options = "-D NBINS=%i  -D NIMAGE=%i -D WORKGROUP_SIZE=%i" % \
                          (self.bins, self.size, self.BLOCK_SIZE)
        OpenclProcessing.compile_kernels(self, kernels, compile_options)
        for kernel in self.program.all_kernels():
            kernel_name = kernel.function_name
            wg = kernel_workgroup_size(self.program, kernel)
            self.workgroup_size[kernel_name] = (min(wg, self.BLOCK_SIZE),)  # this is a tuple

    def set_kernel_arguments(self):
        """Tie arguments of OpenCL kernel-functions to the actual kernels

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

        self.cl_kernel_args["csr_integrate"] = OrderedDict((("output", self.cl_mem["output"]),
                                                            ("data", self.cl_mem["data"]),
                                                            ("indices", self.cl_mem["indices"]),
                                                            ("indptr", self.cl_mem["indptr"]),
                                                            ("do_dummy", numpy.int8(0)),
                                                            ("dummy", numpy.float32(0)),
                                                            ("outData", self.cl_mem["outData"]),
                                                            ("outCount", self.cl_mem["outCount"]),
                                                            ("outMerge", self.cl_mem["outMerge"])))
        self.cl_kernel_args["csr_integrate_single"] = self.cl_kernel_args["csr_integrate"]

        self.cl_kernel_args["memset_out"] = OrderedDict(((i, self.cl_mem[i]) for i in ("outData", "outCount", "outMerge")))
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
            kernel = getattr(self.program, self.mapping[data.dtype.type])
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
                  preprocess_only=False, safe=True, normalization_factor=1.0):
        """
        Before performing azimuthal integration, the preprocessing is:

        .. math::

            data = (data - dark) / (flat * solidangle * polarization)

        Integration is performed using the CSR representation of the look-up table

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
        :return: averaged data, weighted histogram, unweighted histogram
        """
        events = []
        with self.sem:
            self.send_buffer(data, "image")
            wg = self.workgroup_size["memset_out"]
            wdim_bins = (self.bins + wg[0] - 1) & ~(wg[0] - 1),
            memset = self.program.memset_out(self.queue, wdim_bins, wg, *list(self.cl_kernel_args["memset_out"].values()))
            events.append(EventDescription("memset_out", memset))
            kw1 = self.cl_kernel_args["corrections"]
            kw2 = self.cl_kernel_args["csr_integrate"]

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

            if dark is not None:
                do_dark = numpy.int8(1)
                # TODO: what is do_checksum=False and image not on device ...
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
                    self.send_buffer(solidangle, "solidangle", solidangle_checksum)
            else:
                do_solidangle = numpy.int8(0)
            kw1["do_solidangle"] = do_solidangle

            if polarization is not None:
                do_polarization = numpy.int8(1)
                if not polarization_checksum:
                    polarization_checksum = calc_checksum(polarization, safe)
                if polarization_checksum != self.on_device["polarization"]:
                    self.send_buffer(polarization, "polarization", polarization_checksum)
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

            wg = self.workgroup_size["corrections"]
            ev = self.program.corrections(self.queue, self.wdim_data, wg, *list(kw1.values()))
            events.append(EventDescription("corrections", ev))

            if preprocess_only:
                image = numpy.empty(data.shape, dtype=numpy.float32)
                ev = pyopencl.enqueue_copy(self.queue, image, self.cl_mem["output"])
                events.append(EventDescription("copy D->H image", ev))
                if self.profile:
                    self.events += events
                ev.wait()
                return image
            wg = self.workgroup_size["csr_integrate"][0]

            wdim_bins = (self.bins * wg),
            if wg == 1:
                integrate = self.program.csr_integrate_single(self.queue, wdim_bins, (wg,), *kw2.values())
                events.append(EventDescription("integrate_single", integrate))
            else:
                integrate = self.program.csr_integrate(self.queue, wdim_bins, (wg,), *kw2.values())
                events.append(EventDescription("integrate", integrate))
            outMerge = numpy.empty(self.bins, dtype=numpy.float32)
            outData = numpy.empty(self.bins, dtype=numpy.float32)
            outCount = numpy.empty(self.bins, dtype=numpy.float32)
            ev = pyopencl.enqueue_copy(self.queue, outMerge, self.cl_mem["outMerge"])
            events.append(EventDescription("copy D->H outMerge", ev))
            ev = pyopencl.enqueue_copy(self.queue, outData, self.cl_mem["outData"])
            events.append(EventDescription("copy D->H outData", ev))
            ev = pyopencl.enqueue_copy(self.queue, outCount, self.cl_mem["outCount"])
            events.append(EventDescription("copy D->H outCount", ev))
            ev.wait()
        if self.profile:
            self.events += events
        return outMerge, outData, outCount
