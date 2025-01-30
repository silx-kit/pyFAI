# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2014-2023 European Synchrotron Radiation Facility, Grenoble, France
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
__date__ = "06/12/2024"
__copyright__ = "ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import math
import logging
from collections import OrderedDict
import numpy
from . import pyopencl, dtype_converter
from ..utils import calc_checksum
if pyopencl:
    mf = pyopencl.mem_flags
else:
    raise ImportError("pyopencl is not installed")
from ..containers import Integrate1dtpl, Integrate2dtpl, ErrorModel
from ..utils import EPS32
from . import processing, OpenclProcessing
EventDescription = processing.EventDescription
BufferDescription = processing.BufferDescription

logger = logging.getLogger(__name__)


class OCL_CSR_Integrator(OpenclProcessing):
    """Class in charge of doing a sparse-matrix multiplication in OpenCL
    using the CSR representation of the matrix.

    It also performs the preprocessing using the preproc kernel
    """
    BLOCK_SIZE = 32
    # Intel CPU driver calims preferred workgroup is 128 !
    buffers = [BufferDescription("output", 1, numpy.float32, mf.READ_WRITE),
               BufferDescription("output4", 4, numpy.float32, mf.READ_WRITE),
               BufferDescription("tmp", 1, numpy.float32, mf.READ_WRITE),
               BufferDescription("image_raw", 1, numpy.int64, mf.READ_WRITE),
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
    kernel_files = ["silx:opencl/doubleword.cl",
                    "pyfai:openCL/preprocess.cl",
                    "pyfai:openCL/memset.cl",
                    "pyfai:openCL/ocl_azim_CSR.cl",
                    "pyfai:openCL/collective/reduction.cl",
                    "pyfai:openCL/collective/scan.cl",
                    "pyfai:openCL/collective/comb_sort.cl",
                    "pyfai:openCL/medfilt.cl"
                    ]
    mapping = {numpy.int8: "s8_to_float",
               numpy.uint8: "u8_to_float",
               numpy.int16: "s16_to_float",
               numpy.uint16: "u16_to_float",
               numpy.uint32: "u32_to_float",
               numpy.uintc: "u32_to_float",
               numpy.int32: "s32_to_float",
               numpy.intc: "s32_to_float"
               }

    def __init__(self, lut, image_size, checksum=None,
                 empty=None, unit=None, bin_centers=None, azim_centers=None, mask_checksum=None,
                 ctx=None, devicetype="all", platformid=None, deviceid=None,
                 block_size=None, profile=False, extra_buffers=None):
        """
        :param lut: 3-tuple of arrays
            data: coefficient of the matrix in a 1D vector of float32 - size of nnz
            indices: Column index position for the data (same size as data)
            indptr: row pointer indicates the start of a given row. len nbin+1
        :param image_size: Expected image size: image.size
        :param checksum: pre-calculated checksum of the LUT to prevent re-calculating it :)
        :param empty: value to be assigned to bins without contribution from any pixel
        :param unit: Storage for the unit related to the LUT
        :param bin_centers: the radial position of the bin_center, place_holder
        :param azim_centers: the radial position of the bin_center, place_holder
        :param mask_checksum: placeholder for the checksum of the mask
        :param ctx: actual working context, left to None for automatic
                    initialization from device type or platformid/deviceid
        :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
        :param platformid: integer with the platform_identifier, as given by clinfo
        :param deviceid: Integer with the device identifier, as given by clinfo
        :param block_size: preferred workgroup size, may vary depending on the outcome of the compilation
        :param profile: switch on profiling to be able to profile at the kernel level,
                        store profiling elements (makes code slightly slower)
        :param extra_buffers: List of additional buffer description  needed by derived classes
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
        self.empty = empty or 0
        self.unit = unit
        self.space = tuple(str(u).split("_")[0] for u in unit) if isinstance(unit, (list, tuple)) else  str(unit).split("_")[0]
        self.bin_centers = bin_centers
        self.azim_centers = azim_centers
        # a few place-folders
        self.mask_checksum = mask_checksum
        self.pos0_range = self.pos1_range = None

        if not checksum:
            checksum = calc_checksum(self._data)
        self.on_device = {"data": checksum,
                          "dark": None,
                          "flat": None,
                          "polarization": None,
                          "solidangle": None,
                          "absorption": None,
                          "dark_variance": None}

        block_size = self.guess_workgroup_size(block_size)
        self.BLOCK_SIZE = min(block_size, self.device.max_work_group_size)
        self.workgroup_size = {}

        self.buffers = [BufferDescription(i.name, i.size * self.size, i.dtype, i.flags)
                        for i in self.__class__.buffers]

        if extra_buffers is not None:
            self.buffers += extra_buffers

        self.buffers += [BufferDescription("data", self.data_size, numpy.float32, mf.READ_ONLY),
                         BufferDescription("indices", self.data_size, numpy.int32, mf.READ_ONLY),
                         BufferDescription("indptr", (self.bins + 1), numpy.int32, mf.READ_ONLY),
                         BufferDescription("sum_data", self.bins, numpy.float32, mf.WRITE_ONLY),
                         BufferDescription("sum_count", self.bins, numpy.float32, mf.WRITE_ONLY),
                         BufferDescription("averint", self.bins, numpy.float32, mf.READ_WRITE),
                         BufferDescription("std", self.bins, numpy.float32, mf.READ_WRITE),
                         BufferDescription("sem", self.bins, numpy.float32, mf.READ_WRITE),
                         BufferDescription("merged", self.bins, numpy.float32, mf.WRITE_ONLY),
                         BufferDescription("merged8", (self.bins, 8), numpy.float32, mf.WRITE_ONLY),
                         BufferDescription("work4", (self.data_size, 4), numpy.float32, mf.READ_WRITE),
                         ]
        try:
            self.set_profiling(profile)
            self.allocate_buffers()
            self.compile_kernels()
            self.set_kernel_arguments()
        except (pyopencl.MemoryError, pyopencl.LogicError) as error:
            raise MemoryError(error)
        self.buffer_dtype = {i.name:numpy.dtype(i.dtype) for i in self.buffers}
        if numpy.allclose(self._data, numpy.ones(self._data.shape)):
            self.cl_mem["data"] = None
            self.cl_kernel_args["csr_medfilt"]["data"] = None
            self.cl_kernel_args["csr_sigma_clip4"]["data"] = None
            self.cl_kernel_args["csr_integrate"]["data"] = None
            self.cl_kernel_args["csr_integrate4"]["data"] = None
        else:
            self.send_buffer(self._data, "data")
        self.send_buffer(self._indices, "indices")
        self.send_buffer(self._indptr, "indptr")

        if "amd" in  self.ctx.devices[0].platform.name.lower():
            self.workgroup_size["csr_integrate4_single"] = (1, 1)  # Very bad performances on AMD GPU for diverging threads!

    @property
    def checksum(self):
        return self.on_device.get("data")

    @property
    def check_mask(self):
        return self.mask_checksum is not None

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

    def guess_workgroup_size(self, block_size=None):
        """Determines the optimal workgroup size.

        For azimuthal integration, especially the 2D variant, the
        smallest possible is the size of a warp/wavefront.

        The method can be overwritten by derived classes to select larger workgoup

        :param block_size: Input workgroup size (block is the cuda name)
        :return: the optimal workgoup size as integer
        """
        device = self.ctx.devices[0]
        platform = device.platform.name.lower()
        try:
            devtype = pyopencl.device_type.to_string(device.type).upper()
        except ValueError:
            # pocl does not describe itself as a CPU !
            devtype = "CPU"

        if block_size is None:
            if "nvidia" in  platform:
                try:
                    block_size = device.warp_size_nv
                except:
                    block_size = 32
            elif "amd" in  platform:
                try:
                    block_size = device.wavefront_width_amd
                except:
                    block_size = 64
            elif "intel" in  platform:
                block_size = 128
            elif "portable" in platform and "CPU" in devtype:
                block_size = 8
            else:
                block_size = min(device.max_work_group_size, self.BLOCK_SIZE)
            self.force_workgroup_size = False
        else:
            self.force_workgroup_size = True
            block_size = int(block_size)
        return block_size

    def compile_kernels(self, kernel_file=None):
        """
        Call the OpenCL compiler
        :param kernel_file: path to the kernel (by default use the one in the resources directory)
        """
        # concatenate all needed source files into a single openCL module
        kernel_file = kernel_file or self.kernel_files[-1]
        kernels = self.kernel_files[:-1] + [kernel_file]

        try:
            compile_options = self.get_compiler_options(x87_volatile=True, apple_gpu=True)
        except (AttributeError, TypeError):  # Silx version too old
            logger.warning("Please upgrade to silx v2.2+")
            from . import get_compiler_options
            compile_options = get_compiler_options(self.ctx, x87_volatile=True, apple_gpu=True)

        compile_options += f" -D NBINS={self.bins} -D NIMAGE={self.size}"
        OpenclProcessing.compile_kernels(self, kernels, compile_options.strip())
        for kernel_name in self.kernels.__dict__:
            if kernel_name.startswith("_"):
                continue
            if self.force_workgroup_size:
                self.workgroup_size[kernel_name] = (self.BLOCK_SIZE, self.BLOCK_SIZE)
            else:
                wg_max = self.kernels.max_workgroup_size(kernel_name)
                wg_min = self.kernels.min_workgroup_size(kernel_name)
                if kernel_name=="csr_medfilt":
                    # limit the wg size due to
                    device = self.ctx.devices[0]
                    maxthreads = device.local_mem_size/12/4
                    self.workgroup_size[kernel_name] = (wg_min,
                                                        min(wg_max, 2**(int(math.log2(maxthreads)))))
                else:
                    self.workgroup_size[kernel_name] = (wg_min, wg_max)



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
                                                          ("apply_normalization", numpy.int8(0)),
                                                          ("output", self.cl_mem["output"])))
        self.cl_kernel_args["csr_integrate"] = OrderedDict((("output", self.cl_mem["output"]),
                                                            ("data", self.cl_mem["data"]),
                                                            ("indices", self.cl_mem["indices"]),
                                                            ("indptr", self.cl_mem["indptr"]),
                                                            ("nbins", numpy.int32(self.bins)),
                                                            ("do_dummy", numpy.int8(0)),
                                                            ("dummy", numpy.float32(0)),
                                                            ("coef_power", numpy.int32(1)),
                                                            ("sum_data", self.cl_mem["sum_data"]),
                                                            ("sum_count", self.cl_mem["sum_count"]),
                                                            ("merged", self.cl_mem["merged"]),
                                                            ("shared", pyopencl.LocalMemory(16))))
        self.cl_kernel_args["corrections4a"] = OrderedDict((("image", self.cl_mem["image_raw"]),
                                                            ("dtype", numpy.int8(0)),
                                                           ("error_model", numpy.int8(0)),
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
                                                           ("apply_normalization", numpy.int8(0)),
                                                           ("output4", self.cl_mem["output4"])))
        self.cl_kernel_args["csr_integrate4"] = OrderedDict((("output4", self.cl_mem["output4"]),
                                                            ("data", self.cl_mem["data"]),
                                                            ("indices", self.cl_mem["indices"]),
                                                            ("indptr", self.cl_mem["indptr"]),
                                                            ("nbins", numpy.int32(self.bins)),
                                                            ("empty", numpy.float32(self.empty)),
                                                            ("error_model", numpy.int8(1)),
                                                            ("merged8", self.cl_mem["merged8"]),
                                                            ("averint", self.cl_mem["averint"]),
                                                            ("std", self.cl_mem["std"]),
                                                            ("sem", self.cl_mem["sem"]),
                                                            ("shared", pyopencl.LocalMemory(32))
                                                             ))
        self.cl_kernel_args["csr_sigma_clip4"] = OrderedDict((("output4", self.cl_mem["output4"]),
                                                              ("data", self.cl_mem["data"]),
                                                              ("indices", self.cl_mem["indices"]),
                                                              ("indptr", self.cl_mem["indptr"]),
                                                              ("cutoff", numpy.float32(5)),
                                                              ("cycle", numpy.int32(5)),
                                                              ("error_model", numpy.int8(1)),
                                                              ("empty", numpy.float32(self.empty)),
                                                              ("merged8", self.cl_mem["merged8"]),
                                                              ("averint", self.cl_mem["averint"]),
                                                              ("std", self.cl_mem["std"]),
                                                              ("sem", self.cl_mem["sem"]),
                                                              ("shared", pyopencl.LocalMemory(32))
                                                             ))
        self.cl_kernel_args["csr_integrate_single"] = self.cl_kernel_args["csr_integrate"]
        self.cl_kernel_args["csr_integrate4_single"] = self.cl_kernel_args["csr_integrate4"]
        self.cl_kernel_args["csr_medfilt"] =     OrderedDict((("output4", self.cl_mem["output4"]),
                                                              ("work4", self.cl_mem["work4"]),
                                                              ("data", self.cl_mem["data"]),
                                                              ("indices", self.cl_mem["indices"]),
                                                              ("indptr", self.cl_mem["indptr"]),
                                                              ("quant_min", numpy.float32(0.5)),
                                                              ("quant_max", numpy.float32(0.5)),
                                                              ("error_model", numpy.int8(1)),
                                                              ("empty", numpy.float32(self.empty)),
                                                              ("merged8", self.cl_mem["merged8"]),
                                                              ("averint", self.cl_mem["averint"]),
                                                              ("std", self.cl_mem["std"]),
                                                              ("sem", self.cl_mem["sem"]),
                                                              ("shared_int", pyopencl.LocalMemory(128)),
                                                              ("shared_float", pyopencl.LocalMemory(128)),
                                                             ))
        self.cl_kernel_args["memset_out"] = OrderedDict(((i, self.cl_mem[i]) for i in ("sum_data", "sum_count", "merged")))
        self.cl_kernel_args["memset_ng"] = OrderedDict(((i, self.cl_mem[i]) for i in ("averint", "std", "merged8")))
        self.cl_kernel_args["u8_to_float"] = OrderedDict(((i, self.cl_mem[i]) for i in ("tmp", "image")))
        self.cl_kernel_args["s8_to_float"] = OrderedDict(((i, self.cl_mem[i]) for i in ("tmp", "image")))
        self.cl_kernel_args["u16_to_float"] = OrderedDict(((i, self.cl_mem[i]) for i in ("tmp", "image")))
        self.cl_kernel_args["s16_to_float"] = OrderedDict(((i, self.cl_mem[i]) for i in ("tmp", "image")))
        self.cl_kernel_args["u32_to_float"] = OrderedDict(((i, self.cl_mem[i]) for i in ("tmp", "image")))
        self.cl_kernel_args["s32_to_float"] = OrderedDict(((i, self.cl_mem[i]) for i in ("tmp", "image")))

    def send_buffer(self, data, dest, checksum=None, workgroup_size=None, convert=True):
        """Send a numpy array to the device, including the type conversion on the device if possible

        :param data: numpy array with data
        :param dest: name of the buffer as registered in the class
        :param checksum: Checksum of the data to determine if the data needs to be transfered
        :param workgroup_size: enforce kernel to run with given workgroup size
        :param convert: if True (default) convert dtype on GPU, if false, leave as it is.
        :return: the actual buffer where the data were sent
        """
        dest_type = self.buffer_dtype[dest]
        events = []
        if isinstance(data, pyopencl.array.Array):
            if (data.dtype == dest_type):
                dest_buffer = self.cl_mem[dest]
                copy_image = pyopencl.enqueue_copy(self.queue, dest_buffer, data.data)
                events.append(EventDescription(f"copy D->D {dest}", copy_image))
            elif convert:
                tmp_buffer = self.cl_mem["tmp"]
                dest_buffer = self.cl_mem[dest]
                copy_image = pyopencl.enqueue_copy(self.queue, tmp_buffer, data.data)
                kernel_name = self.mapping[data.dtype.type]
                kernel = self.kernels.get_kernel(kernel_name)
                wg = workgroup_size if workgroup_size else max(self.workgroup_size[kernel_name])
                convert_to_float = kernel(self.queue, ((self.size + wg - 1) // wg * wg,), (wg,), tmp_buffer, dest_buffer)
                events += [EventDescription(f"copy raw D->D {dest}", copy_image),
                           EventDescription(f"convert {kernel_name}", convert_to_float)]
            else:  # no convert
                actual_dest = f"{dest}_raw"
                dest_buffer = self.cl_mem[actual_dest]
                if data.dtype.itemsize > dest_type.itemsize:
                    converted_data = data.astype(dest_type)
                else:
                    converted_data = data
                copy_image = pyopencl.enqueue_copy(self.queue, dest_buffer, converted_data.data)
                events.append(EventDescription(f"copy D->D {actual_dest}", copy_image))
        else:
            # Assume it is a numpy array
            if (data.dtype == dest_type) or (data.dtype.itemsize > dest_type.itemsize):
                dest_buffer = self.cl_mem[dest]
                copy_image = pyopencl.enqueue_copy(self.queue, dest_buffer, numpy.ascontiguousarray(data, dest_type))
                events.append(EventDescription(f"copy H->D {dest}", copy_image))
            elif convert:
                tmp_buffer = self.cl_mem["tmp"]
                dest_buffer = self.cl_mem[dest]
                copy_image = pyopencl.enqueue_copy(self.queue, tmp_buffer, numpy.ascontiguousarray(data))
                kernel_name = self.mapping[data.dtype.type]
                kernel = self.kernels.get_kernel(kernel_name)
                wg = workgroup_size if workgroup_size else max(self.workgroup_size[kernel_name])
                convert_to_float = kernel(self.queue, ((self.size + wg - 1) // wg * wg,), (wg,), tmp_buffer, dest_buffer)
                events += [EventDescription(f"copy raw H->D {dest}", copy_image),
                           EventDescription(f"convert {kernel_name}", convert_to_float)]
            else:
                actual_dest = f"{dest}_raw"
                dest_buffer = self.cl_mem[actual_dest]
                if data.dtype.itemsize > dest_type.itemsize:
                    converted_data = numpy.ascontiguousarray(data, dest_type)
                else:
                    converted_data = numpy.ascontiguousarray(data)
                copy_image = pyopencl.enqueue_copy(self.queue, dest_buffer, converted_data)
                events.append(EventDescription(f"copy H->D {actual_dest}", copy_image))
        self.profile_multi(events)
        if checksum is not None:
            self.on_device[dest] = checksum
        return dest_buffer

    def get_buffer(self, name, out=None):
        """retrive a Send a numpy array to the device, including the type conversion on the device if possible

        :param name: name of the buffer
        :param out: pre-allocated destination numpy array
        :return: the numpy array
        """
        if out is None:
            if name in self.cl_mem:
                for buf in self.buffers:
                    if buf.name == name:
                        shape = buf.size
                        dtype = buf.dtype
                        out = numpy.empty(shape, dtype)
            else:
                logger.error("No such buffer declared")

        ev = pyopencl.enqueue_copy(self.queue, out, self.cl_mem[name])
        self.events.append(EventDescription(f"copy D->H {name}", ev))
        return out

    def integrate_legacy(self, data, dummy=None, delta_dummy=None,
                         dark=None, flat=None, solidangle=None, polarization=None, absorption=None,
                         dark_checksum=None, flat_checksum=None, solidangle_checksum=None,
                         polarization_checksum=None, absorption_checksum=None,
                         preprocess_only=False, safe=True,
                         normalization_factor=1.0, coef_power=1,
                         out_merged=None, out_sum_data=None, out_sum_count=None):
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
        :param coef_power: set to 2 for variance propagation, leave to 1 for mean calculation
        :param out_merged: destination array or pyopencl array for averaged data
        :param out_sum_data: destination array or pyopencl array for sum of all data
        :param out_sum_count: destination array or pyopencl array for sum of the number of pixels
        :return: averaged data, weighted histogram, unweighted histogram
        """
        events = []
        with self.sem:
            self.send_buffer(data, "image")
            wg = max(self.workgroup_size["memset_out"])
            wdim_bins = int(self.bins + wg - 1) // wg * wg,
            memset = self.kernels.memset_out(self.queue, wdim_bins, (wg,), *list(self.cl_kernel_args["memset_out"].values()))
            events.append(EventDescription("memset_out", memset))
            kw_corr = self.cl_kernel_args["corrections"]
            kw_int = self.cl_kernel_args["csr_integrate"]

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

            kw_corr["do_dummy"] = do_dummy
            kw_corr["dummy"] = dummy
            kw_corr["delta_dummy"] = delta_dummy
            kw_corr["normalization_factor"] = numpy.float32(normalization_factor)
            kw_int["do_dummy"] = do_dummy
            kw_int["dummy"] = dummy
            kw_int["coef_power"] = numpy.int32(coef_power)

            if dark is not None:
                do_dark = numpy.int8(1)
                # TODO: what is do_checksum=False and image not on device ...
                if not dark_checksum:
                    dark_checksum = calc_checksum(dark, safe)
                if dark_checksum != self.on_device["dark"]:
                    self.send_buffer(dark, "dark", dark_checksum)
            else:
                do_dark = numpy.int8(0)
            kw_corr["do_dark"] = do_dark

            if flat is not None:
                do_flat = numpy.int8(1)
                if not flat_checksum:
                    flat_checksum = calc_checksum(flat, safe)
                if self.on_device["flat"] != flat_checksum:
                    self.send_buffer(flat, "flat", flat_checksum)
            else:
                do_flat = numpy.int8(0)
            kw_corr["do_flat"] = do_flat

            if solidangle is not None:
                do_solidangle = numpy.int8(1)
                if not solidangle_checksum:
                    solidangle_checksum = calc_checksum(solidangle, safe)
                if solidangle_checksum != self.on_device["solidangle"]:
                    self.send_buffer(solidangle, "solidangle", solidangle_checksum)
            else:
                do_solidangle = numpy.int8(0)
            kw_corr["do_solidangle"] = do_solidangle

            if polarization is not None:
                do_polarization = numpy.int8(1)
                if not polarization_checksum:
                    polarization_checksum = calc_checksum(polarization, safe)
                if polarization_checksum != self.on_device["polarization"]:
                    self.send_buffer(polarization, "polarization", polarization_checksum)
                    self.on_device["polarization"] = polarization_checksum
            else:
                do_polarization = numpy.int8(0)
            kw_corr["do_polarization"] = do_polarization

            if absorption is not None:
                do_absorption = numpy.int8(1)
                if not absorption_checksum:
                    absorption_checksum = calc_checksum(absorption, safe)
                if absorption_checksum != self.on_device["absorption"]:
                    self.send_buffer(absorption, "absorption", absorption_checksum)
            else:
                do_absorption = numpy.int8(0)
            kw_corr["do_absorption"] = do_absorption

            wg = max(self.workgroup_size["corrections"])
            wdim_data = int(self.size + wg - 1) // wg * wg,
            ev = self.kernels.corrections(self.queue, wdim_data, (wg,), *list(kw_corr.values()))
            events.append(EventDescription("corrections", ev))

            if preprocess_only:
                image = numpy.empty(data.shape, dtype=numpy.float32)
                ev = pyopencl.enqueue_copy(self.queue, image, self.cl_mem["output"])
                events.append(EventDescription("copy D->H image", ev))
                ev.wait()
                self.profile_multi(events)
                return image

            wg_min, wg_max = self.workgroup_size["csr_integrate"]
            if wg_max == 1:
                # thread-synchronization is probably not possible.
                wg_max = max(self.workgroup_size["csr_integrate_single"])
                wdim_bins = int(self.bins + wg_max - 1) // wg_max * wg_max,
                integrate = self.kernels.csr_integrate_single(self.queue, wdim_bins, (wg_max,), *kw_int.values())
                events.append(EventDescription("csr_integrate_single", integrate))
            else:
                #TODO reshape this kernel with (wg, nbin)\(wg,1) rather than (self.bins * wg_min)\(wg_min) #2348
                wdim_bins = (self.bins * wg_min),
                kw_int["shared"] = pyopencl.LocalMemory(16 * wg_min)  # sizeof(float4) == 16
                integrate = self.kernels.csr_integrate(self.queue, wdim_bins, (wg_min,), *kw_int.values())
                events.append(EventDescription("csr_integrate", integrate))
            if out_merged is None:
                merged = numpy.empty(self.bins, dtype=numpy.float32)
            elif out_merged is False:
                merged = None
            else:
                merged = out_merged.data
            if out_sum_count is None:
                sum_count = numpy.empty(self.bins, dtype=numpy.float32)
            elif out_sum_count is False:
                sum_count = None
            else:
                sum_count = out_sum_count.data
            if out_sum_data is None:
                sum_data = numpy.empty(self.bins, dtype=numpy.float32)
            elif out_sum_data is False:
                sum_data = None
            else:
                sum_data = out_sum_data.data

            if merged is not None:
                ev = pyopencl.enqueue_copy(self.queue, merged, self.cl_mem["merged"])
                events.append(EventDescription("copy D->H merged", ev))
            if sum_data is not None:
                ev = pyopencl.enqueue_copy(self.queue, sum_data, self.cl_mem["sum_data"])
                events.append(EventDescription("copy D->H sum_data", ev))
            if sum_count is not None:
                ev = pyopencl.enqueue_copy(self.queue, sum_count, self.cl_mem["sum_count"])
                events.append(EventDescription("copy D->H sum_count", ev))
            ev.wait()
        self.profile_multi(events)
        return merged, sum_data, sum_count

    def integrate_ng(self, data, dark=None, dummy=None, delta_dummy=None,
                     error_model=ErrorModel.NO, variance=None, dark_variance=None,
                     flat=None, solidangle=None, polarization=None, absorption=None,
                     dark_checksum=None, flat_checksum=None, solidangle_checksum=None,
                     polarization_checksum=None, absorption_checksum=None, dark_variance_checksum=None,
                     safe=True, workgroup_size=None,
                     normalization_factor=1.0, weighted_average=True,
                     out_avgint=None, out_sem=None, out_std=None, out_merged=None):
        """
        Before performing azimuthal integration with proper variance propagation, the preprocessing is:

        .. math::

            signal = (raw - dark)
            variance = variance + dark_variance
            normalization  = normalization_factor*(flat * solidangle * polarization * absortoption)
            count = number of pixel contributing

        Integration is performed using the CSR representation of the look-up table on all
        arrays: signal, variance, normalization and count

        :param dark: array of same shape as data for pre-processing
        :param dummy: value for invalid data
        :param delta_dummy: precesion for dummy assessement
        :param error_model: enum ErrorModel
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
        :param workgroup_size: enforce this workgroup size
        :param preprocess_only: return the dark subtracted; flat field & solidangle & polarization corrected image, else
        :param normalization_factor: divide raw signal by this value
        :param bool weighted_average: set to False to use an unweighted mean (similar to legacy) instead of the weighted average. WIP
        :param out_avgint: destination array or pyopencl array for average intensity
        :param out_sem: destination array or pyopencl array for standard deviation (of mean)
        :param out_std: destination array or pyopencl array for standard deviation (of pixels)
        :param out_merged: destination array or pyopencl array for averaged data (float8!)

        :return: named-tuple
        """
        events = []
        with self.sem:
            kernel_correction_name = "corrections4a"
            corrections4 = self.kernels.corrections4a
            kw_corr = self.cl_kernel_args[kernel_correction_name]
            kw_corr["image"] = self.send_buffer(data, "image", convert=False)
            kw_corr["dtype"] = numpy.int8(32) if data.dtype.itemsize > 4 else dtype_converter(data.dtype)
            wg = workgroup_size if workgroup_size else max(self.workgroup_size["memset_ng"])
            wdim_bins = (self.bins + wg - 1) // wg * wg,
            memset = self.kernels.memset_out(self.queue, wdim_bins, (wg,), *list(self.cl_kernel_args["memset_ng"].values()))
            events.append(EventDescription("memset_ng", memset))
            kw_int = self.cl_kernel_args["csr_integrate4"]

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

            kw_corr["do_dummy"] = do_dummy
            kw_corr["dummy"] = dummy
            kw_corr["delta_dummy"] = delta_dummy
            kw_corr["normalization_factor"] = numpy.float32(normalization_factor)
            kw_corr["apply_normalization"] = numpy.int8(not weighted_average)
            kw_int["error_model"] = kw_corr["error_model"] = numpy.int8(error_model.value)
            if variance is not None:
                self.send_buffer(variance, "variance", workgroup_size=workgroup_size)
            if dark_variance is not None:
                if not dark_variance_checksum:
                    dark_variance_checksum = calc_checksum(dark_variance, safe)
                if dark_variance_checksum != self.on_device["dark_variance"]:
                    self.send_buffer(dark_variance, "dark_variance", dark_variance_checksum, workgroup_size=workgroup_size)
            else:
                do_dark = numpy.int8(0)
            kw_corr["do_dark"] = do_dark

            if dark is not None:
                do_dark = numpy.int8(1)
                # TODO: what is do_checksum=False and image not on device ...
                if not dark_checksum:
                    dark_checksum = calc_checksum(dark, safe)
                if dark_checksum != self.on_device["dark"]:
                    self.send_buffer(dark, "dark", dark_checksum, workgroup_size=workgroup_size)
            else:
                do_dark = numpy.int8(0)
            kw_corr["do_dark"] = do_dark

            if flat is not None:
                do_flat = numpy.int8(1)
                if not flat_checksum:
                    flat_checksum = calc_checksum(flat, safe)
                if self.on_device["flat"] != flat_checksum:
                    self.send_buffer(flat, "flat", flat_checksum, workgroup_size=workgroup_size)
            else:
                do_flat = numpy.int8(0)
            kw_corr["do_flat"] = do_flat

            if solidangle is not None:
                do_solidangle = numpy.int8(1)
                if not solidangle_checksum:
                    solidangle_checksum = calc_checksum(solidangle, safe)
                if solidangle_checksum != self.on_device["solidangle"]:
                    self.send_buffer(solidangle, "solidangle", solidangle_checksum, workgroup_size=workgroup_size)
            else:
                do_solidangle = numpy.int8(0)
            kw_corr["do_solidangle"] = do_solidangle

            if polarization is not None:
                do_polarization = numpy.int8(1)
                if not polarization_checksum:
                    polarization_checksum = calc_checksum(polarization, safe)
                if polarization_checksum != self.on_device["polarization"]:
                    self.send_buffer(polarization, "polarization", polarization_checksum, workgroup_size=workgroup_size)
            else:
                do_polarization = numpy.int8(0)
            kw_corr["do_polarization"] = do_polarization

            if absorption is not None:
                do_absorption = numpy.int8(1)
                if not absorption_checksum:
                    absorption_checksum = calc_checksum(absorption, safe)
                if absorption_checksum != self.on_device["absorption"]:
                    self.send_buffer(absorption, "absorption", absorption_checksum, workgroup_size=workgroup_size)
            else:
                do_absorption = numpy.int8(0)
            kw_corr["do_absorption"] = do_absorption

            wg = workgroup_size if workgroup_size else max(self.workgroup_size[kernel_correction_name])
            wdim_data = (self.size + wg - 1) // wg * wg ,
            ev = corrections4(self.queue, wdim_data, (wg,), *list(kw_corr.values()))
            events.append(EventDescription(kernel_correction_name, ev))

            kw_int["empty"] = dummy
            wg_min, wg_max = (workgroup_size, workgroup_size) if workgroup_size else self.workgroup_size["csr_integrate4"]
            if  wg_max == 1:
                wg = workgroup_size if workgroup_size else max(self.workgroup_size["csr_integrate4_single"])
                wdim_bins = (self.bins + wg - 1) // wg * wg,
                integrate = self.kernels.csr_integrate4_single(self.queue, wdim_bins, (wg,), *kw_int.values())
                events.append(EventDescription("csr_integrate4_single", integrate))
            else:
                wdim_bins = (self.bins * wg_min),
                kw_int["shared"] = pyopencl.LocalMemory(32 * wg_min)  # sizeof(float8) == 32
                integrate = self.kernels.csr_integrate4(self.queue, wdim_bins, (wg_min,), *kw_int.values())
                events.append(EventDescription("csr_integrate4", integrate))

            if out_merged is None:
                merged = numpy.empty((self.bins, 8), dtype=numpy.float32)
            elif out_merged is False:
                merged = None
            else:
                merged = out_merged.data

            if out_avgint is None:
                avgint = numpy.empty(self.bins, dtype=numpy.float32)
            elif out_avgint is False:
                avgint = None
            else:
                avgint = out_avgint.data

            if out_sem is None:
                sem = numpy.empty(self.bins, dtype=numpy.float32)
            elif out_sem is False:
                sem = None
            else:
                sem = out_sem.data

            if out_std is None:
                std = numpy.empty(self.bins, dtype=numpy.float32)
            elif out_std is False:
                std = None
            else:
                std = out_std.data

            if avgint is not None:
                ev = pyopencl.enqueue_copy(self.queue, avgint, self.cl_mem["averint"])
                events.append(EventDescription("copy D->H avgint", ev))
            if std is not None:
                ev = pyopencl.enqueue_copy(self.queue, std, self.cl_mem["std"])
                events.append(EventDescription("copy D->H std", ev))
            if sem is not None:
                ev = pyopencl.enqueue_copy(self.queue, sem, self.cl_mem["sem"])
                events.append(EventDescription("copy D->H sem", ev))

            if self.azim_centers is None:  # 1D case
                if merged is None:
                    # position intensity sigma signal variance normalization count std sem norm_sq
                    res = Integrate1dtpl(self.bin_centers, avgint, sem, None, None, None, None, std, sem, None)
                else:
                    ev = pyopencl.enqueue_copy(self.queue, merged, self.cl_mem["merged8"])
                    events.append(EventDescription("copy D->H merged8", ev))
                    res = Integrate1dtpl(self.bin_centers, avgint, sem, merged[:, 0], merged[:, 2], merged[:, 4], merged[:, 6],
                                         std, sem, merged[:, 7])
            else:  # 2D case
                outshape = self.bin_centers.size, self.azim_centers.size
                std = std.reshape(outshape).T if std is not None else None
                sem = sem.reshape(outshape).T if sem is not None else None
                if merged is None:  # "radial azimuthal intensity sigma signal variance normalization count std sem norm_sq"
                    res = Integrate2dtpl(self.bin_centers, self.azim_centers,
                                         avgint.reshape(outshape).T, sem,
                                         None, None, None, None, std, sem)
                else:
                    ev = pyopencl.enqueue_copy(self.queue, merged, self.cl_mem["merged8"])
                    events.append(EventDescription("copy D->H merged8", ev))
                    res = Integrate2dtpl(self.bin_centers, self.azim_centers,
                                         avgint.reshape(outshape).T, sem,
                                         merged[:, 0].reshape(outshape).T, merged[:, 2].reshape(outshape).T, merged[:, 4].reshape(outshape).T, merged[:, 6].reshape(outshape).T,
                                         std, sem, merged[:, 7].reshape(outshape).T)

        self.profile_multi(events)
        return res

    integrate = integrate_ng

    def sigma_clip(self, data, dark=None, dummy=None, delta_dummy=None,
                   variance=None, dark_variance=None,
                   flat=None, solidangle=None, polarization=None, absorption=None,
                   dark_checksum=None, flat_checksum=None, solidangle_checksum=None,
                   polarization_checksum=None, absorption_checksum=None, dark_variance_checksum=None,
                   safe=True, error_model=ErrorModel.NO,
                   normalization_factor=1.0,
                   cutoff=4.0, cycle=5,
                   out_avgint=None, out_sem=None, out_std=None, out_merged=None):
        """
        Perform a sigma-clipping iterative filter within each along each row.
        see the doc of scipy.stats.sigmaclip for more descriptions.

        If the error model is "azimuthal": the variance is the variance within a bin,
        which is refined at each iteration, can be costly !

        Else, the error is propagated according to:

        .. math::

            signal = (raw - dark)
            variance = variance + dark_variance
            normalization  = normalization_factor*(flat * solidangle * polarization * absortoption)
            count = number of pixel contributing

        Integration is performed using the CSR representation of the look-up table on all
        arrays: signal, variance, normalization and count

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
        :param error_model: enum ErrorModel
        :param normalization_factor: divide raw signal by this value
        :param cutoff: discard all points with ``|value - avg| > cutoff * sigma``. 3-4 is quite common
        :param cycle: perform at maximum this number of cycles. 5 is common.
        :param out_avgint: destination array or pyopencl array for sum of all data
        :param out_sem: destination array or pyopencl array for uncertainty on mean value
        :param out_std: destination array or pyopencl array for uncertainty on pixel value
        :param out_merged: destination array or pyopencl array for averaged data (float8!)
        :return: namedtuple with "position intensity error signal variance normalization count"
        """
        error_model = ErrorModel.parse(error_model)
        events = []
        with self.sem:
            kernel_correction_name = "corrections4a"
            corrections4 = self.kernels.corrections4a
            kw_corr = self.cl_kernel_args[kernel_correction_name]
            kw_corr["image"] = self.send_buffer(data, "image", convert=False)
            kw_corr["dtype"] = numpy.int8(32) if data.dtype.itemsize > 4 else dtype_converter(data.dtype)
            wg = max(self.workgroup_size["memset_ng"])
            wdim_bins = int(self.bins + wg - 1) // wg * wg,
            memset = self.kernels.memset_out(self.queue, wdim_bins, (wg,), *list(self.cl_kernel_args["memset_ng"].values()))
            events.append(EventDescription("memset_ng", memset))
            kw_int = self.cl_kernel_args["csr_sigma_clip4"]

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

            kw_corr["do_dummy"] = do_dummy
            kw_corr["dummy"] = dummy
            kw_corr["delta_dummy"] = delta_dummy
            kw_corr["normalization_factor"] = numpy.float32(normalization_factor)

            if variance is not None:
                error_model = ErrorModel.VARIANCE
                self.send_buffer(variance, "variance")
            kw_int["error_model"] = kw_corr["error_model"] = numpy.int8(error_model)
            if dark_variance is not None:
                if not dark_variance_checksum:
                    dark_variance_checksum = calc_checksum(dark_variance, safe)
                if dark_variance_checksum != self.on_device["dark_variance"]:
                    self.send_buffer(dark_variance, "dark_variance", dark_variance_checksum)
            else:
                do_dark = numpy.int8(0)
            kw_corr["do_dark"] = do_dark

            if dark is not None:
                do_dark = numpy.int8(1)
                # TODO: what is do_checksum=False and image not on device ...
                if not dark_checksum:
                    dark_checksum = calc_checksum(dark, safe)
                if dark_checksum != self.on_device["dark"]:
                    self.send_buffer(dark, "dark", dark_checksum)
            else:
                do_dark = numpy.int8(0)
            kw_corr["do_dark"] = do_dark

            if flat is not None:
                do_flat = numpy.int8(1)
                if not flat_checksum:
                    flat_checksum = calc_checksum(flat, safe)
                if self.on_device["flat"] != flat_checksum:
                    self.send_buffer(flat, "flat", flat_checksum)
            else:
                do_flat = numpy.int8(0)
            kw_corr["do_flat"] = do_flat

            if solidangle is not None:
                do_solidangle = numpy.int8(1)
                if not solidangle_checksum:
                    solidangle_checksum = calc_checksum(solidangle, safe)
                if solidangle_checksum != self.on_device["solidangle"]:
                    self.send_buffer(solidangle, "solidangle", solidangle_checksum)
            else:
                do_solidangle = numpy.int8(0)
            kw_corr["do_solidangle"] = do_solidangle

            if polarization is not None:
                do_polarization = numpy.int8(1)
                if not polarization_checksum:
                    polarization_checksum = calc_checksum(polarization, safe)
                if polarization_checksum != self.on_device["polarization"]:
                    self.send_buffer(polarization, "polarization", polarization_checksum)
            else:
                do_polarization = numpy.int8(0)
            kw_corr["do_polarization"] = do_polarization

            if absorption is not None:
                do_absorption = numpy.int8(1)
                if not absorption_checksum:
                    absorption_checksum = calc_checksum(absorption, safe)
                if absorption_checksum != self.on_device["absorption"]:
                    self.send_buffer(absorption, "absorption", absorption_checksum)
            else:
                do_absorption = numpy.int8(0)
            kw_corr["do_absorption"] = do_absorption

            wg = max(self.workgroup_size[kernel_correction_name])
            wdim_data = int(self.size + wg - 1) // wg * wg,
            ev = corrections4(self.queue, wdim_data, (wg,), *list(kw_corr.values()))
            events.append(EventDescription(kernel_correction_name, ev))

            kw_int["cutoff"] = numpy.float32(cutoff)
            kw_int["cycle"] = numpy.int32(cycle)

            wg_min = min(self.workgroup_size["csr_sigma_clip4"])
            kw_int["shared"] = pyopencl.LocalMemory(32 * wg_min)
            wdim_bins = (self.bins * wg_min),
            integrate = self.kernels.csr_sigma_clip4(self.queue, wdim_bins, (wg_min,), *kw_int.values())
            events.append(EventDescription("csr_sigma_clip4", integrate))

            if out_merged is None:
                merged = numpy.empty((self.bins, 8), dtype=numpy.float32)
            else:
                merged = out_merged.data
            if out_avgint is None:
                avgint = numpy.empty(self.bins, dtype=numpy.float32)
            else:
                avgint = out_avgint.data
            if out_sem is None:
                sem = numpy.empty(self.bins, dtype=numpy.float32)
            elif out_sem is  False:
                sem = None
            else:
                sem = out_sem.data

            if out_std is None:
                std = numpy.empty(self.bins, dtype=numpy.float32)
            elif out_std is  False:
                std = None
            else:
                std = out_std.data

            if avgint is not None:
                ev = pyopencl.enqueue_copy(self.queue, avgint, self.cl_mem["averint"])
                events.append(EventDescription("copy D->H avgint", ev))
            if std is not None:
                ev = pyopencl.enqueue_copy(self.queue, std, self.cl_mem["std"])
                events.append(EventDescription("copy D->H std", ev))
            if sem is not None:
                ev = pyopencl.enqueue_copy(self.queue, sem, self.cl_mem["sem"])
                events.append(EventDescription("copy D->H sem", ev))

            ev = pyopencl.enqueue_copy(self.queue, merged, self.cl_mem["merged8"])
            events.append(EventDescription("copy D->H merged8", ev))
        self.profile_multi(events)
        res = Integrate1dtpl(self.bin_centers, avgint, sem, merged[:, 0], merged[:, 2], merged[:, 4], merged[:, 6],
                             std, sem, merged[:, 7])
        return res

    def medfilt(self, data, dark=None, dummy=None, delta_dummy=None,
                   variance=None, dark_variance=None,
                   flat=None, solidangle=None, polarization=None, absorption=None,
                   dark_checksum=None, flat_checksum=None, solidangle_checksum=None,
                   polarization_checksum=None, absorption_checksum=None, dark_variance_checksum=None,
                   safe=True, error_model=ErrorModel.NO,
                   normalization_factor=1.0,
                   quant_min=0.5, quant_max=0.5,
                   out_avgint=None, out_sem=None, out_std=None, out_merged=None):
        """
        Perform a median-filter/quantile mean in azimuthal space.


        Averaging is performed using the CSR representation of the look-up table on all
        arrays after sorting pixels by apparant intensity and taking only the selected ones
        based on quantiles and the length of the ensemble.

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
        :param error_model: enum ErrorModel
        :param normalization_factor: divide raw signal by this value
        :param quant_min: start percentile/100 to use. Use 0.5 for the median (default). 0<=quant_min<=1
        :param quant_max: stop percentile/100 to use. Use 0.5 for the median (default). 0<=quant_max<=1
        :param out_avgint: destination array or pyopencl array for sum of all data
        :param out_sem: destination array or pyopencl array for uncertainty on mean value
        :param out_std: destination array or pyopencl array for uncertainty on pixel value
        :param out_merged: destination array or pyopencl array for averaged data (float8!)
        :return: namedtuple with "position intensity error signal variance normalization count"
        """
        error_model = ErrorModel.parse(error_model)
        events = []
        with self.sem:
            kernel_correction_name = "corrections4a"
            corrections4 = self.kernels.corrections4a
            kw_corr = self.cl_kernel_args[kernel_correction_name]
            kw_corr["image"] = self.send_buffer(data, "image", convert=False)
            kw_corr["dtype"] = numpy.int8(32) if data.dtype.itemsize > 4 else dtype_converter(data.dtype)
            wg = max(self.workgroup_size["memset_ng"])
            wdim_bins = int(self.bins + wg - 1) // wg * wg,
            memset = self.kernels.memset_out(self.queue, wdim_bins, (wg,), *list(self.cl_kernel_args["memset_ng"].values()))
            events.append(EventDescription("memset_ng", memset))
            kw_int = self.cl_kernel_args["csr_medfilt"]

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

            kw_corr["do_dummy"] = do_dummy
            kw_corr["dummy"] = dummy
            kw_corr["delta_dummy"] = delta_dummy
            kw_corr["normalization_factor"] = numpy.float32(normalization_factor)

            if variance is not None:
                error_model = ErrorModel.VARIANCE
                self.send_buffer(variance, "variance")
            kw_int["error_model"] = kw_corr["error_model"] = numpy.int8(error_model)
            if dark_variance is not None:
                if not dark_variance_checksum:
                    dark_variance_checksum = calc_checksum(dark_variance, safe)
                if dark_variance_checksum != self.on_device["dark_variance"]:
                    self.send_buffer(dark_variance, "dark_variance", dark_variance_checksum)
            else:
                do_dark = numpy.int8(0)
            kw_corr["do_dark"] = do_dark

            if dark is not None:
                do_dark = numpy.int8(1)
                # TODO: what is do_checksum=False and image not on device ...
                if not dark_checksum:
                    dark_checksum = calc_checksum(dark, safe)
                if dark_checksum != self.on_device["dark"]:
                    self.send_buffer(dark, "dark", dark_checksum)
            else:
                do_dark = numpy.int8(0)
            kw_corr["do_dark"] = do_dark

            if flat is not None:
                do_flat = numpy.int8(1)
                if not flat_checksum:
                    flat_checksum = calc_checksum(flat, safe)
                if self.on_device["flat"] != flat_checksum:
                    self.send_buffer(flat, "flat", flat_checksum)
            else:
                do_flat = numpy.int8(0)
            kw_corr["do_flat"] = do_flat

            if solidangle is not None:
                do_solidangle = numpy.int8(1)
                if not solidangle_checksum:
                    solidangle_checksum = calc_checksum(solidangle, safe)
                if solidangle_checksum != self.on_device["solidangle"]:
                    self.send_buffer(solidangle, "solidangle", solidangle_checksum)
            else:
                do_solidangle = numpy.int8(0)
            kw_corr["do_solidangle"] = do_solidangle

            if polarization is not None:
                do_polarization = numpy.int8(1)
                if not polarization_checksum:
                    polarization_checksum = calc_checksum(polarization, safe)
                if polarization_checksum != self.on_device["polarization"]:
                    self.send_buffer(polarization, "polarization", polarization_checksum)
            else:
                do_polarization = numpy.int8(0)
            kw_corr["do_polarization"] = do_polarization

            if absorption is not None:
                do_absorption = numpy.int8(1)
                if not absorption_checksum:
                    absorption_checksum = calc_checksum(absorption, safe)
                if absorption_checksum != self.on_device["absorption"]:
                    self.send_buffer(absorption, "absorption", absorption_checksum)
            else:
                do_absorption = numpy.int8(0)
            kw_corr["do_absorption"] = do_absorption

            wg = max(self.workgroup_size[kernel_correction_name])
            wdim_data = int(self.size + wg - 1) // wg * wg,
            ev = corrections4(self.queue, wdim_data, (wg,), *list(kw_corr.values()))
            events.append(EventDescription(kernel_correction_name, ev))

            kw_int["quant_min"] = numpy.float32(quant_min)
            kw_int["quant_max"] = numpy.float32(quant_max)
            wg_min = max(self.workgroup_size["csr_medfilt"])
            kw_int["shared_int"] = pyopencl.LocalMemory(4 * wg_min)
            kw_int["shared_float"] = pyopencl.LocalMemory(8 * wg_min)
            wdim_bins = (wg_min, self.bins)


            integrate = self.kernels.csr_medfilt(self.queue, wdim_bins, (wg_min,1), *kw_int.values())
            events.append(EventDescription("medfilt", integrate))

            if out_merged is None:
                merged = numpy.empty((self.bins, 8), dtype=numpy.float32)
            else:
                merged = out_merged.data
            if out_avgint is None:
                avgint = numpy.empty(self.bins, dtype=numpy.float32)
            else:
                avgint = out_avgint.data
            if out_sem is None:
                sem = numpy.empty(self.bins, dtype=numpy.float32)
            elif out_sem is  False:
                sem = None
            else:
                sem = out_sem.data

            if out_std is None:
                std = numpy.empty(self.bins, dtype=numpy.float32)
            elif out_std is  False:
                std = None
            else:
                std = out_std.data

            if avgint is not None:
                ev = pyopencl.enqueue_copy(self.queue, avgint, self.cl_mem["averint"])
                events.append(EventDescription("copy D->H avgint", ev))
            if std is not None:
                ev = pyopencl.enqueue_copy(self.queue, std, self.cl_mem["std"])
                events.append(EventDescription("copy D->H std", ev))
            if sem is not None:
                ev = pyopencl.enqueue_copy(self.queue, sem, self.cl_mem["sem"])
                events.append(EventDescription("copy D->H sem", ev))

            ev = pyopencl.enqueue_copy(self.queue, merged, self.cl_mem["merged8"])
            events.append(EventDescription("copy D->H merged8", ev))
        self.profile_multi(events)

        res = Integrate1dtpl(self.bin_centers, avgint, sem, merged[:, 0], merged[:, 2], merged[:, 4], merged[:, 6],
                             std, sem, merged[:, 7])
        return res


    # Name of the default "process" method
    __call__ = integrate
