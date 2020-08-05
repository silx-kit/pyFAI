# -*- coding: utf-8 -*-
#
#    Project: Peak finder in a single 2D diffraction frame
#             https://github.com/silx-kit/pyFAI
#
#
#    Copyright (C) 2014-2019 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
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

__authors__ = ["Jérôme Kieffer"]
__license__ = "MIT"
__date__ = "04/08/2020"
__copyright__ = "2014-2019, ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import logging
from collections import OrderedDict
import numpy
import math
from ..utils import EPS32
from .azim_csr import OCL_CSR_Integrator, BufferDescription, EventDescription, mf, calc_checksum, pyopencl, OpenclProcessing
from . import get_x87_volatile_option
from . import kernel_workgroup_size

logger = logging.getLogger(__name__)


class OCL_PeakFinder(OCL_CSR_Integrator):
    BLOCK_SIZE = 512  # unlike in OCL_CSR_Integrator, here we need large blocks
    buffers = [BufferDescription("output", 1, numpy.float32, mf.WRITE_ONLY),
               BufferDescription("output4", 4, numpy.float32, mf.READ_WRITE),
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
               BufferDescription("peak_position", 1, numpy.int32, mf.READ_WRITE),
               BufferDescription("radius2d", 1, numpy.float32, mf.READ_ONLY),
               ]
    kernel_files = ["pyfai:openCL/kahan.cl",
                    "pyfai:openCL/preprocess.cl",
                    "pyfai:openCL/memset.cl",
                    "pyfai:openCL/ocl_azim_CSR.cl",
                    "pyfai:openCL/peak_finder.cl"
                    ]

    def __init__(self, lut, image_size, checksum=None,
                 empty=None, unit=None, bin_centers=None,
                 radius=None, mask=None,
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
        :param unit: Storage for the unit related to the LUT
        :param bin_centers: the radial position of the bin_center
        :param radius: radial position of every-pixel center.
        :param ctx: actual working context, left to None for automatic
                    initialization from device type or platformid/deviceid
        :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
        :param platformid: integer with the platform_identifier, as given by clinfo
        :param deviceid: Integer with the device identifier, as given by clinfo
        :param block_size: preferred workgroup size, may vary depending on the outpcome of the compilation
        :param profile: switch on profiling to be able to profile at the kernel level,
                        store profiling elements (makes code slightly slower)
        """
        self.radius = radius
        if mask is not None:
            self.mask = numpy.ascontiguousarray(mask, "int8")
        else:
            self.mask = None
        assert image_size == radius.size
        nbin = lut[2].size - 1
        self.buffers += [BufferDescription("radius1d", nbin, numpy.float32, mf.READ_ONLY),
                         BufferDescription("counter", 1, numpy.float32, mf.WRITE_ONLY),
                         ]
        if radius is None:
            raise RuntimeError("2D radius position is mandatory")
        if bin_centers is None:
            raise RuntimeError("1D bin center position is mandatory")
        OCL_CSR_Integrator.__init__(self, lut, image_size, checksum,
                 empty, unit, bin_centers,
                 ctx, devicetype, platformid, deviceid,
                 block_size, profile)

        if self.mask is not None:
            self.send_buffer(self.mask, "mask")
            self.cl_kernel_args["corrections4"]["do_mask"] = numpy.int8(1)
        if self.bin_centers is not None:
            self.send_buffer(self.bin_centers, "radius1d")
        if self.radius is not None:
            self.send_buffer(self.radius, "radius2d")

    def set_kernel_arguments(self):
        OCL_CSR_Integrator.set_kernel_arguments(self)
        self.cl_kernel_args["find_peaks"] = OrderedDict((("output4", self.cl_mem["output4"]),
                                                          ("radius2d", self.cl_mem["radius2d"]),
                                                          ("radius1d", self.cl_mem["radius1d"]),
                                                          ("averint", self.cl_mem["averint"]),
                                                          ("stderr", self.cl_mem["stderr"]),
                                                          ("radius_min", numpy.float32(0.0)),
                                                          ("radius_max", numpy.float32(numpy.finfo(numpy.float32).max)),
                                                          ("cutoff", numpy.float32(5.0)),
                                                          ("noise", numpy.float32(1.0)),
                                                          ("counter", self.cl_mem["counter"]),
                                                          ("peak_position", self.cl_mem["peak_position"])))

    def peak_finder(self, data, dark=None, dummy=None, delta_dummy=None,
                   variance=None, dark_variance=None,
                   flat=None, solidangle=None, polarization=None, absorption=None,
                   dark_checksum=None, flat_checksum=None, solidangle_checksum=None,
                   polarization_checksum=None, absorption_checksum=None, dark_variance_checksum=None,
                   safe=True, error_model=None,
                   normalization_factor=1.0,
                   cutoff=4.0, cycle=5, noise=1,
                   radial_range=None,
                   out_avgint=None, out_stderr=None, out_merged=None):
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
        :param normalization_factor: divide raw signal by this value
        :param cutoff: discard all points with |value - avg| > cutoff * sigma. 3-4 is quite common 
        :param cycle: perform at maximum this number of cycles. 5 is common.
        :param out_avgint: destination array or pyopencl array for sum of all data
        :param out_stderr: destination array or pyopencl array for sum of the number of pixels
        :param out_merged: destination array or pyopencl array for averaged data (float8!)
        :return: out_avgint, out_stderr, out_merged
        """
        events = []
        if isinstance(error_model, str):
            error_model = error_model.lower()
        else:
            error_model = ""
        with self.sem:
            self.send_buffer(data, "image")
            wg = self.workgroup_size["memset_ng"]
            wdim_bins = (self.bins + wg[0] - 1) & ~(wg[0] - 1),
            memset = self.kernels.memset_out(self.queue, wdim_bins, wg, *list(self.cl_kernel_args["memset_ng"].values()))
            events.append(EventDescription("memset_ng", memset))
            kw_corr = self.cl_kernel_args["corrections4"]
            kw_int = self.cl_kernel_args["csr_sigma_clip4"]
            kw_proj = self.cl_kernel_args["find_peaks"]

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
                self.send_buffer(variance, "variance")
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

            if error_model.startswith("poisson"):
                kw_corr["poissonian"] = numpy.int8(1)
            else:
                kw_corr["poissonian"] = numpy.int8(0)
            wg = self.workgroup_size["corrections4"]
            ev = self.kernels.corrections4(self.queue, self.wdim_data, wg, *list(kw_corr.values()))
            events.append(EventDescription("corrections", ev))

            wg = self.workgroup_size["csr_sigma_clip4"][0]
            kw_proj["cutoff"] = kw_int["cutoff"] = numpy.float32(cutoff)
            kw_int["cycle"] = numpy.int32(cycle)
            if error_model.startswith("azim"):
                kw_int["azimuthal"] = numpy.int32(1)
            else:
                kw_int["azimuthal"] = numpy.int32(0)

            wdim_bins = (self.bins * wg),
            if wg == 1:
                raise RuntimeError("csr_sigma_clip4 is not yet available in single threaded OpenCL !")
                integrate = self.kernels.csr_integrate4_single(self.queue, wdim_bins, (wg,), *kw_int.values())
                events.append(EventDescription("integrate4_single", integrate))
            else:
                integrate = self.kernels.csr_sigma_clip4(self.queue, wdim_bins, (wg,), *kw_int.values())
                events.append(EventDescription("csr_sigma_clip4", integrate))

            # now perform the calc_from_1d on the device and count the number of pixels
            memset1 = self.program.memset_int(self.queue, self.wdim_data, self.workgroup_size["corrections"], self.cl_mem["peak_position"], numpy.int32(0), numpy.int32(self.size))
            memset2 = self.program.memset_int(self.queue, (1,), (1,), self.cl_mem["counter"], numpy.int32(0), numpy.int32(1))
            events += [EventDescription("memset peak_position", memset1), EventDescription("memset counter", memset2)]

            if radial_range is not None:
                kw_proj["radius_min"] = numpy.float32(min(radial_range))
                kw_proj["radius_max"] = numpy.float32(max(radial_range) * EPS32)
            else:
                kw_proj["radius_min"] = numpy.float32(0.0)
                kw_proj["radius_max"] = numpy.float32(numpy.finfo(numpy.float32).max)

            kw_proj["noise"] = numpy.float32(noise)

            print("Call find_peaks", self.wdim_data, self.workgroup_size)
            for k, v in kw_proj.items():
                print(" ", k, ": ", v)

            peak_search = self.program.find_peaks(self.queue, self.wdim_data, self.workgroup_size["corrections4"], *list(kw_proj.values()))
            events.append(EventDescription("peak_search", peak_search))
            # call the find_peaks kernel

            # Return the number of peaks
            cnt = numpy.empty(1, dtype=numpy.int32)
            ev = pyopencl.enqueue_copy(self.queue, cnt, self.cl_mem["counter"])
            events.append(EventDescription("copy D->H counter", ev))
            ev.wait()
            high = numpy.empty(cnt, dtype=numpy.int32)
            ev = pyopencl.enqueue_copy(self.queue, high, self.cl_mem["peak_position"])
            events.append(EventDescription("copy D->H high_position", ev))
#             if out_merged is None:
#                 merged = numpy.empty((self.bins, 8), dtype=numpy.float32)
#             else:
#                 merged = out_merged.data
#             if out_avgint is None:
#                 avgint = numpy.empty(self.bins, dtype=numpy.float32)
#             else:
#                 avgint = out_avgint.data
#             if out_stderr is None:
#                 stderr = numpy.empty(self.bins, dtype=numpy.float32)
#             else:
#                 stderr = out_stderr.data
#
#             ev = pyopencl.enqueue_copy(self.queue, avgint, self.cl_mem["averint"])
#             events.append(EventDescription("copy D->H avgint", ev))
#
#             ev = pyopencl.enqueue_copy(self.queue, stderr, self.cl_mem["stderr"])
#             events.append(EventDescription("copy D->H stderr", ev))
#             ev = pyopencl.enqueue_copy(self.queue, merged, self.cl_mem["merged8"])
#             events.append(EventDescription("copy D->H merged8", ev))
        if self.profile:
            self.events += events
        # res = Integrate1dtpl(self.bin_centers, avgint, stderr, merged[:, 0], merged[:, 2], merged[:, 4], merged[:, 6])
        # "position intensity error signal variance normalization count"
        return high

    # Name of the default "process" method
    __call__ = peak_finder

class OCL_SimplePeakFinder(OpenclProcessing):
    BLOCK_SIZE = 1024 #works with 32x32 patches (1024 threads)

    kernel_files = ["pyfai:openCL/simple_peak_picker.cl"]
    buffers = [BufferDescription("image", 1, numpy.float32, mf.READ_WRITE),
               BufferDescription("image_raw", 1, numpy.float32, mf.READ_ONLY),
               BufferDescription("mask", 1, numpy.int8, mf.READ_ONLY),
               BufferDescription("output", 1, numpy.int32, mf.WRITE_ONLY)
               ]

    mapping = {numpy.int8: "s8_to_float",
               numpy.uint8: "u8_to_float",
               numpy.int16: "s16_to_float",
               numpy.uint16: "u16_to_float",
               numpy.uint32: "u32_to_float",
               numpy.int32: "s32_to_float",
               numpy.float32: "f32_to_float"
               }

    def __init__(self, image_shape=None, mask=None,
                 ctx=None, devicetype="all", platformid=None, deviceid=None,
                 block_size=None, profile=False):
        """
        :param image_shape: 2-tuple with the size of the image
        :param mask: array with invalid pixel flagged.
        :param ctx: actual working context, left to None for automatic
                    initialization from device type or platformid/deviceid
        :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
        :param platformid: integer with the platform_identifier, as given by clinfo
        :param deviceid: Integer with the device identifier, as given by clinfo
        :param block_size: preferred workgroup size, may vary depending on the outpcome of the compilation.
        :param profile: switch on profiling to be able to profile at the kernel level,
                        store profiling elements (makes code slightly slower)
        """
        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
                                  platformid=platformid, deviceid=deviceid,
                                  block_size=block_size, profile=profile)

        if mask is not None:
            if  image_shape:
                assert mask.shape == image_shape
            else:
                image_shape = mask.shape
            self.do_mask = True
        else:
            assert len(image_shape) == 2, "expect a 2-tuple with the size of the image"
            mask = numpy.zeros(image_shape, dtype=numpy.int8) 
            self.do_mask = False
        self.shape = image_shape
        self.on_device = {"mask": mask}

        if block_size is None:
            block_size = self.BLOCK_SIZE

        self.BLOCK_SIZE = min(block_size, self.device.max_work_group_size)
        self.workgroup_size = {}
        self.wg = self.size_to_doublet(self.BLOCK_SIZE)
        self.wdim = tuple((shape + BLOCK_SIZE - 1) & ~(BLOCK_SIZE - 1) for shape, BLOCK_SIZE in zip(self.shape[-1::-1], self.wg))

        self.buffers = [BufferDescription(i.name, i.size * numpy.prod(self.shape), i.dtype, i.flags)
                        for i in self.__class__.buffers]
        self.buffers.append(BufferDescription("count", 1, numpy.int32, mf.WRITE_ONLY))
        
        try:
            self.set_profiling(profile)
            self.allocate_buffers()
            self.compile_kernels()
            self.set_kernel_arguments()
        except (pyopencl.MemoryError, pyopencl.LogicError) as error:
            raise MemoryError(error)
        self.send_buffer(numpy.ascontiguousarray(mask, dtype=numpy.int8), "mask")

    def __copy__(self):
        """Shallow copy of the object

        :return: copy of the object
        """
        return self.__class__(self.shape,
                              mask=self.on_device.get("data"),
                              ctx=self.ctx,
                              block_size=self.block_size,
                              profile=self.profile)

    def __deepcopy__(self, memo=None):
        """deep copy of the object

        :return: deepcopy of the object
        """
        if memo is None:
            memo = {}
        mask = self.on_device.get("data")
        new_mask = mask.copy()
        memo[id(mask)] = new_mask
        new_obj = self.__class__(self.shape,
                                 mask = new_mask,
                                 ctx=self.ctx,
                                 block_size=self.block_size,
                                 profile=self.profile)
        memo[id(self)] = new_obj
        return new_obj

    @staticmethod
    def size_to_doublet(size):
        "Try to find the squarrest possible 2-tuple of this size"        
        small = 2**int(math.log(size ** 0.5, 2))
        large = size//small
        return (large,small)

    def compile_kernels(self, kernel_file=None):
        """
        Call the OpenCL compiler
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

        if default_compiler_options:
            compile_options = default_compiler_options
        else:
            compile_options = ""
        OpenclProcessing.compile_kernels(self, kernels, compile_options)
        for kernel_name, kernel in self.kernels.get_kernels().items():
            wg = kernel_workgroup_size(self.program, kernel)
            doublet1 = self.size_to_doublet(wg)
            self.workgroup_size[kernel_name] = tuple(min(a,b) for a,b in zip(doublet1, self.wg))

    def set_kernel_arguments(self):
        """Tie arguments of OpenCL kernel-functions to the actual kernels
        """
        cast_kernel = OrderedDict([("image_raw", self.cl_mem["image_raw"]),
                                   ("height", numpy.int32(self.shape[0])),
                                   ("width", numpy.int32(self.shape[1])),
                                   ("do_mask", numpy.int8(self.do_mask)),
                                   ("mask", self.cl_mem["mask"]),
                                   ("image", self.cl_mem["image"])])
                                   
        self.cl_kernel_args["u8_to_float"] = cast_kernel
        self.cl_kernel_args["s8_to_float"] = cast_kernel
        self.cl_kernel_args["u16_to_float"] = cast_kernel
        self.cl_kernel_args["s16_to_float"] = cast_kernel
        self.cl_kernel_args["u32_to_float"] = cast_kernel
        self.cl_kernel_args["s32_to_float"] = cast_kernel
        self.cl_kernel_args["f32_to_float"] = cast_kernel
        self.cl_kernel_args["memset_int"] = OrderedDict([("count", self.cl_mem["count"]),
                                                        ("pattern", numpy.int32(0)),
                                                        ("size", numpy.int32(1))])
        self.cl_kernel_args["simple_spot_finder"] = OrderedDict([("image", self.cl_mem["image"]),
                                                                 ("height", numpy.int32(self.shape[0])),
                                                                 ("width", numpy.int32(self.shape[1])),
                                                                 ("half_wind_height", numpy.int32(3)),
                                                                 ("half_wind_width", numpy.int32(3)),
                                                                 ("threshold", numpy.float32(3.0)),
                                                                 ("radius", numpy.float32(1.0)),
                                                                 ("noise", numpy.float32(1.0)),
                                                                 ("count", self.cl_mem["count"]),
                                                                 ("output", self.cl_mem["output"]),
                                                                 ("output_size", numpy.int32(numpy.prod(self.shape))),
                                                                 ("local_high", pyopencl.LocalMemory(self.BLOCK_SIZE*4)),
                                                                 ("local_size", numpy.int32(self.BLOCK_SIZE))])

    def send_buffer(self, data, dest, checksum=None, force_cast=False):
        """Send a numpy array to the device, including the cast on the device if possible

        :param data: numpy array with data
        :param dest: name of the buffer as registered in the class
        """

        dest_type = numpy.dtype([i.dtype for i in self.buffers if i.name == dest][0])
        events = []
        if isinstance(data, pyopencl.array.Array):
            if (data.dtype == dest_type) and not force_cast:
                copy_image = pyopencl.enqueue_copy(self.queue, self.cl_mem[dest], data.data)
                events.append(EventDescription("copy D->D %s" % dest, copy_image))
            else:
                copy_image = pyopencl.enqueue_copy(self.queue, self.cl_mem["image_raw"], data.data)
                kernel_name = self.mapping[data.dtype.type]
                kernel = self.kernels.get_kernel(kernel_name)
                kw = self.cl_kernel_args[kernel_name].copy()
                kw["image"] = self.cl_mem[dest]
                cast_to_float = kernel(self.queue, self.wdim, self.workgroup_size[kernel_name], *list(kw.values()))
                events += [EventDescription("copy raw D->D " + dest, copy_image),
                           EventDescription("cast " + kernel_name, cast_to_float)]
        else:
            # Assume it is a numpy array
            if ((data.dtype == dest_type) or (data.dtype.itemsize > dest_type.itemsize)) and not force_cast:
                copy_image = pyopencl.enqueue_copy(self.queue, self.cl_mem[dest], numpy.ascontiguousarray(data, dest_type))
                events.append(EventDescription("copy H->D %s" % dest, copy_image))
            else:
                copy_image = pyopencl.enqueue_copy(self.queue, self.cl_mem["image_raw"], numpy.ascontiguousarray(data))
                kernel_name = self.mapping[data.dtype.type]
                kernel = self.kernels.get_kernel(kernel_name)
                kw = self.cl_kernel_args[kernel_name].copy()
                kw["image"] = self.cl_mem[dest]
                cast_to_float = kernel(self.queue, self.wdim, self.workgroup_size[kernel_name], *list(kw.values()))
                events += [EventDescription("copy raw H->D " + dest, copy_image),
                           EventDescription("cast " + kernel_name, cast_to_float)]
        if self.profile:
            self.events += events
        if checksum is not None:
            self.on_device[dest] = checksum

    def search(self, 
               image,
               window=7,
               threshold=3.0,
               radius=1.0,
               noise=1.0
               ):
        """
        Search for peaks in this image and return a list of them
        
        :param image: 2d array with an image
        :param window: size of the window, i.e. 7 for 7x7 patch size.
        :param threshhold: keep peaks with I > mean + threshold*std
        :param radius: keep points with centroid on center within this radius (in pixel)
        :param noise: minimum signal for peak to discard noisy region.
        :return: array of peak coordinates.  
        """
        events = []
        assert image.shape == self.shape
        with self.sem:
            self.send_buffer(image, "image", force_cast=True)
            self.kernels.memset_int(self.queue, (1,), (1,), *list(self.cl_kernel_args["memset_int"].values()))
            
            kw = self.cl_kernel_args["simple_spot_finder"]
            kw["half_wind_height"]=kw["half_wind_width"]=numpy.int32(window//2)
            kw["threshold"] = numpy.float32(threshold)
            kw["radius"] = numpy.float32(radius)
            kw["noise"] = numpy.float32(noise)
            wg = self.workgroup_size["simple_spot_finder"]
            ev = self.kernels.simple_spot_finder(self.queue, self.wdim, wg, *list(kw.values()))
            events.append(EventDescription("simple_spot_finder", ev))
            count = numpy.empty(1, dtype=numpy.int32)
            copy_count = pyopencl.enqueue_copy(self.queue, count, self.cl_mem["count"])
            events.append(EventDescription("copy_count", copy_count))
            indexes = numpy.empty(count[0], dtype=numpy.int32)
            copy_index = pyopencl.enqueue_copy(self.queue, indexes, self.cl_mem["output"])
            events.append(EventDescription("copy_index", copy_index))
        if self.profile:
            self.events += events
        peaks = numpy.empty(count, dtype=numpy.dtype([("x", numpy.int32),("y", numpy.int32)]))
        peaks["x"] = indexes % self.shape[1]
        peaks["y"] = indexes // self.shape[1]
        return peaks
    
    __call__ = search
        
