# -*- coding: utf-8 -*-
#
#    Project: Peak finder in a single 2D diffraction frame
#             https://github.com/silx-kit/pyFAI
#
#
#    Copyright (C) 2014-2025 European Synchrotron Radiation Facility, Grenoble, France
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
__date__ = "07/10/2025"
__copyright__ = "2014-2023, ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import logging
from collections import OrderedDict
import math
import numpy
from ..containers import SparseFrame, ErrorModel
from ..utils.mathutil import EPS32
from .azim_csr import OCL_CSR_Integrator, BufferDescription, EventDescription, mf, calc_checksum, pyopencl, OpenclProcessing
from . import kernel_workgroup_size, dtype_converter

logger = logging.getLogger(__name__)


class OCL_PeakFinder(OCL_CSR_Integrator):
    BLOCK_SIZE = 1024  # unlike in OCL_CSR_Integrator, here we need larger blocks
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
               BufferDescription("position", 1, numpy.int32, mf.READ_WRITE),
               BufferDescription("descriptor", 4, numpy.float32, mf.WRITE_ONLY),
               BufferDescription("radius2d", 1, numpy.float32, mf.READ_ONLY),
               ]
    kernel_files = ["silx:opencl/doubleword.cl",
                    "pyfai:openCL/preprocess.cl",
                    "pyfai:openCL/memset.cl",
                    "pyfai:openCL/ocl_azim_CSR.cl",
                    "pyfai:openCL/sparsify.cl",
                    "pyfai:openCL/peakfinder.cl",
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
        if image_size != radius.size:
            raise RuntimeError("Image size mismatch")
        nbin = lut[2].size - 1
        extra_buffers = [
                         BufferDescription("radius1d", nbin, numpy.float32, mf.READ_ONLY),
                         BufferDescription("counter", 1, numpy.int32, mf.READ_WRITE),
                         ]

        OCL_CSR_Integrator.__init__(self, lut=lut, image_size=image_size, checksum=checksum,
                 empty=empty, unit=unit, bin_centers=bin_centers,
                 ctx=ctx, devicetype=devicetype, platformid=platformid, deviceid=deviceid,
                 block_size=block_size,
                 profile=profile, extra_buffers=extra_buffers)

        if mask is None:
            self.cl_kernel_args["corrections4a"]["do_mask"] = numpy.int8(0)
            self.mask = None
        else:
            self.mask = numpy.ascontiguousarray(mask, numpy.int8)
            self.send_buffer(self.mask, "mask")
            self.cl_kernel_args["corrections4a"]["do_mask"] = numpy.int8(1)

        if self.bin_centers is None:
            raise RuntimeError("1D bin center position is mandatory")
        else:
            self.send_buffer(self.bin_centers, "radius1d")

        if radius is None:
            raise RuntimeError("2D radius position is mandatory")
        else:
            self.radius2d = numpy.array(radius, dtype=numpy.float32)  # this makes explicitely a copy
            if self.mask is not None:
                msk = numpy.where(self.mask)
                self.radius2d[msk] = numpy.nan
            self.send_buffer(self.radius2d, "radius2d")

    def guess_workgroup_size(self, block_size=None):
        """Determines the optimal workgroup size.

        For peak finding, the larger, the better.
        this is limited by the amount of shared memory available on the device.

        :param block_size: Input workgroup size (block is the cuda name)
        :return: the optimal workgoup size
        """
        if block_size is None:
            # one float8, i.e. 32 bytes per thread of storage is needed
            device = self.ctx.devices[0]
            # platform = device.platform.name.lower()
            block_size = 1 << int(math.floor(math.log((device.local_mem_size - 40) / 32.0, 2.0)))
            self.force_workgroup_size = False
        else:
            self.force_workgroup_size = True
            block_size = int(block_size)
        return block_size

    def set_kernel_arguments(self):
        OCL_CSR_Integrator.set_kernel_arguments(self)
        self.cl_kernel_args["find_intense"] = OrderedDict((("output4", self.cl_mem["output4"]),
                                                          ("radius2d", self.cl_mem["radius2d"]),
                                                          ("radius1d", self.cl_mem["radius1d"]),
                                                          ("averint", self.cl_mem["averint"]),
                                                          ("std", self.cl_mem["std"]),
                                                          ("radius_min", numpy.float32(0.0)),
                                                          ("radius_max", numpy.float32(numpy.finfo(numpy.float32).max)),
                                                          ("cutoff", numpy.float32(5.0)),
                                                          ("noise", numpy.float32(1.0)),
                                                          ("counter", self.cl_mem["counter"]),
                                                          ("position", self.cl_mem["position"]),
                                                          ('shared', None)))
        self.cl_kernel_args["copy_intense"] = OrderedDict((("position", self.cl_mem["position"]),
                                                           ("counter", 0),
                                                           ("output4", self.cl_mem["output4"]),
                                                           ("descriptor", self.cl_mem["descriptor"])))
        self.cl_kernel_args["peakfinder"] = OrderedDict((("output4", self.cl_mem["output4"]),
                                                         ("radius2d", self.cl_mem["radius2d"]),
                                                         ("radius1d", self.cl_mem["radius1d"]),
                                                         ("averint", self.cl_mem["averint"]),
                                                         ("std", self.cl_mem["std"]),
                                                         ("radius_min", numpy.float32(0.0)),
                                                         ("radius_max", numpy.float32(numpy.finfo(numpy.float32).max)),
                                                         ("cutoff", numpy.float32(5.0)),
                                                         ("noise", numpy.float32(1.0)),
                                                         ("height", numpy.int32(2)),
                                                         ("width", numpy.int32(2)),
                                                         ("half_patch", numpy.int32(1)),
                                                         ("connected", numpy.int32(2)),
                                                         # ("max_pkg", numpy.int32(2)),
                                                         # ("max_pkl", numpy.int32(2)),
                                                         ("counter", self.cl_mem["counter"]),
                                                         ("position", self.cl_mem["position"]),
                                                         ("descriptor", self.cl_mem["descriptor"]),
                                                         ("local_highidx", None),
                                                         ("local_peaks", None),
                                                         ("local_buffer", None)))

    def _sigma_clip(self, data, dark=None, dummy=None, delta_dummy=None,
               variance=None, dark_variance=None,
               flat=None, solidangle=None, polarization=None, absorption=None,
               dark_checksum=None, flat_checksum=None, solidangle_checksum=None,
               polarization_checksum=None, absorption_checksum=None, dark_variance_checksum=None,
               safe=True, error_model=ErrorModel.NO,
               normalization_factor=1.0,
               cutoff_clip=5.0, cycle=5):
        """Performs the sigma-clipping which is common to all processing
        Note: this function does not lock the OpenCL context!

        :param data: 2D array with the signal
        :param dark: array of same shape as data for pre-processing
        :param dummy: value for invalid data
        :param delta_dummy: precision for dummy assessment
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
        :param error_model: ErrorModel enum "poisson" for poissonian model V=I, "azimuthal" or "hybrid", i.e azimuthal for clipping and Poisson for picking
        :param normalization_factor: divide raw signal by this value
        :param cutoff_clip: discard all points with `|value - avg| > cutoff * sigma` during sigma_clipping. 4-5 is quite common
        :param cycle: perform at maximum this number of cycles. 5 is common.
        :return: list of event to wait for.
        """
        events = []
        # convert = (data.dtype.itemsize>4)
        # self.send_buffer(data, "image", convert=convert)
        kernel_correction_name = "corrections4a"
        corrections4 = self.kernels.corrections4a
        kw_corr = self.cl_kernel_args[kernel_correction_name]
        kw_corr["image"] = self.send_buffer(data, "image", convert=False)
        kw_corr["dtype"] = numpy.int8(32) if data.dtype.itemsize > 4 else dtype_converter(data.dtype)

        wg = max(self.workgroup_size["memset_ng"])
        wdim_bins = int(self.bins + wg - 1) // wg * wg,
        memset = self.kernels.memset_out(self.queue, wdim_bins, (wg,), *list(self.cl_kernel_args["memset_ng"].values()))
        events.append(EventDescription("memset_ng", memset))

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

        kw_int = self.cl_kernel_args["csr_sigma_clip4"]
        kw_corr["error_model"] = kw_int["error_model"] = numpy.int8(error_model.value)

        wg = max(self.workgroup_size[kernel_correction_name])
        wdim_data = int(self.size + wg - 1) // wg * wg,
        ev = corrections4(self.queue, wdim_data, (wg,), *list(kw_corr.values()))
        events.append(EventDescription(kernel_correction_name, ev))

        # Prepare sigma-clipping
        kw_int["cutoff"] = numpy.float32(cutoff_clip)
        kw_int["cycle"] = numpy.int32(cycle)

        wg_min = min(self.workgroup_size["csr_sigma_clip4"])
        wdim_bins = (self.bins * wg_min),
        kw_int["shared"] = pyopencl.LocalMemory(32 * wg_min)
        integrate = self.kernels.csr_sigma_clip4(self.queue, wdim_bins, (wg_min,), *kw_int.values())
        events.append(EventDescription("csr_sigma_clip4", integrate))
        return events

    def _count_intense(self, noise=1.0, cutoff_pick=3.0, radial_range=None, events=None):
        """
        Count the number of high-pixel by:
        * reconstruct the background in 2D
        * count the number of pixel above max(noise, mean + cutoff*sigma)

        Note:
        *this function does not lock the OpenCL context!
        *sigma clipping has to be performed previously

        :param noise: minimum meaningful signal. Fixed threshold for picking
        :param cutoff_pick: pick points with `value > background + cutoff * sigma` 3-4 is quite common value
        :param radial_range: 2-tuple with the minimum and maximum radius values for picking points. Reduces the region of search.
        :param events: list of OpenCL events for timing
        :return: number of pixel of high intensity found
        """
        if events is None:
            events = []
        # now perform the calc_from_1d on the device and count the number of pixels
        memset = self.program.memset_int(self.queue, (1,), (1,), self.cl_mem["counter"], numpy.int32(0), numpy.int32(1))
        events.append(EventDescription("memset counter", memset))

        # Prepare peak-picking
        kw_proj = self.cl_kernel_args["find_intense"]
        kw_proj["cutoff"] = numpy.float32(cutoff_pick)
        kw_proj["noise"] = numpy.float32(noise)
        if radial_range is not None:
            kw_proj["radius_min"] = numpy.float32(min(radial_range))
            kw_proj["radius_max"] = numpy.float32(max(radial_range) * EPS32)
        else:
            kw_proj["radius_min"] = numpy.float32(0.0)
            kw_proj["radius_max"] = numpy.float32(numpy.finfo(numpy.float32).max)

        wg = max(self.workgroup_size["find_intense"])
        wdim_data = int(self.size + wg - 1) // wg * wg,
        kw_proj["shared"] = pyopencl.LocalMemory(wg * 4)  # stores int
        peak_search = self.program.find_intense(self.queue, wdim_data, (wg,), *list(kw_proj.values()))
        events.append(EventDescription("find_intense", peak_search))

        # Return the number of peaks
        cnt = numpy.empty(1, dtype=numpy.int32)
        ev = pyopencl.enqueue_copy(self.queue, cnt, self.cl_mem["counter"])
        events.append(EventDescription("copy D->H counter", ev))
        self.profile_multi(events)
        return cnt[0]

    def _count_peak(self, data, dark=None, dummy=None, delta_dummy=None,
                    variance=None, dark_variance=None,
                    flat=None, solidangle=None, polarization=None, absorption=None,
                    dark_checksum=None, flat_checksum=None, solidangle_checksum=None,
                    polarization_checksum=None, absorption_checksum=None, dark_variance_checksum=None,
                    safe=True, error_model=ErrorModel.NO,
                    normalization_factor=1.0,
                    cutoff_clip=5.0, cycle=5, noise=1.0, cutoff_peak=3.0,
                    radial_range=None, patch_size=3, connected=3):
        """
        Count the number of peaks by:
        * sigma_clipping within a radial bin to measure the mean and the deviation of the background
        * reconstruct the background in 2D
        * count the number of peaks above mean + cutoff*sigma

        Note: this function does not lock the OpenCL context!

        :param data: 2D array with the signal
        :param dark: array of same shape as data for pre-processing
        :param dummy: value for invalid data
        :param delta_dummy: precision for dummy assessment
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
        :param error_model: ErrorModel enum can be "poisson" for poissonian model V=I, "azimuthal" or "hybrid", i.e azimuthal for clipping and Poisson for picking
        :param normalization_factor: divide raw signal by this value
        :param cutoff_clip: discard all points with `|value - avg| > cutoff * sigma` during sigma_clipping. 4-5 is quite common
        :param cycle: perform at maximum this number of cycles. 5 is common.
        :param noise: minimum meaningful signal. Fixed threshold for picking
        :param cutoff_peak: pick points with `value > background + cutoff * sigma` 3-4 is quite common value
        :param radial_range: 2-tuple with the minimum and maximum radius values for picking points. Reduces the region of search.
        :param patch_size: defines the size of the vinicy to explore 3x3 or 5x5
        :param connected: number of pixels above threshold in local patch
        :return: number of pixel of high intensity found
        """
        events = self._sigma_clip(data, dark, dummy, delta_dummy,
               variance, dark_variance,
               flat, solidangle, polarization, absorption,
               dark_checksum, flat_checksum, solidangle_checksum,
               polarization_checksum, absorption_checksum, dark_variance_checksum,
               safe, error_model,
               normalization_factor,
               cutoff_clip, cycle)
        return self._peak_picking(data, noise, cutoff_peak, radial_range, patch_size, connected,
                                  events=events)

    def _peak_picking(self, data, noise, cutoff_peak, radial_range=None, patch_size=3, connected=3,
                      events=None):
        """This calls only the peak-picking kernel, unlocked

        :param data: input 2d image
        :param noise: minimum meaningful signal. Fixed threshold for picking
        :param cutoff_peak: pick points with `value > background + cutoff * sigma` 3-4 is quite common value
        :param radial_range: 2-tuple with the minimum and maximum radius values for picking points. Reduces the region of search.
        :param patch_size: defines the size of the vinicy to explore 3x3 or 5x5
        :param connected: number of pixels above threshold in local patch
        :return: number of pixel of high intensity found
        """
        if events is None:
            events = []
        # reset the count of the number of pixels
        memset = self.program.memset_int(self.queue, (1,), (1,), self.cl_mem["counter"], numpy.int32(0), numpy.int32(1))
        events.append(EventDescription("memset counter", memset))
        # Prepare for picking peaks
        kw_proj = self.cl_kernel_args["peakfinder"]
        kw_proj["height"] = numpy.int32(data.shape[0])
        kw_proj["width"] = numpy.int32(data.shape[1])
        kw_proj["connected"] = numpy.int32(connected)
        kw_proj["cutoff"] = numpy.float32(cutoff_peak)
        kw_proj["noise"] = numpy.float32(noise)
        if radial_range is not None:
            kw_proj["radius_min"] = numpy.float32(min(radial_range))
            kw_proj["radius_max"] = numpy.float32(max(radial_range) * EPS32)
        else:
            kw_proj["radius_min"] = numpy.float32(0.0)
            kw_proj["radius_max"] = numpy.float32(numpy.finfo(numpy.float32).max)

        wg = max(self.workgroup_size["peakfinder"])
        if wg > 32:
            wg1 = 32
            wg0 = wg // 32
        else:
            swg = math.sqrt(wg)
            iswg = int(swg)
            if iswg == swg:
                wg0 = wg1 = iswg
            else:
                sqrt2 = math.sqrt(2.0)
                wg1 = int(swg * sqrt2)
                wg0 = wg // wg1

        wdim_data = int(data.shape[0] + wg0 - 1) // wg0 * wg0, int(data.shape[1] + wg1 - 1) // wg1 * wg
        # allocate local memory: we store 4 bytes but at most 1 pixel out of 4 can be a peak

        hw = patch_size // 2  # Half width of the patch
        buffer_size = int(math.ceil(wg * 4 / ((1 + hw) * min(wg0, 1 + hw))))
        kw_proj["local_highidx"] = pyopencl.LocalMemory(1 * buffer_size)
        kw_proj["local_peaks"] = pyopencl.LocalMemory(4 * buffer_size)
        kw_proj["local_buffer"] = pyopencl.LocalMemory(8 * (wg0 + 2 * hw) * (wg1 + 2 * hw))
        kw_proj["half_patch"] = numpy.int32(hw)
        peak_search = self.program.peakfinder(self.queue, wdim_data, (wg0, wg1), *list(kw_proj.values()))

        events.append(EventDescription("peakfinder", peak_search))

        # Return the number of peaks
        cnt = numpy.empty(1, dtype=numpy.int32)
        ev = pyopencl.enqueue_copy(self.queue, cnt, self.cl_mem["counter"])
        events.append(EventDescription("copy D->H counter", ev))
        self.profile_multi(events)
        return cnt[0]

#==================================================
#                Start of the public API:
#==================================================

    def count_intense(self, data, dark=None, dummy=None, delta_dummy=None,
               variance=None, dark_variance=None,
               flat=None, solidangle=None, polarization=None, absorption=None,
               dark_checksum=None, flat_checksum=None, solidangle_checksum=None,
               polarization_checksum=None, absorption_checksum=None, dark_variance_checksum=None,
               safe=True, error_model=None,
               normalization_factor=1.0,
               cutoff_clip=5.0, cycle=5, noise=1.0, cutoff_pick=3.0,
               radial_range=None):
        """
        Count the number of intense by:
        * sigma_clipping within a radial bin to measure the mean and the deviation of the background
        * reconstruct the background in 2D
        * count the number of pixels above mean + cutoff*sigma

        :param data: 2D array with the signal
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
        :param error_model: ErrorModel enum "poisson" for poissonian model V=I, "azimuthal" or "hybrid", i.e azimuthal for clipping and Poisson for picking
        :param normalization_factor: divide raw signal by this value
        :param cutoff_clip: discard all points with `|value - avg| > cutoff * sigma` during sigma_clipping.
                   Values of 4-5 are quite common.
                   The minimum value is obtained from Chauvenet criterion: sqrt(2ln(n/sqrt(2pi)))
                   where n is the number of pixel in the bin, usally around 2 to 3.
        :param cycle: perform at maximum this number of cycles. 5 is common.
        :param noise: minimum meaningful signal. Fixed threshold for picking
        :param cutoff_pick: pick points with `value > background + cutoff * sigma` 3-4 is quite common value
        :param radial_range: 2-tuple with the minimum and maximum radius values for picking points. Reduces the region of search.
        :return: number of pixel of high intensity found
        """
        error_model = ErrorModel.parse(error_model)
        if error_model == ErrorModel.NO:
            if variance is None:
                logger.warning("Nor variance nor error-model is provided ... expect garbage-out")
            else:
                error_model = ErrorModel.VARIANCE
        with self.sem:
            events = self._sigma_clip(data, dark, dummy, delta_dummy, variance, dark_variance, flat, solidangle, polarization, absorption,
                             dark_checksum, flat_checksum, solidangle_checksum, polarization_checksum, absorption_checksum, dark_variance_checksum,
                             safe, error_model, normalization_factor, cutoff_clip, cycle)
            count = self._count_intense(noise, cutoff_pick, radial_range, events=events)
        return count

    count = count_intense

    def sparsify(self, data, dark=None, dummy=None, delta_dummy=None,
                 variance=None, dark_variance=None,
                 flat=None, solidangle=None, polarization=None, absorption=None,
                 dark_checksum=None, flat_checksum=None, solidangle_checksum=None,
                 polarization_checksum=None, absorption_checksum=None, dark_variance_checksum=None,
                 safe=True, error_model=None,
                 normalization_factor=1.0,
                 cutoff_clip=5.0, cycle=5, noise=1.0,
                 cutoff_pick=3.0, cutoff_peak=None,
                 radial_range=None, patch_size=3, connected=2):
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
        :param error_model: ErrorModel enum "poisson" for poissonian model V=I, "azimuthal" or "hybrid", i.e azimuthal for clipping and Poisson for picking
        :param normalization_factor: divide raw signal by this value
        :param cutoff_clip: discard all points with `|value - avg| > cutoff * sigma` during sigma_clipping. 4-5 is quite common
        :param cycle: perform at maximum this number of cycles. 5 is common.
        :param noise: minimum meaningful signal. Fixed threshold for picking
        :param cutoff_pick: pick points with `value > background + cutoff * sigma` 3-4 is quite common value
        :param cutoff_peak: cut-off to consider a pixel as part of a peak (activate peak-picking)
        :param radial_range: 2-tuple with the minimum and maximum radius values for picking points. Reduces the region of search.
        :param patch_size: defines the size of the vinicy to explore 3x3 or 5x5 when peak-picking. Depends on `cutoff_peak`
        :param connected: number of pixels above threshold in local patch ot be considered as a peak. Depends on `cutoff_peak`

        :return: SparseFrame object, see `intensity`, `x` and `y` properties

        """
        error_model = ErrorModel.parse(error_model)
        if error_model == ErrorModel.NO:
            if variance is None:
                logger.warning("Nor variance nor error-model is provided ... expect garbage-out")
            else:
                error_model = ErrorModel.VARIANCE

        with self.sem:
            events = self._sigma_clip(data, dark, dummy, delta_dummy, variance, dark_variance, flat, solidangle, polarization, absorption,
                             dark_checksum, flat_checksum, solidangle_checksum, polarization_checksum, absorption_checksum, dark_variance_checksum,
                             safe, error_model, normalization_factor, cutoff_clip, cycle)

            background_avg = numpy.empty(self.bins, dtype=numpy.float32)
            background_std = numpy.empty(self.bins, dtype=numpy.float32)
            ev1 = pyopencl.enqueue_copy(self.queue, background_avg, self.cl_mem["averint"])
            ev2 = pyopencl.enqueue_copy(self.queue, background_std, self.cl_mem["std"])
            events += [EventDescription("copy D->H background_avg", ev1),
                           EventDescription("copy D->H background_std", ev2)]

            # Perform first peak-picking before sparsification because it does not mangle the preproc4 array
            if cutoff_peak:
                cnt_peaks = self._peak_picking(data, noise, cutoff_peak, radial_range, patch_size, connected)
                index = numpy.empty(cnt_peaks, dtype=numpy.int32)
                peak4 = numpy.empty((cnt_peaks, 4), dtype=numpy.float32)
                peaks = numpy.empty(cnt_peaks, dtype=[("index", numpy.int32), ("intensity", numpy.float32), ("sigma", numpy.float32),
                                                   ("pos0", numpy.float32), ("pos1", numpy.float32)])
                if cnt_peaks:
                    ev_idx = pyopencl.enqueue_copy(self.queue, index, self.cl_mem["position"])
                    ev_pks = pyopencl.enqueue_copy(self.queue, peak4, self.cl_mem["descriptor"])
                    events += [ EventDescription("copy D->H peak positions", ev_idx),
                                EventDescription("copy D->H peak descriptor", ev_pks)]
                peaks["index"] = index
                peaks["pos0"], peaks["pos1"], peaks["intensity"], peaks["sigma"] = peak4.T

            # Perform sparsification after peak-finding since it mangles the "preproc4" array
            cnt_intense = self._count_intense(noise, cutoff_pick, radial_range)

            indexes = numpy.empty(cnt_intense, dtype=numpy.int32)
            dtype = data.dtype
            if dtype.kind == 'f':
                dtype = numpy.float32
                kernel = self.program.copy_intense
            elif dtype.kind in "iu":
                if dtype.itemsize > 4:
                    dtype = numpy.dtype("uint32") if dtype.kind == "u" else numpy.dtype("int32")
                kernel = self.program.__getattr__("copy_intense_" + dtype.name)
            signal = numpy.empty(cnt_intense, dtype)
            if cnt_intense > 0:
                # Call kernel to copy intensities
                kw = self.cl_kernel_args["copy_intense"]
                kw["counter"] = cnt_intense
                size = int(cnt_intense + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE * self.BLOCK_SIZE
                ev0 = kernel(self.queue, (size,), (self.BLOCK_SIZE,),
                                         *list(kw.values()))
                ev1 = pyopencl.enqueue_copy(self.queue, indexes, self.cl_mem["position"])
                ev2 = pyopencl.enqueue_copy(self.queue, signal, self.cl_mem["descriptor"])

                events += [EventDescription(f"copy D->D + cast {numpy.dtype(dtype).name} intensity", ev0),
                           EventDescription("copy D->H intense pixels position", ev1),
                           EventDescription("copy D->H intense pixels intensity", ev2)]

        result = SparseFrame(indexes, signal)
        result._shape = data.shape
        result._dtype = data.dtype
        result._compute_engine = self.__class__.__name__
        result._mask = self.radius2d
        result._cutoff_clip = cutoff_clip
        result._cutoff_pick = cutoff_pick
        result._noise = noise
        result._radius = self.bin_centers
        result._background_avg = background_avg
        result._background_std = background_std
        result._unit = self.unit
        result._has_dark_correction = dark is not None
        result._has_flat_correction = flat is not None
        result._normalization_factor = normalization_factor
        result._has_polarization_correction = polarization is not None
        result._has_solidangle_correction = solidangle is not None
        result._has_absorption_correction = absorption is not None
        result._metadata = None
        result._method = "sparsify"
        result._method_called = None
        result._background_cycle = cycle
        result._radial_range = radial_range
        result._dummy = dummy
        result._error_model = error_model
        if cutoff_peak:
            result._peaks = peaks
            result._cutoff_peak = cutoff_peak
            result._peak_patch_size = patch_size
            result._peak_connected = connected
        self.profile_multi(events)
        return result

    def peakfinder(self, data, dark=None, dummy=None, delta_dummy=None,
                   variance=None, dark_variance=None,
                   flat=None, solidangle=None, polarization=None, absorption=None,
                   dark_checksum=None, flat_checksum=None, solidangle_checksum=None,
                   polarization_checksum=None, absorption_checksum=None, dark_variance_checksum=None,
                   safe=True, error_model=None,
                   normalization_factor=1.0,
                   cutoff_clip=5.0, cycle=5, noise=1.0, cutoff_peak=3.0,
                   radial_range=None, patch_size=3, connected=2):
        """
        Count the number of peaks by:
        * sigma_clipping within a radial bin to measure the mean and the deviation of the background
        * reconstruct the background in 2D
        * count the number of peaks above mean + cutoff*sigma

        :param data: 2D array with the signal
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
        :param error_model: ErrorModel enum "poisson" for poissonian model V=I, "azimuthal" or "hybrid", i.e azimuthal for clipping and Poisson for picking
        :param normalization_factor: divide raw signal by this value
        :param cutoff_clip: discard all points with `|value - avg| > cutoff * sigma` during sigma_clipping.
                   Values of 4-5 are quite common.
                   The minimum value is obtained from Chauvenet criterion: sqrt(2ln(n/sqrt(2pi)))
                   where n is the number of pixel in the bin, usally around 2 to 3.
        :param cycle: perform at maximum this number of cycles. 5 is common.
        :param noise: minimum meaningful signal. Fixed threshold for picking
        :param cutoff_peak: pick points with `value > background + cutoff * sigma` 3-4 is quite common value
        :param radial_range: 2-tuple with the minimum and maximum radius values for picking points. Reduces the region of search.
        :param patch_size: defines the size of the vinicy to explore 3x3 or 5x5
        :param connected: number of pixels above threshold in 3x3 region
        :return: number of pixel of high intensity found
        """
        error_model = ErrorModel.parse(error_model)
        if error_model == ErrorModel.NO:
            if variance is None:
                logger.warning("Nor variance nor error-model is provided ... expect garbage-out")
            else:
                error_model = ErrorModel.VARIANCE
        with self.sem:
            count = self._count_peak(data, dark, dummy, delta_dummy, variance, dark_variance, flat, solidangle, polarization, absorption,
                                     dark_checksum, flat_checksum, solidangle_checksum, polarization_checksum, absorption_checksum, dark_variance_checksum,
                                     safe, error_model, normalization_factor, cutoff_clip, cycle, noise, cutoff_peak, radial_range, patch_size, connected)
            index = numpy.empty(count, dtype=numpy.int32)
            peaks = numpy.empty((count, 4), dtype=numpy.float32)
            if count:
                idxp = pyopencl.enqueue_copy(self.queue, index, self.cl_mem["position"])
                idxd = pyopencl.enqueue_copy(self.queue, peaks, self.cl_mem["descriptor"])
                self.profile_multi([ EventDescription("copy D->H peak positions", idxp),
                                     EventDescription("copy D->H peak descriptor", idxd)])
        output = numpy.empty(count, dtype=[("index", numpy.int32), ("intensity", numpy.float32), ("sigma", numpy.float32),
                                           ("pos0", numpy.float32), ("pos1", numpy.float32)])
        output["index"] = index
        output["pos0"], output["pos1"], output["intensity"], output["sigma"] = peaks.T
        return output

    peakfinder8 = peakfinder
    # Name of the default "process" method
    __call__ = sparsify

#===============================================================================
# Simple variante
#===============================================================================


class OCL_SimplePeakFinder(OpenclProcessing):
    BLOCK_SIZE = 1024  # works with 32x32 patches (1024 threads)

    kernel_files = ["pyfai:openCL/simple_peak_picker.cl"]
    buffers = [BufferDescription("image", 1, numpy.float32, mf.READ_WRITE),
               BufferDescription("image_raw", 1, numpy.float32, mf.READ_ONLY),
               BufferDescription("mask", 1, numpy.int8, mf.READ_ONLY),
               BufferDescription("output", 1, numpy.int32, mf.READ_WRITE),
               BufferDescription("peak_intensity", 1, numpy.float32, mf.WRITE_ONLY)
               ]

    mapping = {numpy.int8: "s8_to_float",
               numpy.uint8: "u8_to_float",
               numpy.int16: "s16_to_float",
               numpy.uint16: "u16_to_float",
               numpy.uint32: "u32_to_float",
               numpy.uintc: "u32_to_float",
               numpy.int32: "s32_to_float",
               numpy.intc: "s32_to_float",
               numpy.float32: "f32_to_float",
               numpy.single: "f32_to_float",
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
                if mask.shape != image_shape:
                    raise RuntimeError("Mask size mismatch")
            else:
                image_shape = mask.shape
            self.do_mask = True
        else:
            if len(image_shape) != 2:
                raise RuntimeError("expect a 2-tuple with the size of the image")
            mask = numpy.zeros(image_shape, dtype=numpy.int8)
            self.do_mask = False
        self.shape = image_shape
        self.on_device = {"mask": mask}

        if block_size is None:
            block_size = self.BLOCK_SIZE

        self.BLOCK_SIZE = min(block_size, self.device.max_work_group_size)
        self.workgroup_size = {}
        self.wg = self.size_to_doublet(self.BLOCK_SIZE)
        if sum(i < j for i, j in zip(self.ctx.devices[0].max_work_item_sizes, self.wg)):
            self.wg = self.ctx.devices[0].max_work_item_sizes[:2]
        self.wdim = tuple(int(shape + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE for shape, BLOCK_SIZE in zip(self.shape[-1::-1], self.wg))

        self.buffers = [BufferDescription(i.name, i.size * numpy.prod(self.shape), i.dtype, i.flags)
                        for i in self.__class__.buffers]
        self.buffers.append(BufferDescription("count", 1, numpy.int32, mf.READ_WRITE))

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
                                 mask=new_mask,
                                 ctx=self.ctx,
                                 block_size=self.block_size,
                                 profile=self.profile)
        memo[id(self)] = new_obj
        return new_obj

    @staticmethod
    def size_to_doublet(size):
        "Try to find the squarrest possible 2-tuple of this size"
        small = 2 ** int(math.log(size ** 0.5, 2))
        large = size // small
        return (large, small)

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

        OpenclProcessing.compile_kernels(self, kernels, compile_options)
        for kernel_name, kernel in self.kernels.get_kernels().items():
            wg = kernel_workgroup_size(self.program, kernel)
            doublet1 = self.size_to_doublet(wg)
            self.workgroup_size[kernel_name] = tuple(min(a, b) for a, b in zip(doublet1, self.wg))

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
                                                                 ("cutoff", numpy.float32(3.0)),
                                                                 ("radius", numpy.float32(1.0)),
                                                                 ("noise", numpy.float32(1.0)),
                                                                 ("count", self.cl_mem["count"]),
                                                                 ("output", self.cl_mem["output"]),
                                                                 ("output_size", numpy.int32(numpy.prod(self.shape))),
                                                                 ("local_high", pyopencl.LocalMemory(self.BLOCK_SIZE * 4)),
                                                                 ("local_size", numpy.int32(self.BLOCK_SIZE))])
        self.cl_kernel_args["copy_intense"] = OrderedDict((("peak_position", self.cl_mem["output"]),
                                                        ("count", self.cl_mem["count"]),
                                                        ("image", self.cl_mem["image"]),
                                                        ("peak_intensity", self.cl_mem["peak_intensity"])))

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
        self.profile_multi(events)
        if checksum is not None:
            self.on_device[dest] = checksum

    def _count(self,
               image,
               window=7,
               cutoff=3.0,
               radius=1.0,
               noise=1.0
               ):
        """Just count the number of peaks

        Note: this method in unprotected

        See doc of `count`
        :return: number of peak found in image
        """
        self.send_buffer(image, "image", force_cast=True)
        self.kernels.memset_int(self.queue, (1,), (1,), *list(self.cl_kernel_args["memset_int"].values()))

        kw = self.cl_kernel_args["simple_spot_finder"]
        kw["half_wind_height"] = kw["half_wind_width"] = numpy.int32(window // 2)
        kw["cutoff"] = numpy.float32(cutoff)
        kw["radius"] = numpy.float32(radius)
        kw["noise"] = numpy.float32(noise)
        wg = self.workgroup_size["simple_spot_finder"]
        ev = self.kernels.simple_spot_finder(self.queue, self.wdim, wg, *list(kw.values()))
        count = numpy.empty(1, dtype=numpy.int32)
        copy_count = pyopencl.enqueue_copy(self.queue, count, self.cl_mem["count"])

        self.profile_multi([
                EventDescription("simple_spot_finder", ev),
                EventDescription("copy_count", copy_count)])
        return count[0]

    def count(self,
               image,
               window=7,
               cutoff=3.0,
               radius=1.0,
               noise=1.0
               ):
        """Just count the number of peaks in the image

        A peak is a positive outlier at background + cutoff * deviation
        where the background is assessed as the mean over a patch of size (window x window)
        The deviation is the std of the same patch.

        :param image: 2d array with an image
        :param window: size of the window, i.e. 7 for 7x7 patch size.
        :param threshhold: keep peaks with I > mean + cutoff*std
        :param radius: keep points with centroid on center within this radius (in pixel)
        :param noise: minimum signal for peak to discard noisy region.
        :return: number of peak found in image
        """
        if image.shape != self.shape:
            raise RuntimeError("image shape does not match")
        with self.sem:
            return self._count(image, window, cutoff, radius, noise)

    def sparsify(self,
                 image,
                 window=7,
                 cutoff=3.0,
                 radius=1.0,
                 noise=1.0
                 ):
        """
        Search for peaks in this image and return a list of them

        :param image: 2d array with an image
        :param window: size of the window, i.e. 7 for 7x7 patch size.
        :param threshhold: keep peaks with I > mean + cutoff*std
        :param radius: keep points with centroid on center within this radius (in pixel)
        :param noise: minimum signal for peak to discard noisy region.
        :return: SparseFrame object, see `intensity`, `x` and `y` properties
        """
        if image.shape != self.shape:
            raise RuntimeError("image shape does not match")
        with self.sem:
            count = self._count(image, window, cutoff, radius, noise)
            kw = self.cl_kernel_args["copy_intense"]
            size = int(count + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE * self.BLOCK_SIZE

            copy_intense = self.program.copy_intense(self.queue, (size,), (self.BLOCK_SIZE,),
                                           *list(kw.values()))

            indexes = numpy.empty(count, dtype=numpy.int32)
            values = numpy.empty(count, dtype=numpy.float32)

            copy_index = pyopencl.enqueue_copy(self.queue, indexes, self.cl_mem["output"])
            copy_value = pyopencl.enqueue_copy(self.queue, values, self.cl_mem["peak_intensity"])
        self.profile_multi([EventDescription("copy D->D values", copy_intense),
                            EventDescription("copy D->H index", copy_index),
                            EventDescription("copy D->H values", copy_value)
                          ])
        result = SparseFrame(indexes, values)
        result._shape = self.shape
        result._compute_engine = self.__class__.__name__
        result._mask = self.on_device["mask"]
        result._clip_cutoff = cutoff
        result._noise = noise
        result._radius = radius
        return result

    __call__ = sparsify

#===============================================================================
# Rebuild an array from sparse informations
#===============================================================================


def densify(sparse):
    """Convert a SparseFrame object into a dense image

    :param sparse: SparseFrame object
    :return: dense image as numpy array
    """
    if not isinstance(sparse, SparseFrame):
        raise RuntimeError("Expected sparseFrame as input")
    background = numpy.array(sparse.background_avg, dtype=numpy.float64)  # explicitly make a copy
    if background is None:
        dense = numpy.zeros(sparse.shape)
    else:
        # the mask contains the 2D radius with NaNs at masked positions
        dense = numpy.interp(sparse.mask, sparse.radius, background)
    flat = dense.ravel()
    flat[sparse.index] = sparse.intensity
    if sparse.mask is not None:
        if numpy.issubdtype(sparse.mask.dtype, numpy.integer):
            masked = numpy.where(sparse.mask)
        else:
            masked = numpy.where(numpy.logical_not(numpy.isfinite(sparse.mask)))
    if numpy.issubdtype(sparse.dtype, numpy.integer):
        dense = numpy.round(dense)
        dense[masked] = sparse.dummy
    else:
        dense[masked] = numpy.nan
    return numpy.ascontiguousarray(dense, sparse.dtype)
