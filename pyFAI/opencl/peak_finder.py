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
__date__ = "06/12/2019"
__copyright__ = "2014-2019, ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import logging
from collections import OrderedDict
import numpy
from ..utils import EPS32
from .azim_csr import OCL_CSR_Integrator, BufferDescription, EventDescription, mf, calc_checksum, pyopencl

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
        nbin = lut[2].size-1
        self.buffers += [BufferDescription("radius1d", nbin, numpy.float32, mf.READ_ONLY),
                         BufferDescription("counter", 1, numpy.float32, mf.WRITE_ONLY),
                         ]
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
            events+=[EventDescription("memset peak_position", memset1), EventDescription("memset counter", memset2)]

            if radial_range is not None:
                kw_proj["radius_min"] = numpy.float32(min(radial_range))
                kw_proj["radius_max"] = numpy.float32(max(radial_range) * EPS32)
            else:
                kw_proj["radius_min"] = numpy.float32(0.0)
                kw_proj["radius_max"] = numpy.float32(numpy.finfo(numpy.float32).max)

            kw_proj["noise"] = numpy.float32(noise)
            
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
        #res = Integrate1dtpl(self.bin_centers, avgint, stderr, merged[:, 0], merged[:, 2], merged[:, 4], merged[:, 6])
        #"position intensity error signal variance normalization count"
        return high

    # Name of the default "process" method
    __call__ = peak_finder

