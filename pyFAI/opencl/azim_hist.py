# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2021 European Synchrotron Radiation Facility, Grenoble, France
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

"""
Histogram (atomic-add) based integrator 

"""
__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__date__ = "10/01/2022"
__copyright__ = "2012-2021, ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import logging
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
from ..utils.decorators import deprecated
EventDescription = processing.EventDescription
OpenclProcessing = processing.OpenclProcessing
BufferDescription = processing.BufferDescription
from ..utils import calc_checksum
logger = logging.getLogger(__name__)


class OCL_Histogram1d(OpenclProcessing):
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
    kernel_files = ["silx:opencl/doubleword.cl",
                    "pyfai:openCL/preprocess.cl",
                    "pyfai:openCL/ocl_histo.cl"
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

    def __init__(self, radial, bins, radial_checksum=None, empty=None, unit=None,
                 azimuthal=None, azimuthal_checksum=None,
                 mask=None, mask_checksum=None,
                 ctx=None, devicetype="all", platformid=None, deviceid=None,
                 block_size=None, profile=False):
        """
        :param radial: array with the radial position of every single pixel. Same as image size
        :param bins: number of bins on which to histogram
        :param checksum: pre-calculated checksum of the radial array to prevent re-calculating it :)
        :param empty: value to be assigned to bins without contribution from any pixel
        :param unit: just a place_holder for the units of radial array.
        :param azimuthal: array with the azimuthal position, same size as radial
        :param azimuthal_checksum: Checksum of the azimuthal array
        :param mask: Array with the mask, 0 for valid values, anything for masked pixels, same size as radial
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
        self.size = numpy.uint32(radial.size)
        self.empty = numpy.float32(empty) if empty is not None else numpy.float32(0.0)
        self.radial_mini = numpy.float32(numpy.min(radial))
        self.radial_maxi = numpy.float32(numpy.max(radial) * (1.0 + numpy.finfo(numpy.float32).eps))
        self.degraded = False
        if not radial_checksum:
            radial_checksum = calc_checksum(radial)
        self.radial = radial
        self.mask_checksum = None
        self.check_azim = numpy.int8(0)
        self.azim_mini = self.azim_maxi = numpy.float32(0.0)
        self.check_mask = False
        self.on_device = {"radial": radial_checksum,
                          "azimuthal": None,
                          "dark": None,
                          "flat": None,
                          "polarization": None,
                          "solidangle": None,
                          "absorption": None,
                          "mask":None}

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
        self.send_buffer(radial, "radial", radial_checksum)
        if azimuthal is not None:
            self.check_azim = numpy.int8(1)
            self.azim_mini = numpy.float32(numpy.min(azimuthal))
            self.azim_maxi = numpy.float32(numpy.max(azimuthal) * (1.0 + numpy.finfo(numpy.float32).eps))
            if azimuthal_checksum is None:
                azimuthal_checksum = calc_checksum(azimuthal)
            self.send_buffer(azimuthal, "azimuthal", azimuthal_checksum)
            self.azimuthal = azimuthal
        else:
            self.azimuthal = None
        if mask is not None:
            self.check_mask = True
            if mask_checksum is None:
                mask_checksum = calc_checksum(mask)
            self.send_buffer(numpy.ascontiguousarray(mask, dtype=numpy.int8), "mask", mask_checksum)
            self.cl_kernel_args["corrections4"]["do_mask"] = numpy.int8(1)

    def __copy__(self):
        """Shallow copy of the object

        :return: copy of the object
        """
        if self.on_device.get("mask"):
            mask = self.cl_mem["mask"].get()
        else:
            mask = None
        if self.on_device.get("azimuthal"):
            azimuthal = self.cl_mem["azimuthal"].get()
        else:
            azimuthal = None

        return self.__class__(self.radial,
                              self.bins,
                              checksum=self.on_device.get("radial"),
                              empty=self.empty,
                              mask=mask, mask_checksum=self.on_device.get("mask"),
                              azimuthal=azimuthal, azimuthal_checksum=self.on_device.get("azimuthal"),
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
        memo[id(self.radial)] = radial
        if self.on_device.get("mask"):
            mask = self.cl_mem["mask"].get()
        else:
            mask = None
        if self.on_device.get("azimuthal"):
            mask = self.cl_mem["mask"].get()
        else:
            mask = None
        if self.on_device.get("azimuthal"):
            azimuthal = self.cl_mem["azimuthal"].get()
        else:
            azimuthal = None

        new_obj = self.__class__(radial, self.bins,
                                 checksum=self.on_device.get("radial"),
                                 empty=self.empty,
                                 mask=mask, mask_checksum=self.on_device.get("mask"),
                                 azimuthal=azimuthal, azimuthal_checksum=self.on_device.get("azimuthal"),
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
                self.degraded = True
            else:
                logger.error("Failed to compile kernel ! Check the compiler. %s", error)

        for kernel_name, kernel in self.kernels.get_kernels().items():
            wg = kernel_workgroup_size(self.program, kernel)
            self.workgroup_size[kernel_name] = (min(wg, self.BLOCK_SIZE),)  # this is a tuple

    def set_kernel_arguments(self):
        """Tie arguments of OpenCL kernel-functions to the actual kernels

        """
        self.cl_kernel_args["corrections4"] = OrderedDict((("image", self.cl_mem["image"]),
                                                           ("poissonian", numpy.int8(0)),
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

        self.cl_kernel_args["histogram_1d_preproc"] = OrderedDict((("radial", self.cl_mem["radial"]),
                                                                   ("azimuthal", self.cl_mem["azimuthal"]),
                                                                   ("preproc4", self.cl_mem["output4"]),
                                                                   ("histo_sig", self.cl_mem["histo_sig"]),
                                                                   ("histo_var", self.cl_mem["histo_var"]),
                                                                   ("histo_nrm", self.cl_mem["histo_nrm"]),
                                                                   ("histo_cnt", self.cl_mem["histo_cnt"]),
                                                                   ("size", self.size),
                                                                   ("bins", self.bins),
                                                                   ("radial_mini", self.radial_mini),
                                                                   ("radial_maxi", self.radial_maxi),
                                                                   ("check_azim", self.check_azim),
                                                                   ("azim_mini", self.azim_mini),
                                                                   ("azim_maxi", self.azim_maxi)))

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
                  poissonian=False,
                  variance=None, dark_variance=None,
                  flat=None, solidangle=None, polarization=None, absorption=None,
                  dark_checksum=None, flat_checksum=None, solidangle_checksum=None,
                  polarization_checksum=None, absorption_checksum=None,
                  preprocess_only=False, safe=True,
                  normalization_factor=1.0,
                  radial_range=None, azimuth_range=None,
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
        :param poissonian: set to assume variance is data (minimum 1) 
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
        :param radial_range: provide lower and upper bound for radial array
        :param azimuth_range: provide lower and upper bound for azimuthal array
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
            kw_correction["poissonian"] = numpy.int8(1 if poissonian else 0)
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

            for k, v in kw_correction.items(): print(k, v)
            ev = self.kernels.corrections4(self.queue, self.wdim_data, self.workgroup_size["corrections4"],
                                           *list(kw_correction.values()))
            events.append(EventDescription("corrections4", ev))

            if preprocess_only:
                image = numpy.empty(data.shape + (4,), dtype=numpy.float32)
                ev = pyopencl.enqueue_copy(self.queue, image, self.cl_mem["output4"])
                events.append(EventDescription("copy D->H image", ev))
                if self.profile:
                    self.events += events
                ev.wait()
                return image

            if radial_range is not None:
                radial_mini = numpy.float32(radial_range[0])
                radial_maxi = numpy.float32(radial_range[1] * (1.0 + numpy.finfo(numpy.float32).eps))
            else:
                radial_mini = self.radial_mini
                radial_maxi = self.radial_maxi
            kw_histogram["radial_mini"] = radial_mini
            kw_histogram["radial_maxi"] = radial_maxi

            if azimuth_range is not None:
                azim_mini = numpy.float32(azimuth_range[0])
                azim_maxi = numpy.float32(azimuth_range[1] * (1.0 + numpy.finfo(numpy.float32).eps))
                kw_histogram["check_azim"] = numpy.int8(1)
                if self.on_device.get("azimuthal") is None:
                    raise RuntimeError("Unable to use azimuthal range: azimuthal array not provided")
            else:
                azim_mini = self.azim_mini
                azim_maxi = self.azim_maxi
                kw_histogram["check_azim"] = numpy.int8(0)
            kw_histogram["azim_mini"] = azim_mini
            kw_histogram["azim_maxi"] = azim_maxi

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
            delta = (radial_maxi - radial_mini) / self.bins
            positions = numpy.linspace(radial_mini + 0.5 * delta, radial_maxi - 0.5 * delta, self.bins)

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
    kernel_files = ["silx:opencl/doubleword.cl",
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
                 radial_checksum=None, azimuthal_checksum=None,
                 empty=None, unit=None, mask=None, mask_checksum=None,
                 ctx=None, devicetype="all", platformid=None, deviceid=None,
                 block_size=None, profile=False):
        """
        :param radial: array with the radial position of every single pixel. Same as image size
        :param azimuthal: array with the azimuthal position of every single pixel. Same as image size
        :param bins_radial: number of bins on which to histogram is calculated in radial direction
        :param bins_azimuthal: number of bins on which to histogram is calculated in azimuthal direction
        :param radial_checksum: pre-calculated checksum of the position array to prevent re-calculating it :)
        :param azimuthal_checksum: pre-calculated checksum of the position array to prevent re-calculating it :)

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
        self.bins_radial = numpy.uint32(bins_radial)
        self.bins_azimuthal = numpy.uint32(bins_azimuthal)

        OCL_Histogram1d.__init__(self,
                                 radial, bins_radial * bins_azimuthal, radial_checksum=radial_checksum,
                                 azimuthal=azimuthal, azimuthal_checksum=azimuthal_checksum,
                                 empty=empty, unit=unit, mask=mask, mask_checksum=mask_checksum,
                                 ctx=ctx, devicetype=devicetype,
                                 platformid=platformid, deviceid=deviceid,
                                 block_size=block_size, profile=profile)

    def __copy__(self):
        """Shallow copy of the object

        :return: copy of the object
        """
        if self.on_device.get("mask"):
            mask = self.cl_mem["mask"].get()
        else:
            mask = None

        return self.__class__(self.radial, self.azimuthal,
                              self.bins_radial, self.bins_azimuthal,
                              checksum_radial=self.on_device.get("radial"),
                              checksum_azimuthal=self.on_device.get("azimuthal"),
                              empty=self.empty,
                              mask=mask, mask_checksum=self.on_device.get("mask"),
                              ctx=self.ctx,
                              block_size=self.block_size,
                              profile=self.profile)

    def __deepcopy__(self, memo=None):
        """deep copy of the object

        :return: deepcopy of the object
        """
        if self.on_device.get("mask"):
            mask = self.cl_mem["mask"].get()
        else:
            mask = None
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
                                 mask=mask, mask_checksum=self.on_device.get("mask"),
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
                                                                   ("radial_mini", self.radial_mini),
                                                                   ("radial_maxi", self.radial_maxi),
                                                                   ("azim_mini", self.azim_mini),
                                                                   ("azim_maxi", self.azim_maxi)))

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
        :param radial_range: provide lower and upper bound for radial array
        :param azimuth_range: provide lower and upper bound for azimuthal array
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
                radial_mini = numpy.float32(min(radial_range))
                radial_maxi = numpy.float32(max(radial_range) * (1.0 + numpy.finfo(numpy.float32).eps))
            else:
                radial_mini = self.radial_mini
                radial_maxi = self.radial_maxi
            kw_histogram["radial_mini"] = radial_mini
            kw_histogram["radial_maxi"] = radial_maxi

            if azimuthal_range:
                azim_mini = numpy.float32(min(azimuthal_range))
                azim_maxi = numpy.float32(max(azimuthal_range) * (1.0 + numpy.finfo(numpy.float32).eps))
            else:
                azim_mini = self.azim_mini
                azim_maxi = self.azim_maxi
            kw_histogram["azim_mini"] = azim_mini
            kw_histogram["azim_maxi"] = azim_maxi

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

            delta_radial = (radial_maxi - radial_mini) / self.bins_radial
            delta_azimuthal = (azim_maxi - azim_mini) / self.bins_azimuthal

            pos_radial = numpy.linspace(radial_mini + 0.5 * delta_radial, radial_maxi - 0.5 * delta_radial, self.bins_radial)
            pos_azim = numpy.linspace(azim_mini + 0.5 * delta_azimuthal, azim_maxi - 0.5 * delta_azimuthal, self.bins_azimuthal)
            ev.wait()
        if self.profile:
            self.events += events
        return Integrate2dtpl(pos_radial, pos_azim, intensity.T, error.T,
                              histo_signal[:,:, 0].T,
                              histo_variance[:,:, 0].T,
                              histo_normalization[:,:, 0].T,
                              histo_count.T)

    # Name of the default "process" method
    __call__ = integrate
