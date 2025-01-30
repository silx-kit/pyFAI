# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2024 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
OpenCL implementation of the preproc module
"""

__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__date__ = "19/11/2024"
__copyright__ = "2015-2017, ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import logging
logger = logging.getLogger(__name__)

from collections import OrderedDict
import numpy
from . import pyopencl
if pyopencl is None:
    raise ImportError("pyopencl is not installed")
from . import mf, processing, OpenclProcessing, dtype_converter
from ..containers import ErrorModel
EventDescription = processing.EventDescription
BufferDescription = processing.BufferDescription


class OCL_Preproc(OpenclProcessing):
    """OpenCL class for pre-processing ... mainly for demonstration"""
    buffers = [BufferDescription("output", 4, numpy.float32, mf.READ_WRITE),
               BufferDescription("image", 1, numpy.float32, mf.READ_WRITE),
               BufferDescription("image_raw", 8, numpy.uint8, mf.READ_WRITE),
               BufferDescription("temp", 4, numpy.uint8, mf.READ_WRITE),
               BufferDescription("variance", 1, numpy.float32, mf.READ_WRITE),
               BufferDescription("dark", 1, numpy.float32, mf.READ_WRITE),
               BufferDescription("dark_variance", 1, numpy.float32, mf.READ_ONLY),
               BufferDescription("flat", 1, numpy.float32, mf.READ_ONLY),
               BufferDescription("polarization", 1, numpy.float32, mf.READ_ONLY),
               BufferDescription("solidangle", 1, numpy.float32, mf.READ_ONLY),
               BufferDescription("absorption", 1, numpy.float32, mf.READ_ONLY),
               BufferDescription("mask", 1, numpy.int8, mf.READ_ONLY),
               ]
    kernel_files = ["pyfai:openCL/preprocess.cl"]
    mapping = {numpy.int8: "s8_to_float",
               numpy.uint8: "u8_to_float",
               numpy.int16: "s16_to_float",
               numpy.uint16: "u16_to_float",
               numpy.uint32: "u32_to_float",
               numpy.uintc: "u32_to_float",
               numpy.int32: "s32_to_float",
               numpy.intc: "s32_to_float"}

    def __init__(self, image_size=None, image_dtype=None, image=None,
                 dark=None, flat=None, solidangle=None, polarization=None, absorption=None,
                 mask=None, dummy=None, delta_dummy=None, empty=None,
                 split_result=False, calc_variance=False, error_model=ErrorModel.NO,
                 ctx=None, devicetype="all", platformid=None, deviceid=None,
                 block_size=32, profile=False,
                 ):
        """
        :param image_size: (int) number of element of the input image
        :param image_dtype: dtype of the input image
        :param image: retrieve image_size and image_dtype from template
        :param dark: dark current image as numpy array
        :param flat: flat field image as numpy array
        :param solidangle: solid angle image as numpy array
        :param absorption: absorption image  as numpy array
        :param mask: array of int8 with 0 where the data are valid
        :param dummy: value of impossible values: dynamic mask
        :param delta_dummy: precision for dummy values
        :param empty: value to be assigned to pixel without contribution (i.e masked)
        :param split_result: return the result a tuple: data, [variance], normalization, so the last dim becomes 2 or 3, can be also 4 for full error propagation
        :param calc_variance: report the result as  data, variance, normalization
        :param error_model: default error-model to be used
        :param ctx: actual working context, left to None for automatic
                    initialization from device type or platformid/deviceid
        :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
        :param platformid: integer with the platform_identifier, as given by clinfo
        :param deviceid: Integer with the device identifier, as given by clinfo
        :param block_size: preferred workgroup size, may vary depending on the outpcome of the compilation
        :param profile: switch on profiling to be able to profile at the kernel level,
                        store profiling elements (makes code slower)
        """
        OpenclProcessing.__init__(self, ctx, devicetype, platformid, deviceid, block_size, profile)
        self.size = image_size if image is None else image.size
        self.shape = (image_size,) if image is None else image.shape
        self.input_dtype = image_dtype if image is None else image.dtype.type
        self.buffers = [BufferDescription(i.name, i.size * self.size, i.dtype, i.flags)
                        for i in self.__class__.buffers]
        self.allocate_buffers()
        self.compile_kernels()
        self.buffer_dtype = {i.name:numpy.dtype(i.dtype) for i in self.buffers}
        if error_model:
            calc_variance = True
        if calc_variance:
            split_result = split_result or True
        self.on_host = {"dummy": dummy,
                        "delta_dummy": delta_dummy,
                        "empty": empty,
                        "calc_variance": calc_variance,
                        "split_result": split_result,
                        "error_model": error_model
                        }
        self.set_kernel_arguments()

        self.on_device = {}

        if image is not None:
            # self.send_buffer(image, "image")
            self.on_device["image"] = numpy.empty_like(image)
        if dark is not None:
            assert dark.size == self.size
            self.send_buffer(dark, "dark")
            do_dark = numpy.int8(1)
        else:
            do_dark = numpy.int8(0)

        if flat is not None:
            assert flat.size == self.size
            self.send_buffer(flat, "flat")
            do_flat = numpy.int8(1)
        else:
            do_flat = numpy.int8(0)

        if solidangle is not None:
            assert solidangle.size == self.size
            self.send_buffer(solidangle, "solidangle")
            do_solidangle = numpy.int8(1)
        else:
            do_solidangle = numpy.int8(0)

        if polarization is not None:
            assert polarization.size == self.size
            self.send_buffer(polarization, "polarization")
            do_polarization = numpy.int8(1)
        else:
            do_polarization = numpy.int8(0)

        if absorption is not None:
            assert absorption.size == self.size
            self.send_buffer(absorption, "absorption")
            do_absorption = numpy.int8(1)
        else:
            do_absorption = numpy.int8(0)

        if mask is not None:
            assert mask.size == self.size
            self.send_buffer(mask, "mask")
            do_mask = numpy.int8(1)
        else:
            do_mask = numpy.int8(0)

        for name, kwargs in self.cl_kernel_args.items():
            if "correction" in name:
                kwargs["do_dark"] = do_dark
                kwargs["do_flat"] = do_flat
                kwargs["do_solidangle"] = do_solidangle
                kwargs["do_polarization"] = do_polarization
                kwargs["do_absorption"] = do_absorption
                kwargs["do_mask"] = do_mask

    @property
    def dummy(self):
        return self.on_host["dummy"]

    @dummy.setter
    def dummy(self, value=None):
        self.on_host["dummy"] = value
        if value is None:
            for name, kwargs in self.cl_kernel_args.items():
                if "correction" in name:
                    kwargs["do_dummy"] = numpy.int8(0)
        else:
            for name, kwargs in self.cl_kernel_args.items():
                if "correction" in name:
                    kwargs["do_dummy"] = numpy.int8(1)
                    kwargs["dummy"] = numpy.float32(value)

    @property
    def delta_dummy(self):
        return self.on_host["delta_dummy"]

    @delta_dummy.setter
    def delta_dummy(self, value=None):
        value = value or numpy.float32(0)
        self.on_host["delta_dummy"] = value
        for name, kwargs in self.cl_kernel_args.items():
            if "correction" in name:
                kwargs["delta_dummy"] = numpy.float32(value)

    @property
    def empty(self):
        return self.on_host["empty"]

    @empty.setter
    def empty(self, value=None):
        value = value or numpy.float32(0)
        if self.dummy is None:
            self.on_host["empty"] = value
            for name, kwargs in self.cl_kernel_args.items():
                if "correction" in name:
                    kwargs["dummy"] = numpy.float32(value)

    def set_kernel_arguments(self):
        """Tie arguments of OpenCL kernel-functions to the actual kernels

        """
        if self.on_host["dummy"] is None:
            do_dummy = numpy.int8(0)
            dummy = numpy.float32(self.on_host["empty"] or 0.0)
            delta_dummy = numpy.float32(self.on_host["delta_dummy"] or 0.0)
        else:
            do_dummy = numpy.int8(1)
            dummy = numpy.float32(self.on_host["dummy"])
            delta_dummy = numpy.float32(self.on_host["delta_dummy"] or 0.0)

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
                                                          ("do_dummy", do_dummy),
                                                          ("dummy", dummy),
                                                          ("delta_dummy", delta_dummy),
                                                          ("normalization_factor", numpy.float32(1.0)),
                                                          ("apply_normalization", numpy.int8(0)),
                                                          ("output", self.cl_mem["output"])))

        self.cl_kernel_args["corrections2"] = OrderedDict((("image", self.cl_mem["image"]),
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
                                                           ("do_dummy", do_dummy),
                                                           ("dummy", dummy),
                                                           ("delta_dummy", delta_dummy),
                                                           ("normalization_factor", numpy.float32(1.0)),
                                                           ("apply_normalization", numpy.int8(0)),
                                                           ("output", self.cl_mem["output"])))

        self.cl_kernel_args["corrections3"] = OrderedDict((("image", self.cl_mem["image"]),
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
                                                           ("do_dummy", do_dummy),
                                                           ("dummy", dummy),
                                                           ("delta_dummy", delta_dummy),
                                                           ("normalization_factor", numpy.float32(1.0)),
                                                           ("apply_normalization", numpy.int8(0)),
                                                           ("output", self.cl_mem["output"])))

        self.cl_kernel_args["corrections4a"] = OrderedDict((("image_raw", self.cl_mem["image_raw"]),
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
                                                           ("do_dummy", do_dummy),
                                                           ("dummy", dummy),
                                                           ("delta_dummy", delta_dummy),
                                                           ("normalization_factor", numpy.float32(1.0)),
                                                           ("apply_normalization", numpy.int8(0)),
                                                           ("output", self.cl_mem["output"])))


    def compile_kernels(self, kernel_files=None):
        """Call the OpenCL compiler

        :param kernel_files: list of path to the kernel
                             (by default use the one declared in the class)
        """
        # concatenate all needed source files into a single openCL module
        kernel_files = kernel_files or self.kernel_files
        # Explicit handling of fp64 since Apple silicon compiler wrongly clams fp64 support see issue #2339
        fp64_support = 1 if "cl_khr_fp64" in self.ctx.devices[0].extensions else 0
        compile_options = f"-D NIMAGE={self.size} -D cl_khr_fp64={fp64_support}"
        OpenclProcessing.compile_kernels(self, kernel_files, compile_options)

    def send_buffer(self, data, dest, convert=True):
        """Send a numpy array to the device

        :param data: numpy array with data
        :param dest: name of the buffer as registered in the class.
        :param convert: if True (default) convert dtype on GPU, if false, leave as it is in buffer named `dest_raw`
        :return: the destination buffer/array
        """
        dest_type = self.buffer_dtype[dest]
        events = []
        if convert:
            actual_dest = self.cl_mem[dest]
            if (data.dtype == dest_type) or (data.dtype.itemsize > dest_type.itemsize):
                copy_image = pyopencl.enqueue_copy(self.queue, actual_dest, numpy.ascontiguousarray(data, dest_type))
                events.append(EventDescription(f"copy H->D {dest}", copy_image))
            else:
                copy_image = pyopencl.enqueue_copy(self.queue, self.cl_mem["temp"], numpy.ascontiguousarray(data))
                kernel = self.kernels.get_kernel(self.mapping[data.dtype.type])
                to_float = kernel(self.queue, (self.size,), None, self.cl_mem["temp"], self.cl_mem[dest])
                events += [EventDescription("copy H->D temp", copy_image),
                           EventDescription("convert to float", to_float)]
            self.on_device[dest] = data
        else:
            new_dest = f"{dest}_raw"
            actual_dest = self.cl_mem[new_dest]
            copy_image = pyopencl.enqueue_copy(self.queue, actual_dest, numpy.ascontiguousarray(data))
            events.append(EventDescription(f"copy H->D {new_dest}", copy_image))
            self.on_device[new_dest] = data

        self.profile_multi(events)
        return actual_dest

    def process(self, image,
                dark=None,
                variance=None,
                dark_variance=None,
                normalization_factor=1.0,
                error_model=None,
                split_result=None,
                apply_normalization=False,
                out=None
                ):
        """Perform the pixel-wise operation of the array

        :param raw: numpy array with the input image
        :param dark: numpy array with the dark-current image
        :param variance: numpy array with the variance of input image
        :param dark_variance: numpy array with the variance of dark-current image
        :param normalization_factor: divide the result by this
        :param error_model: set to "poisson"  to set variance=signal (minimum 1). None uses the default from constructor
        :param apply_normalization: correct (directly) the raw signal & variance with normalization, WIP
        :param out: output buffer to save a malloc
        :return: array with processed data,
                may be an array of (data,variance,normalization) depending on class initialization
        """
        with self.sem:
            if dark is not None:
                do_dark = numpy.int8(1)
                if id(dark) != id(self.on_device.get("dark")):
                    self.send_buffer(dark, "dark")
            else:
                do_dark = numpy.int8(0)
            if (variance is not None) and self.on_host.get("calc_variance"):
                if id(variance) != id(self.on_device.get("variance")):
                    self.send_buffer(variance, "variance")
            if (dark_variance is not None) and self.on_host.get("calc_variance"):
                if id(dark_variance) != id(self.on_device.get("dark_variance")):
                    self.send_buffer(dark_variance, "dark_variance")

            if error_model is None:
                error_model = self.on_host.get("error_model")
            else:
                error_model = ErrorModel.NO
            if split_result is None:
                split_result = self.on_host.get("split_result")

            if split_result == 4:
                dshape = self.shape + (4,)
                kernel_name = "corrections4a"
            elif split_result == 3:
                kernel_name = "corrections3"
                dshape = self.shape + (3,)
            elif split_result == 2:
                kernel_name = "corrections2"
                dshape = self.shape + (2,)
            else:
                kernel_name = "corrections"
                dshape = self.shape
            kwargs = self.cl_kernel_args[kernel_name]

            # finally
            if id(image) != id(self.on_device.get("image")):
                if kernel_name == "corrections4a":
                    if (image.dtype.itemsize <= 4):
                        kwargs["image_raw"] = self.send_buffer(image, "image", convert=False)
                        kwargs["dtype"] = dtype_converter(image.dtype)
                    else:
                        kwargs["image_raw"] = self.send_buffer(numpy.ascontiguousarray(image, numpy.float32), "image", convert=False)
                        kwargs["dtype"] = numpy.int8(32)
                else:
                    kwargs["image"] = self.send_buffer(image, "image")

            if out is None:
                dest = numpy.empty(dshape, dtype=numpy.float32)
            else:
                dest = out
                assert dest.dtype == numpy.float32
                assert dest.shape == dshape

            kwargs["do_dark"] = do_dark
            kwargs["normalization_factor"] = numpy.float32(normalization_factor)
            kwargs["apply_normalization"] = numpy.int8(apply_normalization)
            if split_result >= 3:
                kwargs["error_model"] = numpy.int8(error_model)
            if (kernel_name == "corrections3") and (self.on_device.get("dark_variance") is not None):
                kwargs["do_dark_variance"] = do_dark
            kernel = self.kernels.get_kernel(kernel_name)
            # logger.warning(f"Using kernel {kernel_name}")
            evt = kernel(self.queue, (self.size,), None, *list(kwargs.values()))

            copy_result = pyopencl.enqueue_copy(self.queue, dest, self.cl_mem["output"])
            copy_result.wait()
            self.profile_multi([EventDescription("preproc", evt),
                                EventDescription("copy result", copy_result)])
        return dest

    def __copy__(self):
        """Shallow copy of the object

        :return: copy of the object
        """
        return self.__class__(dummy=self.dummy, delta_dummy=self.delta_dummy, empty=self.self.empty,
                              ctx=self.ctx, profile=self.profile, **self.on_device)

    def __deepcopy__(self, memo=None):
        """deep copy of the object

        :return: deepcopy of the object
        """
        if memo is None:
            memo = {}

        memo[id(self.ctx)] = self.ctx
        od2 = {}
        for k, v in self.on_device.items():
            od2[k] = v.copy()
            memo[id(v)] = od2[k]
        new_obj = self.__class__(dummy=self.dummy, delta_dummy=self.delta_dummy, empty=self.self.empty,
                                 ctx=self.ctx, profile=self.profile, **self.on_device)
        memo[id(self)] = new_obj
        return new_obj


def preproc(raw,
            dark=None,
            flat=None,
            solidangle=None,
            polarization=None,
            absorption=None,
            mask=None,
            dummy=None,
            delta_dummy=None,
            normalization_factor=1.0,
            empty=None,
            split_result=False,
            variance=None,
            dark_variance=None,
            error_model=ErrorModel.NO,
            apply_normalization=False,
            dtype=numpy.float32,
            out=None
            ):
    r"""Common preprocessing step, implemented using OpenCL. May be inefficient

    :param data: raw value, as a numpy array, 1D or 2D
    :param mask: array non null  where data should be ignored
    :param dummy: value of invalid data
    :param delta_dummy: precision for invalid data
    :param dark: array containing the value of the dark noise, to be subtracted
    :param flat: Array containing the flatfield image. It is also checked for dummies if relevant.
    :param solidangle: the value of the solid_angle. This processing may be performed during the rebinning instead. left for compatibility
    :param polarization: Correction for polarization of the incident beam
    :param absorption: Correction for absorption in the sensor volume
    :param normalization_factor: final value is divided by this
    :param empty: value to be given for empty pixels
    :param split_result: set to true to separate numerator from denominator and return an array of float2 or float3 (with variance)
    :param variance: provide an estimation of the variance, enforce split_result=True and return an float3 array with variance in second position.
    :param error_model: set to POISSONIAN to assume variance=signal
    :param apply_normalization: correct (directly) the raw signal & variance with normalization, WIP
    :param dtype: dtype for all processing
    :param out: output buffer to save a malloc

    All calculation are performed in single precision floating point (32 bits).

    NaN are always considered as invalid values

    if neither empty nor dummy is provided, empty pixels are 0.
    Empty pixels are always zero in "split_result" mode

    Split result:

    * When set to False, i.e the default, the pixel-wise operation is:
      :math:`I = (raw - dark)/(flat * solidangle * polarization * absorption)`
      Invalid pixels are set to the dummy or empty value.

    * When split_ressult is set to True, each result result is a float2
      or a float3 (with an additional value for the variance) as such:

      :math:`I = [(raw - dark), (variance), (flat * solidangle * polarization * absorption)]`

      Empty pixels will have all their 2 or 3 values to 0 (and not to dummy or empty value)

    * If error_model is set to poissonian, the variance is evaluated as (raw + dark) minimum (1)
    """
    if raw.dtype.itemsize > 4:  # use numpy to cast to float32
        raw = numpy.ascontiguousarray(raw, numpy.float32)

    engine = OCL_Preproc(image=raw, dark=dark, flat=flat, solidangle=solidangle,
                         polarization=polarization, absorption=absorption,
                         mask=mask, dummy=dummy, delta_dummy=delta_dummy, empty=empty,
                         split_result=split_result,
                         calc_variance=(variance is not None),
                         error_model=error_model,
                         devicetype="all")
    result = engine.process(raw, dark=dark, variance=variance,
                            dark_variance=dark_variance,
                            normalization_factor=normalization_factor,
                            apply_normalization=apply_normalization,
                            out=out)

    if result.dtype != dtype:
        result = numpy.ascontiguousarray(result, dtype)
    return result
