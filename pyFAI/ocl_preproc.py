# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2017 European Synchrotron Radiation Facility, Grenoble, France
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

from __future__ import absolute_import, print_function, division

__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__date__ = "20/01/2017"
__copyright__ = "2015-2017, ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import os
import logging
import threading
import gc
import numpy
from .opencl import ocl, pyopencl, allocate_cl_buffers, release_cl_buffers
from .utils import concatenate_cl_kernel
if pyopencl:
    mf = pyopencl.mem_flags
else:
    raise ImportError("pyopencl is not installed")

from collections import namedtuple
BufferDescription = namedtuple("BufferDescription", ["name", "flags", "dtype", "size"])
EventDescription = namedtuple("EventDescription", ["name", "event"])

# try:
#     from .ext.fastcrc import crc32
# except:
#     from zlib import crc32
logger = logging.getLogger("pyFAI.ocl_preproc")


class OpenclProcessing(object):
    """Abstract class for all OpenCL processing.
    
    This class provides:
    * Generation of the context, queues, profiling mode
    * Additional function to allocate/free all buffers declared as static attributes of the class 
    * Functions to compile kernels, cache them and clean them  
    * helper functions to clone the object
    """
    # The last parameter
    buffers = [BufferDescription("output", mf.WRITE_ONLY, numpy.float32, 10),
               ]
    # list of kernel source files to be concatenated before compilation of the program
    kernel_files = []

    def __init__(self, ctx=None, devicetype="all", platformid=None, deviceid=None,
                 block_size=None, profile=False):
        """Constructor of the abstract OpenCL processing class
        
        :param ctx: actual working context, left to None for automatic 
                    initialization from device type or platformid/deviceid 
        :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
        :param platformid: integer with the platform_identifier, as given by clinfo
        :param deviceid: Integer with the device identifier, as given by clinfo
        :param block_size: preferred workgroup size, may vary depending on the outpcome of the compilation
        :param profile: switch on profiling to be able to profile at the kernel level,
                        store profiling elements (makes code slower) 
        """
        self.sem = threading.Semaphore()
        self.profile = bool(profile)
        self.events = []  # List with of EventDescription, kept for profiling
        self.cl_mem = {}  # dict with all buffer allocated
        self.cl_program = None  # The actual OpenCL program
        self.cl_kernel_args = {}  # dict with all kernel arguments
        if ctx:
            self.ctx = ctx
            device_name = self.ctx.devices[0].name.strip()
            platform_name = self.ctx.devices[0].platform.name.strip()
            platform = ocl.get_platform(platform_name)
            self.device = platform.get_device(device_name)
            # self.device = platform.id, device.id
        else:
            self.ctx = ocl.create_context(devicetype=devicetype, platformid=platformid, deviceid=deviceid)
            device_name = self.ctx.devices[0].name.strip()
            platform_name = self.ctx.devices[0].platform.name.strip()
            platform = ocl.get_platform(platform_name)
            self.device = platform.get_device(device_name)
            # self.device = platform.id, device.id

        if profile:
            self.queue = pyopencl.CommandQueue(self.ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
        else:
            self.queue = pyopencl.CommandQueue(self.ctx)

        self.block_size = block_size

    def __del__(self):
        """Destructor: release all buffers and programs
        """
        self.free_kernels()
        self.free_buffers()
        self.queue = None
        self.ctx = None
        gc.collect()

    def allocate_buffers(self, buffers=None):
        """
        Allocate OpenCL buffers required for a specific configuration

        Note that an OpenCL context also requires some memory, as well
        as Event and other OpenCL functionalities which cannot and are
        not taken into account here.  The memory required by a context
        varies depending on the device. Typical for GTX580 is 65Mb but
        for a 9300m is ~15Mb In addition, a GPU will always have at
        least 3-5Mb of memory in use.  Unfortunately, OpenCL does NOT
        have a built-in way to check the actual free memory on a
        device, only the total memory.
        """
        if buffers is None:
            buffers = self.buffers
        self.cl_mem = allocate_cl_buffers(buffers, self.device, self.ctx)

    def free_buffers(self):
        """free all memory allocated on the device
        """
        self.cl_mem = release_cl_buffers(self.cl_mem)

    def compile_kernels(self, kernel_files=None, compile_options=None):
        """Call the OpenCL compiler
        
        :param kernel_files: list of path to the kernel 
        (by default use the one declared in the class)
        """
        # concatenate all needed source files into a single openCL module
        kernel_files = kernel_files or self.kernel_files
        kernel_src = concatenate_cl_kernel(kernel_files)

        compile_options = compile_options or ""
        logger.info("Compiling file %s with options %s", kernel_files, compile_options)
        try:
            self.program = pyopencl.Program(self.ctx, kernel_src).build(options=compile_options)
        except (pyopencl.MemoryError, pyopencl.LogicError) as error:
            raise MemoryError(error)

    def free_kernels(self):
        """Free all kernels
        """
        for kernel in self.cl_kernel_args:
            self.cl_kernel_args[kernel] = []
        self.program = None

    def log_profile(self):
        """If we are in profiling mode, prints out all timing for every single OpenCL call
        """
        t = 0.0
        out = ["", "Profiling info for OpenCL %s" % self.__class__.__name__]
        if self.profile:
            for e in self.events:
                if "__len__" in dir(e) and len(e) >= 2:
                    et = 1e-6 * (e[1].profile.end - e[1].profile.start)
                    out.append("%50s:\t%.3fms" % (e[0], et))
                    t += et

        out.append("_" * 80)
        out.append("%50s:\t%.3fms" % ("Total execution time", t))
        logger.info(os.linesep.join(out))

# This should be implemented by concrete class
#     def __copy__(self):
#         """Shallow copy of the object
#
#         :return: copy of the object
#         """
#         return self.__class__((self._data, self._indices, self._indptr),
#                               self.size, block_size=self.BLOCK_SIZE,
#                               platformid=self.platform.id,
#                               deviceid=self.device.id,
#                               checksum=self.on_device.get("data"),
#                               profile=self.profile, empty=self.empty)
#
#     def __deepcopy__(self, memo=None):
#         """deep copy of the object
#
#         :return: deepcopy of the object
#         """
#         if memo is None:
#             memo = {}
#         new_csr = self._data.copy(), self._indices.copy(), self._indptr.copy()
#         memo[id(self._data)] = new_csr[0]
#         memo[id(self._indices)] = new_csr[1]
#         memo[id(self._indptr)] = new_csr[2]
#         new_obj = self.__class__(new_csr, self.size,
#                                  block_size=self.BLOCK_SIZE,
#                                  platformid=self.platform.id,
#                                  deviceid=self.device.id,
#                                  checksum=self.on_device.get("data"),
#                                  profile=self.profile, empty=self.empty)
#         memo[id(self)] = new_obj
#         return new_obj


class OCL_Preproc(OpenclProcessing):
    """OpenCL class for pre-processing ... mainly for demonstration"""
    buffers = [
               BufferDescription("output", mf.WRITE_ONLY, numpy.float32, 3),
               BufferDescription("image_raw", mf.READ_ONLY, numpy.float32, 1),
               BufferDescription("image", mf.READ_WRITE, numpy.float32, 1),
               BufferDescription("variance", mf.READ_WRITE, numpy.float32, 1),
               BufferDescription("dark", mf.READ_WRITE, numpy.float32, 1),
               BufferDescription("dark_variance", mf.READ_ONLY, numpy.float32, 1),
               BufferDescription("flat", mf.READ_ONLY, numpy.float32, 1),
               BufferDescription("polarization", mf.READ_ONLY, numpy.float32, 1),
               BufferDescription("solidangle", mf.READ_ONLY, numpy.float32, 1),
               BufferDescription("absorption", mf.READ_ONLY, numpy.float32, 1),
               BufferDescription("mask", mf.READ_ONLY, numpy.int8, 1),
            ]
    kernel_files = ["preprocess.cl"]
    mapping = {numpy.int8: "s8_to_float",
               numpy.uint8: "u8_to_float",
               numpy.int16: "s16_to_float",
               numpy.uint16: "u16_to_float",
               numpy.uint32: "u32_to_float",
               numpy.int32: "s32_to_float"}

    def __init__(self, image_size=None, image_dtype=None, image=None,
                 dark=None, flat=None, solidangle=None, polarization=None, absorption=None,
                 mask=None, dummy=None, delta_dummy=None, empty=None,
                 split_result=False, calc_variance=False, poissonian=False,
                 ctx=None, devicetype="all", platformid=None, deviceid=None,
                 block_size=32, profile=False,
                 ):
        """
        :param image_size: (int) number of element of the input image 
        :param image_dtype: dtype of the input image
        :param template_image: retrieve image_size and image_dtype from example
        
        :param absorption: 
        :param mask:
        :param dummy:
        :param delta_dummy:
        :param empty: value to be assigned to bins without contribution from any pixel
        
        """
        OpenclProcessing.__init__(self, ctx, devicetype, platformid, deviceid, block_size, profile)
        self.size = image_size or image.size
        self.input_dtype = image_dtype or image.dtype.type
        self.buffers = [BufferDescription(*((i[:-1]) + (i[-1] * self.size,)))
                        for i in self.__class__.buffers]
        self.allocate_buffers()
        if poissonian:
            calc_variance = True
        if calc_variance:
            split_result = True
        self.on_host = {"dummy": dummy,
                        "delta_dummy": delta_dummy,
                        "empty": empty,
                        "poissonian": poissonian,
                        "calc_variance": calc_variance,
                        "split_result": split_result
                        }
        self.set_kernel_arguments()

        self.on_device = {}

        if image is not None:
            self.send_buffer(image, "image")
        if dark is not None:
            assert dark.size == self.size
            self.send_buffer(dark, "dark")
            self.cl_kernel_args["corrections"][1] = numpy.int8(1)
            self.cl_kernel_args["corrections2"][1] = numpy.int8(1)
            self.cl_kernel_args["corrections2"][1] = numpy.int8(1)
            self.cl_kernel_args["corrections3Poisson"][2] = numpy.int8(1)
        if flat is not None:
            assert flat.size == self.size
            self.send_buffer(flat, "flat")
            self.cl_kernel_args["corrections"][3] = numpy.int8(1)
            self.cl_kernel_args["corrections2"][3] = numpy.int8(1)
            self.cl_kernel_args["corrections2"][3] = numpy.int8(1)
            self.cl_kernel_args["corrections3Poisson"][4] = numpy.int8(1)
        if solidangle is not None:
            assert solidangle.size == self.size
            self.send_buffer(solidangle, "solidangle")
            self.cl_kernel_args["corrections"][5] = numpy.int8(1)
            self.cl_kernel_args["corrections2"][5] = numpy.int8(1)
            self.cl_kernel_args["corrections2"][5] = numpy.int8(1)
            self.cl_kernel_args["corrections3Poisson"][6] = numpy.int8(1)
        if polarization is not None:
            assert polarization.size == self.size
            self.send_buffer(polarization, "polarization")
            self.cl_kernel_args["corrections"][7] = numpy.int8(1)
            self.cl_kernel_args["corrections2"][7] = numpy.int8(1)
            self.cl_kernel_args["corrections2"][7] = numpy.int8(1)
            self.cl_kernel_args["corrections3Poisson"][8] = numpy.int8(1)
        if absorption is not None:
            assert absorption.size == self.size
            self.send_buffer(absorption, "absorption")
            self.cl_kernel_args["corrections"][9] = numpy.int8(1)
            self.cl_kernel_args["corrections2"][9] = numpy.int8(1)
            self.cl_kernel_args["corrections2"][9] = numpy.int8(1)
            self.cl_kernel_args["corrections3Poisson"][10] = numpy.int8(1)
        if mask is not None:
            assert mask.size == self.size
            self.send_buffer(mask, "mask")
            self.cl_kernel_args["corrections"][11] = numpy.int8(1)
            self.cl_kernel_args["corrections2"][11] = numpy.int8(1)
            self.cl_kernel_args["corrections2"][11] = numpy.int8(1)
            self.cl_kernel_args["corrections3Poisson"][12] = numpy.int8(1)
        self.compile_kernels()

    @property
    def dummy(self):
        return self.on_host["dummy"]

    @dummy.setter
    def dummy(self, value=None):
        self.on_host["dummy"] = value
        if value is None:
            for kernel in ("corrections", "corrections2", "corrections3Poisson"):
                self.cl_kernel_args[kernel][13] = numpy.int8(0)
            self.cl_kernel_args["corrections3"][16] = numpy.int8(0)
        else:
            for kernel in ("corrections", "corrections2", "corrections3Poisson"):
                self.cl_kernel_args[kernel][13] = numpy.int8(1)
                self.cl_kernel_args[kernel][14] = numpy.float32(value)
            self.cl_kernel_args["corrections3"][16] = numpy.int8(1)
            self.cl_kernel_args["corrections3"][17] = numpy.float32(value)

    @property
    def delta_dummy(self):
        return self.on_host["delta_dummy"]

    @delta_dummy.setter
    def delta_dummy(self, value=None):
        value = value or numpy.float32(0)
        self.on_host["delta_dummy"] = value
        for kernel in ("corrections", "corrections2", "corrections3Poisson"):
            self.cl_kernel_args[kernel][15] = value
        self.cl_kernel_args["corrections3"][18] = value

    @property
    def empty(self):
        return self.on_host["empty"]

    @empty.setter
    def empty(self, value=None):
        value = value or numpy.float32(0)
        if self.dummy is None:
            self.on_host["empty"] = value
            for kernel in ("corrections", "corrections2", "corrections3Poisson"):
                self.cl_kernel_args[kernel][14] = value
                self.cl_kernel_args["corrections3"][17] = value

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

        self.cl_kernel_args["corrections"] = [self.cl_mem["image"],
                                              numpy.int8(0), self.cl_mem["dark"],
                                              numpy.int8(0), self.cl_mem["flat"],
                                              numpy.int8(0), self.cl_mem["solidangle"],
                                              numpy.int8(0), self.cl_mem["polarization"],
                                              numpy.int8(0), self.cl_mem["absorption"],
                                              numpy.int8(0), self.cl_mem["mask"],
                                              do_dummy, dummy,
                                              delta_dummy, numpy.float32(1.0),
                                              self.cl_mem["output"]]
        self.cl_kernel_args["corrections2"] = [self.cl_mem["image"],
                                               numpy.int8(0), self.cl_mem["dark"],
                                               numpy.int8(0), self.cl_mem["flat"],
                                               numpy.int8(0), self.cl_mem["solidangle"],
                                               numpy.int8(0), self.cl_mem["polarization"],
                                               numpy.int8(0), self.cl_mem["absorption"],
                                               numpy.int8(0), self.cl_mem["mask"],
                                               do_dummy, dummy,
                                               delta_dummy, numpy.float32(1.0),
                                               self.cl_mem["output"]]
        self.cl_kernel_args["corrections3"] = [self.cl_mem["image"],
                                               self.cl_mem["variance"],
                                               numpy.int8(0), self.cl_mem["dark"],
                                               numpy.int8(0), self.cl_mem["dark_variance"],
                                               numpy.int8(0), self.cl_mem["flat"],
                                               numpy.int8(0), self.cl_mem["solidangle"],
                                               numpy.int8(0), self.cl_mem["polarization"],
                                               numpy.int8(0), self.cl_mem["absorption"],
                                               numpy.int8(0), self.cl_mem["mask"],
                                               do_dummy, dummy,
                                               delta_dummy, numpy.float32(1.0),
                                               self.cl_mem["output"]]
        self.cl_kernel_args["corrections3Poisson"] = [self.cl_mem["image"],
                                                      numpy.int8(0), self.cl_mem["dark"],
                                                      numpy.int8(0), self.cl_mem["flat"],
                                                      numpy.int8(0), self.cl_mem["solidangle"],
                                                      numpy.int8(0), self.cl_mem["polarization"],
                                                      numpy.int8(0), self.cl_mem["absorption"],
                                                      numpy.int8(0), self.cl_mem["mask"],
                                                      do_dummy, dummy,
                                                      delta_dummy, numpy.float32(1.0),
                                                      self.cl_mem["output"]]

    def compile_kernels(self, kernel_files=None, compile_options=None):
        """Call the OpenCL compiler
        
        :param kernel_files: list of path to the kernel 
        (by default use the one declared in the class)
        """
        # concatenate all needed source files into a single openCL module
        kernel_files = kernel_files or self.kernel_files
        compile_options = "-D NIMAGE=%i" % (self.size)
        OpenclProcessing.compile_kernels(self, kernel_files, compile_options)

    def send_buffer(self, data, dest):
        """Send a numpy array to the device
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
            cast_to_float = kernel(self.queue, (self.shape,), None, self.cl_mem["image_raw"], self.cl_mem[dest])
            events += [EventDescription("copy raw %s" % dest,), EventDescription("cast to float", cast_to_float)]
        if self.profile:
            self.events += events
        self.on_device[dest] = data

    def process(self, image,
                dark=None,
                variance=None,
                dark_variance=None,
                normalization_factor=1.0
                ):
        """Perform the pixel-wise operation of the array
        
        :param raw: numpy array with the input image
        
        :param dark: numpy array with the dark-current image
        :param 
        """
        with self.sem:
            if id(image) != id(self.on_device.get("image")):
                self.send_buffer(image, "image")

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

            if self.on_host.get("poissonian"):
                kernel_name = "corrections3Poisson"
                args = self.cl_kernel_args[kernel_name]
                args[1] = do_dark
            elif self.on_host.get("calc_variance"):
                kernel_name = "corrections3"
                args = self.cl_kernel_args[kernel_name]
                args[2] = do_dark
                if self.on_device.get("dark_variance") is not None and do_dark:
                    args[4] = do_dark
            elif self.on_host.get("split_result"):
                kernel_name = "corrections2"
                args = self.cl_kernel_args[kernel_name]
                args[1] = do_dark
            else:
                kernel_name = "corrections"
                args = self.cl_kernel_args[kernel_name]
                args[1] = do_dark
            args[-2] = numpy.float32(normalization_factor)
            kernel = self.program.__getattr__(kernel_name)
            evt = kernel(self.queue, (self.size,), None, *args)
            if kernel_name.startswith("corrections3"):
                dest = numpy.empty(self.on_device.get("image").shape + (3,), dtype=numpy.float32)
            elif kernel_name == "corrections2":
                dest = numpy.empty(self.on_device.get("image").shape + (2,), dtype=numpy.float32)
            else:
                dest = numpy.empty(self.on_device.get("image").shape, dtype=numpy.float32)

            copy_result = pyopencl.enqueue_copy(self.queue, dest, self.cl_mem["output"])
            copy_result.wait()
            if self.profile:
                self.events += [EventDescription("preproc", evt), EventDescription("copy result", copy_result)]
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
            poissonian=False,
            dtype=numpy.float32
            ):
    """Common preprocessing step, implemented using OpenCL. May be inefficient 
    
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
    :param poissonian: set to "True" for assuming the detector is poissonian and variance = raw + dark
    :param dtype: dtype for all processing
    
    All calculation are performed in single precision floating point (32 bits).
    
    NaN are always considered as invalid values
    
    if neither empty nor dummy is provided, empty pixels are 0.
    Empty pixels are always zero in "split_result" mode
    
    Split result:
    -------------
    When set to False, i.e the default, the pixel-wise operation is:
    I = (raw - dark)/(flat \* solidangle \* polarization \* absorption)
    Invalid pixels are set to the dummy or empty value. 
     
    When split_ressult is set to True, each result result is a float2 
    or a float3 (with an additional value for the variance) as such:
    I = [(raw - dark), (variance), (flat \* solidangle \* polarization \* absorption)]
    Empty pixels will have all their 2 or 3 values to 0 (and not to dummy or empty value) 
    
    If poissonian is set to True, the variance is evaluated as (raw + dark)
    """
    if raw.dtype.itemsize > 4:  # use numpy to cast to float32
        raw = numpy.ascontiguousarray(raw, numpy.float32)

    engine = OCL_Preproc(image=raw, dark=dark, flat=flat, solidangle=solidangle,
                         polarization=polarization, absorption=absorption,
                         mask=mask, dummy=dummy, delta_dummy=delta_dummy, empty=empty,
                         split_result=split_result,
                         calc_variance=(variance is not None),
                         poissonian=poissonian,
                         devicetype="all")
    result = engine.process(raw, dark=dark, variance=variance,
                            dark_variance=dark_variance,
                            normalization_factor=normalization_factor)

    if result.dtype != dtype:
        result = numpy.ascontiguousarray(result, dtype)
    return result
