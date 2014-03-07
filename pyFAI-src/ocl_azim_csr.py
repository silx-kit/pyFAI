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

__author__ = "Jerome Kieffer"
__license__ = "GPLv3"
__date__ = "18/10/2012"
__copyright__ = "2012, ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import os, gc, logging
import threading
import hashlib
import numpy
from .opencl import ocl, pyopencl
from .splitBBoxLUT import HistoBBox1d
from .utils import get_cl_file
if pyopencl:
    mf = pyopencl.mem_flags
else:
    raise ImportError("pyopencl is not installed")
try:
    from .fastcrc import crc32
except:
    from zlib import crc32
logger = logging.getLogger("pyFAI.ocl_azim_lut")

class OCL_LUT_Integrator(object):
    def __init__(self, lut, image_size, devicetype="all", platformid=None, deviceid=None, checksum=None):
        """
        @param lut: array of int32 - float32 with shape (nbins, lut_size) with indexes and coefficients
        @param checksum: pre - calculated checksum to prevent re - calculating it :)
        """
        self.BLOCK_SIZE = 16
        self._sem = threading.Semaphore()
        self._lut = lut
        self.bins, self.lut_size = lut.shape
        self.size = image_size
        if not checksum:
            checksum = crc32(self._lut)
        self.on_device = {"lut":checksum, "dark":None, "flat":None, "polarization":None, "solidangle":None}
        self._cl_kernel_args = {}
        self._cl_mem = {}

        if (platformid is None) and (deviceid is None):
            platformid, deviceid = ocl.select_device(devicetype)
        elif platformid is None:
            platformid = 0
        elif deviceid is None:
            deviceid = 0
        self.platform = ocl.platforms[platformid]
        self.device = self.platform.devices[deviceid]
        self.device_type = self.device.type
        if (self.device_type == "CPU") and (self.platform.vendor == "Apple"):
            logger.warning("This is a workaround for Apple's OpenCL on CPU: enforce BLOCK_SIZE=1")
            self.BLOCK_SIZE = 1
        self.workgroup_size = self.BLOCK_SIZE,
        self.wdim_bins = (self.bins + self.BLOCK_SIZE - 1) & ~(self.BLOCK_SIZE - 1),
        self.wdim_data = (self.size + self.BLOCK_SIZE - 1) & ~(self.BLOCK_SIZE - 1),
        try:
            self._ctx = pyopencl.Context(devices=[pyopencl.get_platforms()[platformid].get_devices()[deviceid]])
            self._queue = pyopencl.CommandQueue(self._ctx)
            self._allocate_buffers()
            self._compile_kernels()
            self._set_kernel_arguments()
        except pyopencl.MemoryError as error:
            raise MemoryError(error)
        if self.device_type == "CPU":
            pyopencl.enqueue_copy(self._queue, self._cl_mem["lut"], lut)
        else:
            pyopencl.enqueue_copy(self._queue, self._cl_mem["lut"], lut.T.copy())

    def __del__(self):
        """
        Destructor: release all buffers
        """
        self._free_kernels()
        self._free_buffers()
        self._queue = None
        self._ctx = None
        gc.collect()

    def _allocate_buffers(self):
        """
        Allocate OpenCL buffers required for a specific configuration

        Note that an OpenCL context also requires some memory, as well as Event and other OpenCL functionalities which cannot and
        are not taken into account here.
        The memory required by a context varies depending on the device. Typical for GTX580 is 65Mb but for a 9300m is ~15Mb
        In addition, a GPU will always have at least 3-5Mb of memory in use.
        Unfortunately, OpenCL does NOT have a built-in way to check the actual free memory on a device, only the total memory.
        """
        if self.size < self.BLOCK_SIZE:
            raise RuntimeError("Fatal error in _allocate_buffers. size (%d) must be >= BLOCK_SIZE (%d)\n", self.size, self.BLOCK_SIZE)
        size_of_float = numpy.dtype(numpy.float32).itemsize
        size_of_short = numpy.dtype(numpy.int16).itemsize
        size_of_int = numpy.dtype(numpy.int32).itemsize
        size_of_long = numpy.dtype(numpy.int64).itemsize

        ualloc = (self.size * size_of_float) * 5
        ualloc += (self.bins * self.lut_size * (size_of_float + size_of_int))
        ualloc += (self.bins * size_of_float) * 3
        memory = self.device.memory
        logger.info("%.3fMB are needed on device which has %.3fMB" % (ualloc / 1.0e6, memory / 1.0e6))
        if ualloc >= memory:
            raise MemoryError("Fatal error in _allocate_buffers. Not enough device memory for buffers (%lu requested, %lu available)" % (ualloc, memory))
        # now actually allocate:
        try:
            self._cl_mem["lut"] = pyopencl.Buffer(self._ctx, mf.READ_WRITE, (size_of_float + size_of_int) * self.bins * self.lut_size)
            self._cl_mem["outData"] = pyopencl.Buffer(self._ctx, mf.WRITE_ONLY, size_of_float * self.bins)
            self._cl_mem["outCount"] = pyopencl.Buffer(self._ctx, mf.WRITE_ONLY, size_of_float * self.bins)
            self._cl_mem["outMerge"] = pyopencl.Buffer(self._ctx, mf.WRITE_ONLY, size_of_float * self.bins)
            self._cl_mem["image_u16"] = pyopencl.Buffer(self._ctx, mf.READ_ONLY, size=size_of_short * self.size)
            self._cl_mem["image"] = pyopencl.Buffer(self._ctx, mf.READ_WRITE, size=size_of_float * self.size)
            self._cl_mem["dark"] = pyopencl.Buffer(self._ctx, mf.READ_ONLY, size=size_of_float * self.size)
            self._cl_mem["flat"] = pyopencl.Buffer(self._ctx, mf.READ_ONLY, size=size_of_float * self.size)
            self._cl_mem["polarization"] = pyopencl.Buffer(self._ctx, mf.READ_ONLY, size=size_of_float * self.size)
            self._cl_mem["solidangle"] = pyopencl.Buffer(self._ctx, mf.READ_ONLY, size=size_of_float * self.size)
        except pyopencl.MemoryError as error:
            self._free_buffers()
            raise MemoryError(error)

    def _free_buffers(self):
        """
        free all memory allocated on the device
        """
        for buffer_name in self._cl_mem:
            if self._cl_mem[buffer_name] is not None:
                try:
                    self._cl_mem[buffer_name].release()
                    self._cl_mem[buffer_name] = None
                except pyopencl.LogicError:
                    logger.error("Error while freeing buffer %s" % buffer_name)



    def _compile_kernels(self, kernel_file=None):
        """
        Call the OpenCL compiler
        @param kernel_file: path tothe
        """
        kernel_name = "ocl_azim_LUT.cl"
        if kernel_file is None:
            if os.path.isfile(kernel_name):
                kernel_file = os.path.abspath(kernel_name)
            else:
                kernel_file = get_cl_file(kernel_name)
        else:
            kernel_file = str(kernel_file)
        with open(kernel_file, "r") as kernelFile:
            kernel_src = kernelFile.read()

        compile_options = "-D NBINS=%i  -D NIMAGE=%i -D NLUT=%i -D ON_CPU=%i" % \
                (self.bins, self.size, self.lut_size, int(self.device_type == "CPU"))
        logger.info("Compiling file %s with options %s" % (kernel_file, compile_options))
        try:
            self._program = pyopencl.Program(self._ctx, kernel_src).build(options=compile_options)
        except pyopencl.MemoryError as error:
            raise MemoryError(error)

    def _free_kernels(self):
        """
        free all kernels
        """
        for kernel in self._cl_kernel_args:
            self._cl_kernel_args[kernel] = []
        self._program = None

    def _set_kernel_arguments(self):
        """Tie arguments of OpenCL kernel-functions to the actual kernels

        set_kernel_arguments() is a private method, called by configure().
        It uses the dictionary _cl_kernel_args.
        Note that by default, since TthRange is disabled, the integration kernels have tth_min_max tied to the tthRange argument slot.
        When setRange is called it replaces that argument with tthRange low and upper bounds. When unsetRange is called, the argument slot
        is reset to tth_min_max.
        """
        self._cl_kernel_args["corrections"] = [self._cl_mem["image"], numpy.int32(0), self._cl_mem["dark"], numpy.int32(0), self._cl_mem["flat"], \
                                             numpy.int32(0), self._cl_mem["solidangle"], numpy.int32(0), self._cl_mem["polarization"], \
                                             numpy.int32(0), numpy.float32(0), numpy.float32(0)]
        self._cl_kernel_args["lut_integrate"] = [self._cl_mem["image"], self._cl_mem["lut"], numpy.int32(0), numpy.float32(0), \
                                                self._cl_mem["outData"], self._cl_mem["outCount"], self._cl_mem["outMerge"]]
        self._cl_kernel_args["memset_out"] = [self._cl_mem[i] for i in ["outData", "outCount", "outMerge"]]
        self._cl_kernel_args["u16_to_float"] = [self._cl_mem[i] for i in ["image_u16", "image"]]
        self._cl_kernel_args["s32_to_float"] = [self._cl_mem[i] for i in ["image", "image"]]

    def integrate(self, data, dummy=None, delta_dummy=None, dark=None, flat=None, solidAngle=None, polarization=None,
                            dark_checksum=None, flat_checksum=None, solidAngle_checksum=None, polarization_checksum=None):
        with self._sem:
            if data.dtype == numpy.uint16:
                copy_image = pyopencl.enqueue_copy(self._queue, self._cl_mem["image_u16"], numpy.ascontiguousarray(data))
                cast_u16_to_float = self._program.u16_to_float(self._queue, self.wdim_data, self.workgroup_size, *self._cl_kernel_args["u16_to_float"])
            elif data.dtype == numpy.int32:
                copy_image = pyopencl.enqueue_copy(self._queue, self._cl_mem["image"], numpy.ascontiguousarray(data))
                cast_s32_to_float = self._program.s32_to_float(self._queue, self.wdim_data, self.workgroup_size, *self._cl_kernel_args["s32_to_float"])
            else:
                copy_image = pyopencl.enqueue_copy(self._queue, self._cl_mem["image"], numpy.ascontiguousarray(data, dtype=numpy.float32))
            memset = self._program.memset_out(self._queue, self.wdim_bins, self.workgroup_size, *self._cl_kernel_args["memset_out"])
            if dummy is not None:
                do_dummy = numpy.int32(1)
                dummy = numpy.float32(dummy)
                if delta_dummy == None:
                    delta_dummy = numpy.float32(0)
                else:
                    delta_dummy = numpy.float32(abs(delta_dummy))
            else:
                do_dummy = numpy.int32(0)
                dummy = numpy.float32(0)
                delta_dummy = numpy.float32(0)
            self._cl_kernel_args["corrections"][9] = do_dummy
            self._cl_kernel_args["corrections"][10] = dummy
            self._cl_kernel_args["corrections"][11] = delta_dummy
            self._cl_kernel_args["lut_integrate"][2] = do_dummy
            self._cl_kernel_args["lut_integrate"][3] = dummy

            if dark is not None:
                do_dark = numpy.int32(1)
                if not dark_checksum:
                    dark_checksum = crc32(dark)
                if dark_checksum != self.on_device["dark"]:
                    pyopencl.enqueue_copy(self._queue, self._cl_mem["dark"], numpy.ascontiguousarray(dark, dtype=numpy.float32))
                    self.on_device["dark"] = dark_checksum
            else:
                do_dark = numpy.int32(0)
            self._cl_kernel_args["corrections"][1] = do_dark
            if flat is not None:
                do_flat = numpy.int32(1)
                if not flat_checksum:
                    flat_checksum = crc32(flat)
                if self.on_device["flat"] != flat_checksum:
                    pyopencl.enqueue_copy(self._queue, self._cl_mem["flat"], numpy.ascontiguousarray(flat, dtype=numpy.float32))
                    self.on_device["flat"] = flat_checksum
            else:
                do_flat = numpy.int32(0)
            self._cl_kernel_args["corrections"][3] = do_flat

            if solidAngle is not None:
                do_solidAngle = numpy.int32(1)
                if not solidAngle_checksum:
                    solidAngle_checksum = crc32(solidAngle)
                if solidAngle_checksum != self.on_device["solidangle"]:
                    pyopencl.enqueue_copy(self._queue, self._cl_mem["solidangle"], numpy.ascontiguousarray(solidAngle, dtype=numpy.float32))
                self.on_device["solidangle"] = solidAngle_checksum
            else:
                do_solidAngle = numpy.int32(0)
            self._cl_kernel_args["corrections"][5] = do_solidAngle

            if polarization is not None:
                do_polarization = numpy.int32(1)
                if not polarization_checksum:
                    polarization_checksum = crc32(polarization)
                if polarization_checksum != self.on_device["polarization"]:
                    pyopencl.enqueue_copy(self._queue, self._cl_mem["polarization"], numpy.ascontiguousarray(polarization, dtype=numpy.float32))
                    self.on_device["polarization"] = polarization_checksum
            else:
                do_polarization = numpy.int32(0)
            self._cl_kernel_args["corrections"][7] = do_polarization
            copy_image.wait()
            if do_dummy + do_polarization + do_solidAngle + do_flat + do_dark > 0:
                self._program.corrections(self._queue, self.wdim_data, self.workgroup_size, *self._cl_kernel_args["corrections"]).wait()
            memset.wait()
            integrate = self._program.lut_integrate(self._queue, self.wdim_bins, self.workgroup_size, *self._cl_kernel_args["lut_integrate"])
            outMerge = numpy.zeros(self.bins, dtype=numpy.float32)
            outData = numpy.zeros(self.bins, dtype=numpy.float32)
            outCount = numpy.zeros(self.bins, dtype=numpy.float32)
            integrate.wait()
            pyopencl.enqueue_copy(self._queue, outMerge, self._cl_mem["outMerge"]).wait()
            pyopencl.enqueue_copy(self._queue, outData, self._cl_mem["outData"]).wait()
            pyopencl.enqueue_copy(self._queue, outCount, self._cl_mem["outCount"]).wait()
        return outMerge, outData, outCount
