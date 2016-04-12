# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/pyFAI
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
__date__ = "29/01/2016"
__copyright__ = "2012, ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import gc
import logging
import threading
import numpy
from .opencl import ocl, pyopencl, allocate_cl_buffers, release_cl_buffers
from .ext.splitBBoxLUT import HistoBBox1d
from .utils import concatenate_cl_kernel, calc_checksum
if pyopencl:
    mf = pyopencl.mem_flags
else:
    raise ImportError("pyopencl is not installed")
logger = logging.getLogger("pyFAI.ocl_azim_lut")


class OCL_LUT_Integrator(object):
    BLOCK_SIZE = 16
    def __init__(self, lut, image_size, devicetype="all",
                 platformid=None, deviceid=None,
                 checksum=None, profile=False,
                 empty=None):
        """
        @param lut: array of int32 - float32 with shape (nbins, lut_size) with indexes and coefficients
        @param image_size: Expected image size: image.shape.prod()
        @param devicetype: can be "cpu","gpu","acc" or "all"
        @param platformid: number of the platform as given by clinfo
        @type platformid: int
        @param deviceid: number of the device as given by clinfo
        @type deviceid: int
        @param checksum: pre - calculated checksum to prevent re - calculating it :)
        @param profile: store profiling elements
        @param empty: value to be assigned to bins without contribution from any pixel
        """
        self._sem = threading.Semaphore()
        self._lut = lut
        self.nbytes = lut.nbytes
        self.bins, self.lut_size = lut.shape
        self.size = image_size
        self.profile = profile
        self.empty = empty or 0.0

        if not checksum:
            checksum = calc_checksum(self._lut)
        self.on_device = {"lut":checksum, "dark":None, "flat":None, "polarization":None, "solidangle":None}
        self._cl_kernel_args = {}
        self._cl_mem = {}
        self.events = []
        if (platformid is None) and (deviceid is None):
            res = ocl.select_device(devicetype)
            if res:
                platformid, deviceid = res
            else:
                logger.warning("No such devicetype %s" % devicetype)
                platformid, deviceid = ocl.select_device()
        elif platformid is None:
            platformid = 0
        elif deviceid is None:
            deviceid = 0
        self.platform = ocl.platforms[platformid]
        self.device = self.platform.devices[deviceid]
        self.device_type = self.device.type
        self.BLOCK_SIZE = min(self.BLOCK_SIZE, self.device.max_work_group_size)
        self.workgroup_size = self.BLOCK_SIZE,  # Note this is a tuple
        self.wdim_bins = (self.bins + self.BLOCK_SIZE - 1) & ~(self.BLOCK_SIZE - 1),
        self.wdim_data = (self.size + self.BLOCK_SIZE - 1) & ~(self.BLOCK_SIZE - 1),

        try:
            self.ctx = pyopencl.Context(devices=[pyopencl.get_platforms()[platformid].get_devices()[deviceid]])
            if self.profile:
                self._queue = pyopencl.CommandQueue(self.ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
            else:
                self._queue = pyopencl.CommandQueue(self.ctx)
            self._allocate_buffers()
            self._compile_kernels()
            self._set_kernel_arguments()
        except (pyopencl.MemoryError, pyopencl.LogicError) as error:
            raise MemoryError(error)
        if self.device_type == "CPU":
            ev = pyopencl.enqueue_copy(self._queue, self._cl_mem["lut"], lut)
        else:
            ev = pyopencl.enqueue_copy(self._queue, self._cl_mem["lut"], lut.T.copy())
        if self.profile: self.events.append(("copy LUT", ev))

    def __del__(self):
        """
        Destructor: release all buffers
        """
        self._free_kernels()
        self._free_buffers()
        self._queue = None
        self.ctx = None
        gc.collect()

    def __copy__(self):
        """Shallow copy of the object
        
        :return: copy of the object
        """
        return self.__class__(self._lut, self.size, 
                              block_size=self.BLOCK_SIZE,
                              platformid=self.platform.id,
                              deviceid=self.device.id,
                              checksum=self.on_device.get("lut"),
                              profile=self.profile, empty=self.empty)

    def __deepcopy__(self, memo=None):
        """deep copy of the object
        
        :return: deepcopy of the object
        """
        if memo is None:
            memo = {}
        new_lut = self._lut.copy()
        memo[id(self._lut)] = new_lut
        new_obj = self.__class__(new_lut, self.size, 
                                 block_size=self.BLOCK_SIZE,
                                 platformid=self.platform.id,
                                 deviceid=self.device.id,
                                 checksum=self.on_device.get("lut"),
                                 profile=self.profile, empty=self.empty)
        memo[id(self)] = new_obj
        return new_obj

    def _allocate_buffers(self):
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
        buffers = [
            ("lut", mf.READ_WRITE, [("bins", numpy.float32), ("lut_size", numpy.int32)], self.bins * self.lut_size),  # noqa
            ("outData", mf.WRITE_ONLY, numpy.float32, self.bins),
            ("outCount", mf.WRITE_ONLY, numpy.float32, self.bins),
            ("outMerge", mf.WRITE_ONLY, numpy.float32, self.bins),
            ("image_raw", mf.READ_ONLY, numpy.float32, self.size),
            ("image", mf.READ_WRITE, numpy.float32, self.size),
            ("dark", mf.READ_ONLY, numpy.float32, self.size),
            ("flat", mf.READ_ONLY, numpy.float32, self.size),
            ("polarization", mf.READ_ONLY, numpy.float32, self.size),
            ("solidangle", mf.READ_ONLY, numpy.float32, self.size),
        ]

        if self.size < self.BLOCK_SIZE:
            raise RuntimeError("Fatal error in _allocate_buffers. size (%d) must be >= BLOCK_SIZE (%d)\n", self.size, self.BLOCK_SIZE)

        self._cl_mem = allocate_cl_buffers(buffers, self.device, self.ctx)

    def _free_buffers(self):
        """
        free all memory allocated on the device
        """
        self._cl_mem = release_cl_buffers(self._cl_mem)

    def _compile_kernels(self, kernel_file=None):
        """
        Call the OpenCL compiler
        @param kernel_file: path to the kernel (by default use the one in the src directory)
        """
        # concatenate all needed source files into a single openCL module
        kernel_file = kernel_file or "ocl_azim_LUT.cl"
        kernel_src = concatenate_cl_kernel(["preprocess.cl", kernel_file])

        compile_options = "-D NBINS=%i  -D NIMAGE=%i -D NLUT=%i -D ON_CPU=%i" % \
                (self.bins, self.size, self.lut_size, int(self.device_type == "CPU"))
        logger.info("Compiling file %s with options %s" % (kernel_file, compile_options))
        try:
            self._program = pyopencl.Program(self.ctx, kernel_src).build(options=compile_options)
        except (pyopencl.MemoryError, pyopencl.LogicError) as error:
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
                                             numpy.int32(0), numpy.float32(0.0), numpy.float32(0.0), numpy.float32(0.0)]
        self._cl_kernel_args["lut_integrate"] = [self._cl_mem["image"], self._cl_mem["lut"], numpy.int32(0), numpy.float32(0), \
                                                self._cl_mem["outData"], self._cl_mem["outCount"], self._cl_mem["outMerge"]]
        self._cl_kernel_args["memset_out"] = [self._cl_mem[i] for i in ["outData", "outCount", "outMerge"]]
        self._cl_kernel_args["u8_to_float"] = [self._cl_mem[i] for i in ["image_raw", "image"]]
        self._cl_kernel_args["s8_to_float"] = [self._cl_mem[i] for i in ["image_raw", "image"]]
        self._cl_kernel_args["u16_to_float"] = [self._cl_mem[i] for i in ["image_raw", "image"]]
        self._cl_kernel_args["s16_to_float"] = [self._cl_mem[i] for i in ["image_raw", "image"]]
        self._cl_kernel_args["u32_to_float"] = [self._cl_mem[i] for i in ["image_raw", "image"]]
        self._cl_kernel_args["s32_to_float"] = [self._cl_mem[i] for i in ["image_raw", "image"]]

    def integrate(self, data, dummy=None, delta_dummy=None, dark=None, flat=None, solidAngle=None, polarization=None,
                            dark_checksum=None, flat_checksum=None, solidAngle_checksum=None, polarization_checksum=None,
                            preprocess_only=False, safe=True, normalization_factor=1.0):
        """
        Before performing azimuthal integration, the preprocessing is :

        data = (data - dark) / (flat*solidAngle*polarization)

        Integration is performed using the CSR representation of the look-up table

        @param dark: array of same shape as data for pre-processing
        @param flat: array of same shape as data for pre-processing
        @param solidAngle: array of same shape as data for pre-processing
        @param polarization: array of same shape as data for pre-processing
        @param dark_checksum: CRC32 checksum of the given array
        @param flat_checksum: CRC32 checksum of the given array
        @param solidAngle_checksum: CRC32 checksum of the given array
        @param polarization_checksum: CRC32 checksum of the given array
        @param safe: if True (default) compares arrays on GPU according to their checksum, unless, use the buffer location is used
        @param normalization_factor: divide raw signal by this value
        @param preprocess_only: return the dark subtracted; flat field & solidAngle & polarization corrected image, else
        @return averaged data, weighted histogram, unweighted histogram
        """
        events = []
        with self._sem:
            if data.dtype == numpy.uint8:
                copy_image = pyopencl.enqueue_copy(self._queue, self._cl_mem["image_raw"], numpy.ascontiguousarray(data))
                cast_to_float = self._program.u8_to_float(self._queue, self.wdim_data, self.workgroup_size, *self._cl_kernel_args["u8_to_float"])
                events += [("copy image", copy_image), ("cast", cast_to_float)]
            elif data.dtype == numpy.int8:
                copy_image = pyopencl.enqueue_copy(self._queue, self._cl_mem["image_raw"], numpy.ascontiguousarray(data))
                cast_to_float = self._program.s8_to_float(self._queue, self.wdim_data, self.workgroup_size, *self._cl_kernel_args["s8_to_float"])
                events += [("copy image", copy_image), ("cast", cast_to_float)]
            elif data.dtype == numpy.uint16:
                copy_image = pyopencl.enqueue_copy(self._queue, self._cl_mem["image_raw"], numpy.ascontiguousarray(data))
                cast_to_float = self._program.u16_to_float(self._queue, self.wdim_data, self.workgroup_size, *self._cl_kernel_args["u16_to_float"])
                events += [("copy image", copy_image), ("cast", cast_to_float)]
            elif data.dtype == numpy.int16:
                copy_image = pyopencl.enqueue_copy(self._queue, self._cl_mem["image_raw"], numpy.ascontiguousarray(data))
                cast_to_float = self._program.s16_to_float(self._queue, self.wdim_data, self.workgroup_size, *self._cl_kernel_args["s16_to_float"])
                events += [("copy image", copy_image), ("cast", cast_to_float)]
            elif data.dtype == numpy.uint32:
                copy_image = pyopencl.enqueue_copy(self._queue, self._cl_mem["image_raw"], numpy.ascontiguousarray(data))
                cast_to_float = self._program.u32_to_float(self._queue, self.wdim_data, self.workgroup_size, *self._cl_kernel_args["u32_to_float"])
                events += [("copy image", copy_image), ("cast", cast_to_float)]
            elif data.dtype == numpy.int32:
                copy_image = pyopencl.enqueue_copy(self._queue, self._cl_mem["image_raw"], numpy.ascontiguousarray(data))
                cast_to_float = self._program.s32_to_float(self._queue, self.wdim_data, self.workgroup_size, *self._cl_kernel_args["s32_to_float"])
                events += [("copy image", copy_image), ("cast", cast_to_float)]
            else:
                copy_image = pyopencl.enqueue_copy(self._queue, self._cl_mem["image"], numpy.ascontiguousarray(data, dtype=numpy.float32))
                events += [("copy image", copy_image)]
            memset = self._program.memset_out(self._queue, self.wdim_bins, self.workgroup_size, *self._cl_kernel_args["memset_out"])
            events.append(("memset", memset))

            if dummy is not None:
                do_dummy = numpy.int32(1)
                dummy = numpy.float32(dummy)
                if delta_dummy == None:
                    delta_dummy = numpy.float32(0.0)
                else:
                    delta_dummy = numpy.float32(abs(delta_dummy))
            else:
                do_dummy = numpy.int32(0)
                dummy = numpy.float32(self.empty)
                delta_dummy = numpy.float32(0.0)
            self._cl_kernel_args["corrections"][9] = do_dummy
            self._cl_kernel_args["corrections"][10] = dummy
            self._cl_kernel_args["corrections"][11] = delta_dummy
            self._cl_kernel_args["corrections"][12] = numpy.float32(normalization_factor)
            self._cl_kernel_args["lut_integrate"][2] = do_dummy
            self._cl_kernel_args["lut_integrate"][3] = dummy

            if dark is not None:
                do_dark = numpy.int32(1)
                if not dark_checksum:
                    dark_checksum = calc_checksum(dark, safe)
                if dark_checksum != self.on_device["dark"]:
                    ev = pyopencl.enqueue_copy(self._queue, self._cl_mem["dark"], numpy.ascontiguousarray(dark, dtype=numpy.float32))
                    events.append(("copy dark", ev))
                    self.on_device["dark"] = dark_checksum
            else:
                do_dark = numpy.int32(0)
            self._cl_kernel_args["corrections"][1] = do_dark
            if flat is not None:
                do_flat = numpy.int32(1)
                if not flat_checksum:
                    flat_checksum = calc_checksum(flat, safe)
                if self.on_device["flat"] != flat_checksum:
                    ev = pyopencl.enqueue_copy(self._queue, self._cl_mem["flat"], numpy.ascontiguousarray(flat, dtype=numpy.float32))
                    events.append(("copy flat", ev))
                    self.on_device["flat"] = flat_checksum
            else:
                do_flat = numpy.int32(0)
            self._cl_kernel_args["corrections"][3] = do_flat

            if solidAngle is not None:
                do_solidAngle = numpy.int32(1)
                if not solidAngle_checksum:
                    solidAngle_checksum = calc_checksum(solidAngle, safe)
                if solidAngle_checksum != self.on_device["solidangle"]:
                    ev = pyopencl.enqueue_copy(self._queue, self._cl_mem["solidangle"], numpy.ascontiguousarray(solidAngle, dtype=numpy.float32))
                    events.append(("copy solidangle", ev))
                    self.on_device["solidangle"] = solidAngle_checksum
            else:
                do_solidAngle = numpy.int32(0)
            self._cl_kernel_args["corrections"][5] = do_solidAngle

            if polarization is not None:
                do_polarization = numpy.int32(1)
                if not polarization_checksum:
                    polarization_checksum = calc_checksum(polarization, safe)
                if polarization_checksum != self.on_device["polarization"]:
                    ev = pyopencl.enqueue_copy(self._queue, self._cl_mem["polarization"], numpy.ascontiguousarray(polarization, dtype=numpy.float32))
                    events.append(("copy polarization", ev))
                    self.on_device["polarization"] = polarization_checksum
            else:
                do_polarization = numpy.int32(0)
            self._cl_kernel_args["corrections"][7] = do_polarization
            copy_image.wait()
            if do_dummy + do_polarization + do_solidAngle + do_flat + do_dark > 0:
                ev = self._program.corrections(self._queue, self.wdim_data, self.workgroup_size, *self._cl_kernel_args["corrections"])
                events.append(("corrections", ev))
            if preprocess_only:
                image = numpy.empty(data.shape, dtype=numpy.float32)
                ev = pyopencl.enqueue_copy(self._queue, image, self._cl_mem["image"])
                events.append(("copy D->H image", ev))
                ev.wait()
                return image
            integrate = self._program.lut_integrate(self._queue, self.wdim_bins, self.workgroup_size, *self._cl_kernel_args["lut_integrate"])
            events.append(("integrate", integrate))
            outMerge = numpy.empty(self.bins, dtype=numpy.float32)
            outData = numpy.empty(self.bins, dtype=numpy.float32)
            outCount = numpy.empty(self.bins, dtype=numpy.float32)
            ev = pyopencl.enqueue_copy(self._queue, outMerge, self._cl_mem["outMerge"])
            events.append(("copy D->H outMerge", ev))
            ev = pyopencl.enqueue_copy(self._queue, outData, self._cl_mem["outData"])
            events.append(("copy D->H outData", ev))
            ev = pyopencl.enqueue_copy(self._queue, outCount, self._cl_mem["outCount"])
            events.append(("copy D->H outCount", ev))
            ev.wait()
        if self.profile:
            self.events += events
        return outMerge, outData, outCount

    def log_profile(self):
        """
        If we are in profiling mode, prints out all timing for every single OpenCL call
        """
        t = 0.0
        if self.profile:
            for e in self.events:
                if "__len__" in dir(e) and len(e) >= 2:
                    et = 1e-6 * (e[1].profile.end - e[1].profile.start)
                    print("%50s:\t%.3fms" % (e[0], et))
                    t += et

        print("_"*80)
        print("%50s:\t%.3fms" % ("Total execution time", t))
