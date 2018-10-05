# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#
#    Copyright (C) 2014-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#                            Giannis Ashiotis
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


__authors__ = ["Jérôme Kieffer", "Giannis Ashiotis"]
__license__ = "MIT"
__date__ = "04/10/2018"
__copyright__ = "2014, ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import gc
import logging
import threading
import numpy
from . import ocl, pyopencl, allocate_cl_buffers, release_cl_buffers
from ..ext.splitBBoxLUT import HistoBBox1d
from . import utils
from ..utils import crc32
if pyopencl:
    mf = pyopencl.mem_flags
else:
    raise ImportError("pyopencl is not installed")

logger = logging.getLogger(__name__)


class OCL_Hist_Pixelsplit(object):
    def __init__(self, pos, bins, image_size, pos0Range=None, pos1Range=None, devicetype="all",
                 padded=False, block_size=32,
                 platformid=None, deviceid=None,
                 checksum=None, profile=False):
        """
        :param lut: 3-tuple of arrays
            data: coefficient of the matrix in a 1D vector of float32 - size of nnz
            indices: Column index position for the data (same size as data)
            indptr: row pointer indicates the start of a given row. len nbin+1
        :param image_size: size of the image (for pre-processing)
        :param devicetype: can be "cpu","gpu","acc" or "all"
        :param platformid: number of the platform as given by clinfo
        :type platformid: int
        :param deviceid: number of the device as given by clinfo
        :type deviceid: int
        :param checksum: pre - calculated checksum to prevent re - calculating it :)
        :param profile: store profiling elements
        """
        self.padded = padded
        self._sem = threading.Semaphore()
        self.pos = pos
        self.bins = bins
        self.pos_size = pos.size
        self.size = image_size
        if self.pos_size != 8 * self.size:
            raise RuntimeError("pos.size != 8 * image_size")
        self.pos0Range = numpy.zeros(1, pyopencl.array.vec.float2)
        self.pos1Range = numpy.zeros(1, pyopencl.array.vec.float2)
        if (pos0Range is not None) and (len(pos0Range) is 2):
            self.pos0Range[0][0] = min(pos0Range)
            self.pos0Range[0][1] = max(pos0Range)
        else:
            self.pos0Range[0][0] = -float("inf")
            self.pos0Range[0][1] = +float("inf")

        if (pos1Range is not None) and (len(pos1Range) is 2):
            self.pos1Range[0][0] = min(pos1Range)
            self.pos1Range[0][1] = max(pos1Range)
        else:
            self.pos1Range[0][0] = -float("inf")
            self.pos1Range[0][1] = +float("inf")

        self.profile = profile
        if not checksum:
            checksum = crc32(self.pos)
        self.on_device = {"pos": checksum, "dark": None, "flat": None, "polarization": None, "solidangle": None}
        self._cl_kernel_args = {}
        self._cl_mem = {}
        self.events = []
        if (platformid is None) and (deviceid is None):
            platformid, deviceid = ocl.select_device(devicetype)
        elif platformid is None:
            platformid = 0
        elif deviceid is None:
            deviceid = 0
        self.platform = ocl.platforms[platformid]
        self.device = self.platform.devices[deviceid]
        self.device_type = self.device.type
        self.BLOCK_SIZE = min(self.device.max_work_group_size, block_size)
        self.workgroup_size = self.BLOCK_SIZE,
        self.wdim_bins = (self.bins * self.BLOCK_SIZE),
        self.wdim_data = (self.size + self.BLOCK_SIZE - 1) & ~(self.BLOCK_SIZE - 1),
        try:
            # self._ctx = pyopencl.Context(devices=[pyopencl.get_platforms()[platformid].get_devices()[deviceid]])
            self._ctx = pyopencl.create_some_context()
            if self.profile:
                self._queue = pyopencl.CommandQueue(self._ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
            else:
                self._queue = pyopencl.CommandQueue(self._ctx)
            self._allocate_buffers()
            self._compile_kernels()
            self._set_kernel_arguments()
        except pyopencl.MemoryError as error:
            raise MemoryError(error)
        ev = pyopencl.enqueue_copy(self._queue, self._cl_mem["pos"], self.pos)
        if self.profile:
            self.events.append(("copy pos data", ev))
        reduction_wg_size = 256
        reduce1 = self._program.reduce1(self._queue, (reduction_wg_size * reduction_wg_size,), (reduction_wg_size,), *self._cl_kernel_args["reduce1"])
        self.events.append(("reduce1", reduce1))
        reduce2 = self._program.reduce2(self._queue, (reduction_wg_size,), (reduction_wg_size,), *self._cl_kernel_args["reduce2"])
        self.events.append(("reduce2", reduce2))

        result = numpy.ndarray(4, dtype=numpy.float32)
        pyopencl.enqueue_copy(self._queue, result, self._cl_mem["minmax"])
        print(result)
        min0 = pos[:, :, 0].min()
        max0 = pos[:, :, 0].max()
        min1 = pos[:, :, 1].min()
        max1 = pos[:, :, 1].max()
        minmax = (min0, max0, min1, max1)

        print(minmax)

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
            ("pos", mf.READ_ONLY, numpy.float32, self.pos_size),
            ("preresult", mf.READ_WRITE, numpy.float32, self.BLOCK_SIZE * 4),
            ("minmax", mf.READ_WRITE, numpy.float32, 4),
            ("outData", mf.READ_WRITE, numpy.float32, self.bins),
            ("outCount", mf.READ_WRITE, numpy.float32, self.bins),
            ("outMerge", mf.WRITE_ONLY, numpy.float32, self.bins),
            ("image_u16", mf.READ_ONLY, numpy.int16, self.size),
            ("image", mf.READ_WRITE, numpy.float32, self.size),
            ("dark", mf.READ_ONLY, numpy.float32, self.size),
            ("flat", mf.READ_ONLY, numpy.float32, self.size),
            ("polarization", mf.READ_ONLY, numpy.float32, self.size),
            ("solidangle", mf.READ_ONLY, numpy.float32, self.size),
        ]

        if self.size < self.BLOCK_SIZE:
            raise RuntimeError("Fatal error in _allocate_buffers. size (%d) must be >= BLOCK_SIZE (%d)\n", self.size, self.BLOCK_SIZE)  # noqa

        self._cl_mem = allocate_cl_buffers(buffers, self.device, self._ctx)

    def _free_buffers(self):
        """
        free all memory allocated on the device
        """
        self._cl_mem = release_cl_buffers(self._cl_mem)

    def _compile_kernels(self, kernel_file=None):
        """
        Call the OpenCL compiler
        :param kernel_file: path tothe
        """
        kernel_file = kernel_file or "ocl_hist_pixelsplit.cl"
        kernel_src = utils.concatenate_cl_kernel([kernel_file])

        template_options = "-D BINS=%i  -D NIMAGE=%i -D WORKGROUP_SIZE=%i -D EPS=%f"
        compile_options = template_options % (self.bins, self.size, self.BLOCK_SIZE, numpy.finfo(numpy.float32).eps)
        logger.info("Compiling file %s with options %s", kernel_file, compile_options)
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
        self._cl_kernel_args["reduce1"] = [self._cl_mem["pos"], numpy.int32(self.pos_size), self._cl_mem["preresult"]]
        self._cl_kernel_args["reduce2"] = [self._cl_mem["preresult"], self._cl_mem["minmax"]]
        self._cl_kernel_args["corrections"] = [self._cl_mem["image"], numpy.int32(0), self._cl_mem["dark"], numpy.int32(0), self._cl_mem["flat"],
                                               numpy.int32(0), self._cl_mem["solidangle"], numpy.int32(0), self._cl_mem["polarization"],
                                               numpy.int32(0), numpy.float32(0), numpy.float32(0)]
        self._cl_kernel_args["integrate1"] = [self._cl_mem["pos"], self._cl_mem["image"], self._cl_mem["minmax"], numpy.int32(0), self.pos0Range[0],
                                              self.pos1Range[0], numpy.int32(0), numpy.float32(0), self._cl_mem["outData"], self._cl_mem["outCount"]]
        self._cl_kernel_args["integrate2"] = [self._cl_mem["outData"], self._cl_mem["outCount"], self._cl_mem["outMerge"]]
        self._cl_kernel_args["memset_out"] = [self._cl_mem[i] for i in ["outData", "outCount", "outMerge"]]
        self._cl_kernel_args["u16_to_float"] = [self._cl_mem[i] for i in ["image_u16", "image"]]
        self._cl_kernel_args["s32_to_float"] = [self._cl_mem[i] for i in ["image", "image"]]

    def integrate(self, data, dummy=None, delta_dummy=None, dark=None, flat=None, solidAngle=None, polarization=None, dark_checksum=None, flat_checksum=None, solidAngle_checksum=None, polarization_checksum=None):
        events = []
        with self._sem:
            if data.dtype == numpy.uint16:
                copy_image = pyopencl.enqueue_copy(self._queue, self._cl_mem["image_u16"], numpy.ascontiguousarray(data))
                cast_u16_to_float = self._program.u16_to_float(self._queue, self.wdim_data, self.workgroup_size, *self._cl_kernel_args["u16_to_float"])
                events += [("copy image", copy_image), ("cast", cast_u16_to_float)]
            elif data.dtype == numpy.int32:
                copy_image = pyopencl.enqueue_copy(self._queue, self._cl_mem["image"], numpy.ascontiguousarray(data))
                cast_s32_to_float = self._program.s32_to_float(self._queue, self.wdim_data, self.workgroup_size, *self._cl_kernel_args["s32_to_float"])
                events += [("copy image", copy_image), ("cast", cast_s32_to_float)]
            else:
                copy_image = pyopencl.enqueue_copy(self._queue, self._cl_mem["image"], numpy.ascontiguousarray(data, dtype=numpy.float32))
                events += [("copy image", copy_image)]
            memset = self._program.memset_out(self._queue, self.wdim_bins, self.workgroup_size, *self._cl_kernel_args["memset_out"])
            events += [("memset", memset)]
            if dummy is not None:
                do_dummy = numpy.int32(1)
                dummy = numpy.float32(dummy)
                if delta_dummy is None:
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
            self._cl_kernel_args["integrate1"][6] = do_dummy
            self._cl_kernel_args["integrate1"][7] = dummy

            if dark is not None:
                do_dark = numpy.int32(1)
                if not dark_checksum:
                    dark_checksum = crc32(dark)
                if dark_checksum != self.on_device["dark"]:
                    ev = pyopencl.enqueue_copy(self._queue, self._cl_mem["dark"], numpy.ascontiguousarray(dark, dtype=numpy.float32))
                    events.append("copy dark", ev)
                    self.on_device["dark"] = dark_checksum
            else:
                do_dark = numpy.int32(0)
            self._cl_kernel_args["corrections"][1] = do_dark
            if flat is not None:
                do_flat = numpy.int32(1)
                if not flat_checksum:
                    flat_checksum = crc32(flat)
                if self.on_device["flat"] != flat_checksum:
                    ev = pyopencl.enqueue_copy(self._queue, self._cl_mem["flat"], numpy.ascontiguousarray(flat, dtype=numpy.float32))
                    events.append("copy flat", ev)
                    self.on_device["flat"] = flat_checksum
            else:
                do_flat = numpy.int32(0)
            self._cl_kernel_args["corrections"][3] = do_flat

            if solidAngle is not None:
                do_solidAngle = numpy.int32(1)
                if not solidAngle_checksum:
                    solidAngle_checksum = crc32(solidAngle)
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
                    polarization_checksum = crc32(polarization)
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
            integrate1 = self._program.integrate1(self._queue, self.wdim_bins, self.workgroup_size, *self._cl_kernel_args["integrate1"])
            events.append(("integrate1", integrate1))
            outMerge = numpy.empty(self.bins, dtype=numpy.float32)
            outData = numpy.empty(self.bins, dtype=numpy.float32)
            outCount = numpy.empty(self.bins, dtype=numpy.float32)
            ev = pyopencl.enqueue_copy(self._queue, outData, self._cl_mem["outData"])
            events.append(("copy D->H outData", ev))
            ev = pyopencl.enqueue_copy(self._queue, outCount, self._cl_mem["outCount"])
            events.append(("copy D->H outCount", ev))
            global_size_integrate2 = (self.bins + self.BLOCK_SIZE - 1) & ~(self.BLOCK_SIZE - 1),
            integrate2 = self._program.integrate2(self._queue, global_size_integrate2, self.workgroup_size, *self._cl_kernel_args["integrate2"])
            events.append(("integrate2", integrate2))
            ev = pyopencl.enqueue_copy(self._queue, outMerge, self._cl_mem["outMerge"])
            events.append(("copy D->H outMerge", ev))
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

        print("_" * 80)
        print("%50s:\t%.3fms" % ("Total execution time", t))
