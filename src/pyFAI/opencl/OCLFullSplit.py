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

"""
Deprecated ... restore or delete !
"""

__authors__ = ["Jérôme Kieffer", "Giannis Ashiotis"]
__license__ = "MIT"
__date__ = "08/04/2024"
__copyright__ = "2014, ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import os
import logging
import threading
import numpy
from . import ocl, pyopencl

if pyopencl:
    mf = pyopencl.mem_flags
else:
    raise ImportError("pyopencl is not installed")
from ..utils import crc32, get_cl_file
logger = logging.getLogger(__name__)


class OCLFullSplit1d(object):

    def __init__(self,
                 pos,
                 bins=100,
                 pos0_range=None,
                 pos1_range=None,
                 mask=None,
                 mask_checksum=None,
                 allow_pos0_neg=False,
                 unit="undefined",
                 workgroup_size=256,
                 devicetype="all",
                 platformid=None,
                 deviceid=None,
                 profile=False):

        self.bins = bins
        self.lut_size = 0
        self.allow_pos0_neg = allow_pos0_neg
        if not (pos.ndim in [3,4] and
                pos.shape[-2] == 4 and
                pos.shape[-1] == 2):
            raise ValueError("Pos array dimensions are wrong")
        self.pos_size = pos.size
        self.size = self.pos_size / 8
        self.pos = numpy.ascontiguousarray(pos.ravel(), dtype=numpy.float32)
        self.pos0_range = numpy.empty(2, dtype=numpy.float32)
        self.pos1_range = numpy.empty(2, dtype=numpy.float32)

        if (pos0_range is not None) and (len(pos0_range) == 2):
            self.pos0_range[0] = min(pos0_range)  # do it on GPU?
            self.pos0_range[1] = max(pos0_range)
            if (not self.allow_pos0_neg) and (self.pos0_range[0] < 0):
                self.pos0_range[0] = 0.0
                if self.pos0_range[1] < 0:
                    print("Warning: Invalid 0-dim range! Using the data derived range instead")
                    self.pos0_range[1] = 0.0
            # self.pos0_range[0] = pos0_range[0]
            # self.pos0_range[1] = pos0_range[1]
        else:
            self.pos0_range[0] = 0.0
            self.pos0_range[1] = 0.0
        if (pos1_range is not None) and (len(pos1_range) == 2):
            self.pos1_range[0] = min(pos1_range)  # do it on GPU?
            self.pos1_range[1] = max(pos1_range)
            # self.pos1_range[0] = pos1_range[0]
            # self.pos1_range[1] = pos1_range[1]
        else:
            self.pos1_range[0] = 0.0
            self.pos1_range[1] = 0.0

        if mask is not None:
            if mask.size != self.size:
                raise RuntimeError("Mask shape does not match size")
            self.check_mask = True
            self.cmask = numpy.ascontiguousarray(mask.ravel(), dtype=numpy.int8)
            if mask_checksum:
                self.mask_checksum = mask_checksum
            else:
                self.mask_checksum = crc32(mask)
        else:
            self.check_mask = False
            self.mask_checksum = None

        self._sem = threading.Semaphore()
        self.profile = profile
        self._cl_kernel_args = {}
        self._cl_mem = {}
        self.events = []
        self.workgroup_size = int(workgroup_size)
        if self.size < self.workgroup_size:
            raise RuntimeError("Fatal error in workgroup size selection. Size (%d) must be >= workgroup size (%d)\n", self.size, self.workgroup_size)
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
            self.workgroup_size = 1
        try:
            self._ctx = pyopencl.Context(devices=[pyopencl.get_platforms()[platformid].get_devices()[deviceid]])
            if self.profile:
                self._queue = pyopencl.CommandQueue(self._ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
            else:
                self._queue = pyopencl.CommandQueue(self._ctx)
            self._compile_kernels()
            self._calc_boundaries()
            self._calc_LUT()
        except pyopencl.MemoryError as error:
            raise MemoryError(error)

    def _compile_kernels(self, kernel_file=None):
        """
        Call the OpenCL compiler
        :param kernel_file: path tothe
        """
        kernel_name = "ocl_lut.cl"
        if kernel_file is None:
            if os.path.isfile(kernel_name):
                kernel_file = os.path.abspath(kernel_name)
            else:
                kernel_file = get_cl_file("pyfai:openCL/" + kernel_name)
        else:
            kernel_file = str(kernel_file)
        kernel_src = open(kernel_file).read()
        compile_options = "-D BINS=%i -D POS_SIZE=%i -D SIZE=%i -D WORKGROUP_SIZE=%i -D EPS=%e" % \
                          (self.bins, self.pos_size, self.size, self.workgroup_size, numpy.finfo(numpy.float32).eps)
        logger.info("Compiling file %s with options %s", kernel_file, compile_options)
        try:
            self._program = pyopencl.Program(self._ctx, kernel_src).build(options=compile_options)
        except pyopencl.MemoryError as error:
            raise MemoryError(error)

    def _calc_boundaries(self):
        """
        comments
        """
        # # # # # # # # Check for memory# # # # # # # #
        size_of_float = numpy.dtype(numpy.float32).itemsize

        ualloc = (self.pos_size * size_of_float)
        ualloc += (self.workgroup_size * 4 * size_of_float)
        ualloc += (4 * size_of_float)
        memory = self.device.memory
        if ualloc >= memory:
            raise MemoryError("Fatal error in _allocate_buffers. Not enough device memory for buffers (%lu requested, %lu available)" % (ualloc, memory))

        try:
            self._cl_mem["pos"] = pyopencl.Buffer(self._ctx, mf.READ_ONLY, size_of_float * self.pos_size)
            self._cl_mem["preresult"] = pyopencl.Buffer(self._ctx, mf.READ_WRITE, size_of_float * 4 * self.workgroup_size)
            self._cl_mem["minmax"] = pyopencl.Buffer(self._ctx, mf.READ_WRITE, size_of_float * 4)
        except pyopencl.MemoryError as error:
            self._free_device_memory()
            raise MemoryError(error)
        with self._sem:
            copy_pos = pyopencl.enqueue_copy(self._queue, self._cl_mem["pos"], self.pos)
            self.profile_add("copy pos", copy_pos)
        self._cl_kernel_args["reduce_minmax_1"] = [self._cl_mem["pos"], self._cl_mem["preresult"]]
        self._cl_kernel_args["reduce_minmax_2"] = [self._cl_mem["preresult"], self._cl_mem["minmax"]]
        with self._sem:
            reduce_minmax_1 = self._program.reduce_minmax_1(self._queue, (self.workgroup_size * self.workgroup_size,), (self.workgroup_size,), *self._cl_kernel_args["reduce_minmax_1"])
            self.profile_add("reduce_minmax_1", reduce_minmax_1)
            reduce_minmax_2 = self._program.reduce_minmax_2(self._queue, (self.workgroup_size,), (self.workgroup_size,), *self._cl_kernel_args["reduce_minmax_2"])
            self.profile_add("reduce_minmax_2", reduce_minmax_2)
        self._cl_mem["preresult"].release()
        self._cl_mem.pop("preresult")

    def _calc_LUT(self):
        """
        first need to call lut_1 and lut_2 to find the size of the LUT and the lut_3 to create it
        """
        # # # # # # # # Check for memory# # # # # # # #
        size_of_float = numpy.dtype(numpy.float32).itemsize
        size_of_int = numpy.dtype(numpy.int32).itemsize

        ualloc = (self.pos_size * size_of_float)  # pos
        ualloc += (4 * size_of_float)  # minmax
        ualloc += (2 * size_of_float) * 2  # pos0_range, pos1_range
        ualloc += (self.bins * size_of_int)  # outMax
        ualloc += (1 * size_of_int)  # lutsize
        ualloc += ((self.bins + 1) * size_of_int)  # idx_ptr
        memory = self.device.memory
        if ualloc >= memory:
            raise MemoryError("Fatal error in _allocate_buffers. Not enough device memory for buffers (%lu requested, %lu available)" % (ualloc, memory))
        try:
            self._cl_mem["outMax"] = pyopencl.Buffer(self._ctx, mf.READ_WRITE, size_of_float * self.bins)
            self._cl_mem["lutsize"] = pyopencl.Buffer(self._ctx, mf.READ_WRITE, size_of_float * 1)
            self._cl_mem["idx_ptr"] = pyopencl.Buffer(self._ctx, mf.READ_WRITE, size_of_float * (self.bins + 1))
        except pyopencl.MemoryError as error:
            self._free_device_memory()
            raise MemoryError(error)
        self._cl_kernel_args["memset_outMax"] = [self._cl_mem["outMax"]]
        self._cl_kernel_args["lut_1"] = [self._cl_mem["pos"], self._cl_mem["minmax"], self.pos0_range.data, self.pos1_range.data, self._cl_mem["outMax"]]
        self._cl_kernel_args["lut_2"] = [self._cl_mem["outMax"], self._cl_mem["idx_ptr"], self._cl_mem["lutsize"]]
        memset_size = int(self.bins + self.workgroup_size - 1) // self.workgroup_size * self.workgroup_size,
        global_size = int(self.size + self.workgroup_size - 1) // self.workgroup_size * self.workgroup_size,
        with self._sem:
            memset_outMax = self._program.memset_outMax(self._queue, memset_size, (self.workgroup_size,), *self._cl_kernel_args["memset_outMax"])
            self.profile_add("memset_outMax", memset_outMax)
            lut_1 = self._program.lut_1(self._queue, global_size, (self.workgroup_size,), *self._cl_kernel_args["lut_1"])
            self.profile_add("lut_1", lut_1)
            lut_2 = self._program.lut_2(self._queue, (1,), (1,), *self._cl_kernel_args["lut_2"])
            self.profile_add("lut_2", lut_2)
            self.lutsize = numpy.ndarray(1, dtype=numpy.int32)
            get_lutsize = pyopencl.enqueue_copy(self._queue, self.lutsize, self._cl_mem["lutsize"])
            self.profile_add("get_lutsize", get_lutsize)
        ualloc += (self.lutsize * size_of_int)  # indices
        ualloc += (self.lutsize * size_of_float)  # data
        if ualloc >= memory:
            raise MemoryError("Fatal error in _allocate_buffers. Not enough device memory for buffers (%lu requested, %lu available)" % (ualloc, memory))
        try:
            self._cl_mem["indices"] = pyopencl.Buffer(self._ctx, mf.READ_WRITE, size_of_int * self.lutsize[0])
            self._cl_mem["data"] = pyopencl.Buffer(self._ctx, mf.READ_WRITE, size_of_float * self.lutsize[0])
        except pyopencl.MemoryError as error:
            self._free_device_memory()
            raise MemoryError(error)
        self._cl_kernel_args["lut_3"] = [self._cl_mem["pos"], self._cl_mem["minmax"], self.pos0_range.data, self.pos1_range.data, self._cl_mem["outMax"], self._cl_mem["idx_ptr"], self._cl_mem["indices"], self._cl_mem["data"]]
        with self._sem:
            memset_outMax = self._program.memset_outMax(self._queue, memset_size, (self.workgroup_size,), *self._cl_kernel_args["memset_outMax"])
            self.profile_add("memset_outMax", memset_outMax)
            lut_3 = self._program.lut_3(self._queue, global_size, (self.workgroup_size,), *self._cl_kernel_args["lut_3"])
            self.profile_add("lut_3", lut_3)
        self._cl_mem["pos"].release()
        self._cl_mem.pop("pos")
        self._cl_mem["minmax"].release()
        self._cl_mem.pop("minmax")
        self._cl_mem["outMax"].release()
        self._cl_mem.pop("outMax")
        self._cl_mem["lutsize"].release()
        self._cl_mem.pop("lutsize")

    def _free_device_memory(self):
        """
        free all memory allocated on the device
        """
        for buffer_name in list(self._cl_mem.keys())[:]:
            buf = self._cl_mem.pop[buffer_name]
            if buf is not None:
                try:
                    buf.release()
                except pyopencl.LogicError:
                    logger.error("Error while freeing buffer %s", buffer_name)

    def get_platform(self):
        pass

    def get_queue(self):
        pass
