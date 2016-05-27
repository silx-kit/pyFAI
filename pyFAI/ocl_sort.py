# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) 2015 European Synchrotron Radiation Facility, Grenoble, France
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
Module for 2D sort based on OpenCL for median filtering and Bragg/amorphous 
separation on GPU.

"""

from __future__ import absolute_import, print_function, division
__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__date__ = "23/09/2015"
__copyright__ = "2015, ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import os
import logging
import threading
import numpy
import gc
from .utils import concatenate_cl_kernel
from .opencl import ocl, pyopencl, release_cl_buffers
if pyopencl:
    mf = pyopencl.mem_flags
    import pyopencl.array
else:
    raise ImportError("pyopencl is not installed")
logger = logging.getLogger("pyFAI.ocl_sort")


class Separator(object):
    DUMMY = 1234.5678
    """
    Attempt to implements sort and median filter in  pyopencl
    """
    def __init__(self, npt_height=512, npt_width=1024, ctx=None, max_workgroup_size=None, profile=False):
        """
        @param ctx: context
        @param max_workgroup_size: 1 on macOSX on CPU
        @param profile: turn on profiling
        """
        self._sem = threading.Semaphore()
        self.npt_width = npt_width
        self.npt_height = npt_height
        self.ctx = ctx if ctx else ocl.create_context()

        if max_workgroup_size:
            self.max_workgroup_size = max_workgroup_size
        else:
            self.max_workgroup_size = self.ctx.devices[0].max_work_group_size
        if profile:
            self._queue = pyopencl.CommandQueue(self.ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
            self.profile = True
        else:
            self._queue = pyopencl.CommandQueue(self.ctx)
            self.profile = False
        # Those are pointer to memory on the GPU (or None if uninitialized
        self._cl_mem = {}
        self._cl_kernel_args = {}
        self.events = []  # list with all events for profiling
        self._allocate_buffers()
        self._compile_kernels()
        self._set_kernel_arguments()

    def __del__(self):
        """
        Destructor: release all buffers
        """
        self._free_kernels()
        self._free_buffers()
        self._queue = None
        self.ctx = None
        gc.collect()

    def __repr__(self):
        lst = ["OpenCL implementation of sort/median_filter"]
        return os.linesep.join(lst)

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
            ("input_data", numpy.float32, (self.npt_height, self.npt_width)),
            ("vector_vertical", numpy.float32, (self.npt_width,)),
            ("vector_horizontal", numpy.float32, (self.npt_height,))
        ]
        self._cl_mem = {}
        for name, dtype, shape in buffers:
            self._cl_mem[name] = pyopencl.array.Array(self._queue, shape=shape, dtype=dtype)

    def _free_buffers(self):
        """
        free all memory allocated on the device
        """
        self._cl_mem = release_cl_buffers(self._cl_mem)

    def _free_kernels(self):
        """
        free all kernels
        """
        for kernel in self._cl_kernel_args:
            self._cl_kernel_args[kernel] = []
        self._cl_program = None

    def _compile_kernels(self, sort_kernel=None, separate_kernel=None):
        """
        Compile the kernel
        
        @param kernel_file: filename of the kernel (to test other kernels)
        """
        kernel_file = sort_kernel or "bitonic.cl"
        separate_file = separate_kernel or "separate.cl"
        kernel_src = concatenate_cl_kernel([kernel_file, separate_file])
        compile_options = ""  # -D BLOCK_SIZE=%i  -D BINS=%i -D NN=%i" % \
        try:
            self._cl_program = pyopencl.Program(self.ctx, kernel_src)
            self._cl_program.build(options=compile_options)
        except pyopencl.MemoryError as error:
            raise MemoryError(error)

    def _set_kernel_arguments(self):
        """Tie arguments of OpenCL kernel-functions to the actual kernels

        set_kernel_arguments() is a private method, called by configure().
        It uses the dictionary _cl_kernel_args.

        Note that by default, since TthRange is disabled, the
        integration kernels have tth_min_max tied to the tthRange
        argument slot.

        When setRange is called it replaces that argument with
        tthRange low and upper bounds. When unsetRange is called, the
        argument slot is reset to tth_min_max.
        """
        self._cl_kernel_args["bsort_vertical"] = [self._cl_mem["input_data"].data,
                                                    None]
        self._cl_kernel_args["bsort_horizontal"] = [self._cl_mem["input_data"].data,
                                                    None]

        self._cl_kernel_args["filter_vertical"] = [self._cl_mem["input_data"].data,
                                                   self._cl_mem["vector_vertical"].data,
                                                   numpy.uint32(self.npt_width),
                                                   numpy.uint32(self.npt_height),
                                                   numpy.float32(0), numpy.float32(0.5), ]
        self._cl_kernel_args["filter_horizontal"] = [self._cl_mem["input_data"].data,
                                                     self._cl_mem["vector_horizontal"].data,
                                                     numpy.uint32(self.npt_width),
                                                     numpy.uint32(self.npt_height),
                                                     numpy.float32(0), numpy.float32(0.5), ]


    def sort_vertical(self, data, dummy=None):
        """
        Sort the data along the vertical axis (azimuthal) 
        
        @param data: numpy or pyopencl array
        @param dummy: dummy value
        @return: pyopencl array
        """
        events = []
        assert data.shape[1] == self.npt_width
        assert data.shape[0] <= self.npt_height
        if self.npt_height & (self.npt_height - 1):  # not a power of 2
            raise RuntimeError("Bitonic sort works only for power of two, requested sort on %s element" % self.npt_height)
        if dummy == None:
            dummy = numpy.float32(data.min() - self.DUMMY)
        else:
            dummy = numpy.float32(dummy)
        if data.shape[0] < self.npt_height:
            if isinstance(data, pyopencl.array.Array):
                wg = min(32, self.max_workgroup_size)
                size = ((self.npt_height * self.npt_width) + wg - 1) & ~(wg - 1)
                evt = self._cl_program.copy_pad(self._queue, (size,), (wg,), data.data, self._cl_mem["input_data"].data, data.size, self._cl_mem["input_data"].size, dummy)
                events.append(("copy_pad", evt))
            else:
                data_big = numpy.zeros((self.npt_height, self.npt_width), dtype=numpy.float32) + dummy
                data_big[:data.shape[0], :] = data
                self._cl_mem["input_data"].set(data_big)
        else:
            if isinstance(data, pyopencl.array.Array):
                evt = pyopencl.enqueue(data.queue, self._cl_mem["input_data"].data, data.data)
                events.append(("copy", evt))
            else:
                self._cl_mem["input_data"].set(data)
        ws = self.npt_height // 8
        if self.max_workgroup_size < ws:
            raise RuntimeError("Requested a workgoup size of %s, maximum is %s" % (ws, self.max_workgroup_size))

        local_mem = self._cl_kernel_args["bsort_vertical"][-1]
        if not local_mem or local_mem.size < ws * 32:
            local_mem = pyopencl.LocalMemory(ws * 32)  # 2float4 = 2*4*4 bytes per workgroup size
            self._cl_kernel_args["bsort_vertical"][-1] = local_mem
        evt = self._cl_program.bsort_vertical(self._queue, (ws, self.npt_width), (ws, 1), *self._cl_kernel_args["bsort_vertical"])
        events.append(("bsort_vertical", evt))

        if self.profile:
            with self._sem:
                self.events += events
        return self._cl_mem["input_data"]

    def sort_horizontal(self, data, dummy=None):
        """
        Sort the data along the horizontal axis (radial) 
        
        @param data: numpy or pyopencl array
        @param dummy: dummy value
        @return: pyopencl array
        """
        events = []
        assert data.shape[1] == self.npt_width
        assert data.shape[0] == self.npt_height
        if self.npt_width & (self.npt_width - 1):  # not a power of 2
            raise RuntimeError("Bitonic sort works only for power of two, requested sort on %s element" % self.npt_width)
        if dummy == None:
            dummy = numpy.float32(data.min() - self.DUMMY)
        else:
            dummy = numpy.float32(dummy)
#         if data.shape[0] < self.npt_height:
#             if isinstance(data, pyopencl.array.Array):
#                 wg = min(32, self.max_workgroup_size)
#                 size = ((self.npt_height * self.npt_width) + wg - 1) & ~(wg - 1)
#                 evt = prg.copy_pad(queue, (size,), (ws,), data.data, self._cl_mem["input_data"].data, data.size, self._cl_mem["input_data"].size, dummy)
#                 events.append(("copy_pad", evt))
#             else:
#                 data_big = numpy.zeros((self.npt_height, self.npt_width), dtype=numpy.float32) + dummy
#                 data_big[:data.shape[0], :] = data
#                 self._cl_mem["input_data"].set(data_big)
#         else:
        if isinstance(data, pyopencl.array.Array):
            evt = pyopencl.enqueue(data.queue, self._cl_mem["input_data"].data, data.data)
            events.append(("copy", evt))
        else:
            self._cl_mem["input_data"].set(data)
        ws = self.npt_width // 8
        if self.max_workgroup_size < ws:
            raise RuntimeError("Requested a workgoup size of %s, maximum is %s" % (ws, self.max_workgroup_size))
        local_mem = self._cl_kernel_args["bsort_horizontal"][-1]
        if not local_mem or local_mem.size < ws * 32:
            local_mem = pyopencl.LocalMemory(ws * 32)  # 2float4 = 2*4*4 bytes per workgroup size
            self._cl_kernel_args["bsort_horizontal"][-1] = local_mem
        evt = self._cl_program.bsort_horizontal(self._queue, (self.npt_height, ws), (1, ws), *self._cl_kernel_args["bsort_horizontal"])
        events.append(("bsort_horizontal", evt))

        if self.profile:
            with self._sem:
                self.events += events
        return self._cl_mem["input_data"]


    def filter_vertical(self, data, dummy=None, quantile=0.5):
        """
        Sort the data along the vertical axis (azimuthal) 
        
        @param data: numpy or pyopencl array
        @param dummy: dummy value
        @param quantile: 
        @return: pyopencl array
        """
        if dummy == None:
            dummy = numpy.float32(data.min() - self.DUMMY)
        else:
            dummy = numpy.float32(dummy)
        _sorted = self.sort_vertical(data, dummy)
        wg = min(32, self.max_workgroup_size)
        ws = (self.npt_width + wg - 1) & ~(wg - 1)
        with self._sem:
            args = self._cl_kernel_args["filter_vertical"]
            args[-2] = dummy
            args[-1] = numpy.float32(quantile)
            evt = self._cl_program.filter_vertical(self._queue, (ws,), (wg,), *args)
            self.events.append(("filter_vertical", evt))
        return self._cl_mem["vector_vertical"]

    def filter_horizontal(self, data, dummy=None, quantile=0.5):
        """
        Sort the data along the vertical axis (azimuthal) 
        
        @param data: numpy or pyopencl array
        @param dummy: dummy value
        @param quantile: 
        @return: pyopencl array
        """
        if dummy == None:
            dummy = numpy.float32(data.min() - self.DUMMY)
        else:
            dummy = numpy.float32(dummy)
        _sorted = self.sort_horizontal(data, dummy)
        wg = min(32, self.max_workgroup_size)
        ws = (self.npt_height + wg - 1) & ~(wg - 1)
        with self._sem:
            args = self._cl_kernel_args["filter_horizontal"]
            args[-2] = dummy
            args[-1] = numpy.float32(quantile)
            evt = self._cl_program.filter_horizontal(self._queue, (ws,), (wg,), *args)
            self.events.append(("filter_horizontal", evt))
        return self._cl_mem["vector_horizontal"]


    def log_profile(self):
        """
        If we are in debugging mode, prints out all timing for every single OpenCL call
        """
        t = 0.0
        if self.profile:
            for e in self.events:
                if "__len__" in dir(e) and len(e) >= 2:
                    et = 1e-6 * (e[1].profile.end - e[1].profile.start)
                    print("%50s:\t%.3fms" % (e[0], et))
                    t += et

        print("_"*70)
        print("%50s:\t%.3fms" % ("Total execution time", t))

    def reset_timer(self):
        """
        Resets the profiling timers
        """
        with self._sem:
            self.events = []
