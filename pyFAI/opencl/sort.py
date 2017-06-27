# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
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
__date__ = "27/06/2017"
__copyright__ = "2015, ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import os
import logging
import threading
import numpy
import gc
from .utils import concatenate_cl_kernel
from .common import ocl, pyopencl, release_cl_buffers, mf, kernel_workgroup_size
if ocl:
    import pyopencl.array
else:
    raise ImportError("pyopencl is not installed or no device is available")
from .processing import EventDescription, OpenclProcessing, BufferDescription
logger = logging.getLogger("pyFAI.opencl.sort")


class Separator(OpenclProcessing):
    """
    Implementation of sort, median filter and trimmed-mean in  pyopencl
    """
    DUMMY = numpy.finfo(numpy.float32).min
    kernel_files = ["bitonic.cl", "separate.cl"]

    def __init__(self, npt_height=512, npt_width=1024, ctx=None, devicetype="all",
                 platformid=None, deviceid=None,
                 block_size=None, profile=False):
        """
        :param ctx: context
        :param block_size: 1 on macOSX on CPU
        :param profile: turn on profiling
        """
        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
                                  platformid=platformid, deviceid=deviceid,
                                  block_size=block_size, profile=profile)

        self.npt_width = npt_width
        self.npt_height = npt_height

        self.allocate_buffers()
        self.compile_kernels()
        if block_size is None:
            self.block_size = kernel_workgroup_size(self.program, "filter_vertical")
        else:
            self.block_size = min(block_size, kernel_workgroup_size(self.program, "filter_vertical"))
        self.set_kernel_arguments()

    def __repr__(self):
        lst = ["OpenCL implementation of sort/median_filter/trimmed_mean"]
        return os.linesep.join(lst)

    def allocate_buffers(self, *arg, **kwarg):
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
        mem = {}
        with self.sem:
            try:
                for name, dtype, shape in buffers:
                    mem[name] = pyopencl.array.Array(self.queue, shape=shape, dtype=dtype)
            except pyopencl.MemoryError as error:
                release_cl_buffers(mem)
                raise MemoryError(error)
        self.cl_mem.update(mem)

    def set_kernel_arguments(self):
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
        self.cl_kernel_args["bsort_vertical"] = [self.cl_mem["input_data"].data, None]
        self.cl_kernel_args["bsort_horizontal"] = [self.cl_mem["input_data"].data, None]

        self.cl_kernel_args["filter_vertical"] = [self.cl_mem["input_data"].data,
                                                   self.cl_mem["vector_vertical"].data,
                                                   numpy.uint32(self.npt_width),
                                                   numpy.uint32(self.npt_height),
                                                   numpy.float32(0), numpy.float32(0.5), ]
        self.cl_kernel_args["filter_horizontal"] = [self.cl_mem["input_data"].data,
                                                     self.cl_mem["vector_horizontal"].data,
                                                     numpy.uint32(self.npt_width),
                                                     numpy.uint32(self.npt_height),
                                                     numpy.float32(0), numpy.float32(0.5), ]

    def sort_vertical(self, data, dummy=None):
        """
        Sort the data along the vertical axis (azimuthal)

        :param data: numpy or pyopencl array
        :param dummy: dummy value
        :return: pyopencl array
        """
        events = []
        assert data.shape[1] == self.npt_width
        assert data.shape[0] <= self.npt_height
        if self.npt_height & (self.npt_height - 1):  # not a power of 2
            raise RuntimeError("Bitonic sort works only for power of two, requested sort on %s element" % self.npt_height)
        if dummy is None:
            dummy = self.DUMMY
        else:
            dummy = numpy.float32(dummy)
        if data.shape[0] < self.npt_height:
            if isinstance(data, pyopencl.array.Array):
                wg = min(32, self.block_size)
                size = ((self.npt_height * self.npt_width) + wg - 1) & ~(wg - 1)
                evt = self.program.copy_pad(self.queue, (size,), (wg,), data.data, self.cl_mem["input_data"].data, data.size, self.cl_mem["input_data"].size, dummy)
                events.append(("copy_pad", evt))
            else:
                data_big = numpy.zeros((self.npt_height, self.npt_width), dtype=numpy.float32) + dummy
                data_big[:data.shape[0], :] = data
                self.cl_mem["input_data"].set(data_big)
        else:
            if isinstance(data, pyopencl.array.Array):
                evt = pyopencl.enqueue(data.queue, self.cl_mem["input_data"].data, data.data)
                events.append(("copy", evt))
            else:
                self.cl_mem["input_data"].set(data)
        ws = self.npt_height // 8
        if self.block_size < ws:
            raise RuntimeError("Requested a workgoup size of %s, maximum is %s" % (ws, self.block_size))

        local_mem = self.cl_kernel_args["bsort_vertical"][-1]
        if not local_mem or local_mem.size < ws * 32:
            local_mem = pyopencl.LocalMemory(ws * 32)  # 2float4 = 2*4*4 bytes per workgroup size
            self.cl_kernel_args["bsort_vertical"][-1] = local_mem
        evt = self.program.bsort_vertical(self.queue, (ws, self.npt_width), (ws, 1), *self.cl_kernel_args["bsort_vertical"])
        events.append(("bsort_vertical", evt))

        if self.profile:
            with self.sem:
                self.events += events
        return self.cl_mem["input_data"]

    def sort_horizontal(self, data, dummy=None):
        """
        Sort the data along the horizontal axis (radial)

        :param data: numpy or pyopencl array
        :param dummy: dummy value
        :return: pyopencl array
        """
        events = []
        assert data.shape[1] == self.npt_width
        assert data.shape[0] == self.npt_height
        if self.npt_width & (self.npt_width - 1):  # not a power of 2
            raise RuntimeError("Bitonic sort works only for power of two, requested sort on %s element" % self.npt_width)
        if dummy is None:
            dummy = self.DUMMY
        else:
            dummy = numpy.float32(dummy)
#         if data.shape[0] < self.npt_height:
#             if isinstance(data, pyopencl.array.Array):
#                 wg = min(32, self.block_size)
#                 size = ((self.npt_height * self.npt_width) + wg - 1) & ~(wg - 1)
#                 evt = prg.copy_pad(queue, (size,), (ws,), data.data, self.cl_mem["input_data"].data, data.size, self.cl_mem["input_data"].size, dummy)
#                 events.append(("copy_pad", evt))
#             else:
#                 data_big = numpy.zeros((self.npt_height, self.npt_width), dtype=numpy.float32) + dummy
#                 data_big[:data.shape[0], :] = data
#                 self.cl_mem["input_data"].set(data_big)
#         else:
        if isinstance(data, pyopencl.array.Array):
            evt = pyopencl.enqueue(data.queue, self.cl_mem["input_data"].data, data.data)
            events.append(("copy", evt))
        else:
            self.cl_mem["input_data"].set(data)
        ws = self.npt_width // 8
        if self.block_size < ws:
            raise RuntimeError("Requested a workgoup size of %s, maximum is %s" % (ws, self.block_size))
        local_mem = self.cl_kernel_args["bsort_horizontal"][-1]
        if not local_mem or local_mem.size < ws * 32:
            local_mem = pyopencl.LocalMemory(ws * 32)  # 2float4 = 2*4*4 bytes per workgroup size
            self.cl_kernel_args["bsort_horizontal"][-1] = local_mem
        evt = self.program.bsort_horizontal(self.queue, (self.npt_height, ws), (1, ws), *self.cl_kernel_args["bsort_horizontal"])
        events.append(("bsort_horizontal", evt))

        if self.profile:
            with self.sem:
                self.events += events
        return self.cl_mem["input_data"]

    def filter_vertical(self, data, dummy=None, quantile=0.5):
        """
        Sort the data along the vertical axis (azimuthal)

        :param data: numpy or pyopencl array
        :param dummy: dummy value
        :param quantile:
        :return: pyopencl array
        """
        if dummy is None:
            dummy = self.DUMMY
        else:
            dummy = numpy.float32(dummy)
        _sorted = self.sort_vertical(data, dummy)
        wg = min(32, self.block_size)
        ws = (self.npt_width + wg - 1) & ~(wg - 1)
        with self.sem:
            args = self.cl_kernel_args["filter_vertical"]
            args[-2] = dummy
            args[-1] = numpy.float32(quantile)
            evt = self.program.filter_vertical(self.queue, (ws,), (wg,), *args)
            self.events.append(("filter_vertical", evt))
        return self.cl_mem["vector_vertical"]

    def filter_horizontal(self, data, dummy=None, quantile=0.5):
        """
        Sort the data along the vertical axis (azimuthal)

        :param data: numpy or pyopencl array
        :param dummy: dummy value
        :param quantile:
        :return: pyopencl array
        """
        if dummy is None:
            dummy = self.DUMMY
        else:
            dummy = numpy.float32(dummy)
        _sorted = self.sort_horizontal(data, dummy)
        wg = min(32, self.block_size)
        ws = (self.npt_height + wg - 1) & ~(wg - 1)
        with self.sem:
            args = self.cl_kernel_args["filter_horizontal"]
            args[-2] = dummy
            args[-1] = numpy.float32(quantile)
            evt = self.program.filter_horizontal(self.queue, (ws,), (wg,), *args)
            self.events.append(("filter_horizontal", evt))
        return self.cl_mem["vector_horizontal"]

    def trimmed_mean_vertical(self, data, dummy=None, quantiles=(0.5, 0.5)):
        """
        Perform a trimmed mean (mean without the extremes) 
        After sorting the data along the vertical axis (azimuthal)

        :param data: numpy or pyopencl array
        :param dummy: dummy value
        :param quantile:
        :return: pyopencl array
        """
        if dummy is None:
            dummy = self.DUMMY
        else:
            dummy = numpy.float32(dummy)
        _sorted = self.sort_vertical(data, dummy)
        wg = min(32, self.block_size)
        ws = (self.npt_width + wg - 1) & ~(wg - 1)
        with self.sem:
            args = self.cl_kernel_args["filter_vertical"]
            args[-2] = dummy
            args[-1] = numpy.float32(quantile)
            evt = self.program.filter_vertical(self.queue, (ws,), (wg,), *args)
            self.events.append(("filter_vertical", evt))
        return self.cl_mem["vector_vertical"]

    def trimmed_mean_horizontal(self, data, dummy=None, quantile=(0.5, 0.5)):
        """
        Perform a trimmed mean (mean without the extremes) 
        After sorting the data along the vertical axis (azimuthal)

        :param data: numpy or pyopencl array
        :param dummy: dummy value
        :param quantile:
        :return: pyopencl array
        """
        if dummy is None:
            dummy = self.DUMMY
        else:
            dummy = numpy.float32(dummy)
        _sorted = self.sort_horizontal(data, dummy)
        wg = min(32, self.block_size)
        ws = (self.npt_height + wg - 1) & ~(wg - 1)
        with self.sem:
            args = self.cl_kernel_args["filter_horizontal"]
            args[-2] = dummy
            args[-1] = numpy.float32(quantile)
            evt = self.program.filter_horizontal(self.queue, (ws,), (wg,), *args)
            self.events.append(("filter_horizontal", evt))
        return self.cl_mem["vector_horizontal"]
