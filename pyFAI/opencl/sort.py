# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2018 European Synchrotron Radiation Facility, Grenoble, France
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
__date__ = "16/05/2019"
__copyright__ = "2015, ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import os
import logging
logger = logging.getLogger(__name__)
from collections import OrderedDict
import numpy
from . import ocl
if ocl:
    import pyopencl.array
    from . import processing
    EventDescription = processing.EventDescription
    OpenclProcessing = processing.OpenclProcessing
    BufferDescription = processing.BufferDescription
else:
    raise ImportError("pyopencl is not installed or no device is available")
from. import release_cl_buffers, kernel_workgroup_size, get_x87_volatile_option


class Separator(OpenclProcessing):
    """
    Implementation of sort, median filter and trimmed-mean in  pyopencl
    """
    DUMMY = numpy.finfo(numpy.float32).min
    kernel_files = ["pyfai:openCL/kahan.cl",
                    "pyfai:openCL/bitonic.cl",
                    "pyfai:openCL/separate.cl",
                    "pyfai:openCL/sigma_clip.cl"]

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

        try:
            default_compiler_options = self.get_compiler_options(x87_volatile=True)
        except AttributeError:  # Silx version too old
            logger.warning("Please upgrade to silx v0.10+")
            default_compiler_options = get_x87_volatile_option(self.ctx)
        self.compile_kernels(compile_options=default_compiler_options)
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
            ("vector_vertical_2", numpy.float32, (self.npt_width,)),
            ("vector_horizontal", numpy.float32, (self.npt_height,)),
            ("vector_horizontal_2", numpy.float32, (self.npt_height,))
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
        self.cl_kernel_args["copy_pad"] = OrderedDict([("src", None),
                                                       ("dst", self.cl_mem["input_data"].data),
                                                       ("src_size", None),
                                                       ("dst_size", numpy.int32(self.cl_mem["input_data"].size)),
                                                       ("dummy", numpy.float32(0.0))])
        self.cl_kernel_args["bsort_vertical"] = OrderedDict([("g_data", self.cl_mem["input_data"].data),
                                                             ("l_data", None)])

        self.cl_kernel_args["bsort_horizontal"] = OrderedDict([("g_data", self.cl_mem["input_data"].data),
                                                               ("l_data", None)])

        self.cl_kernel_args["filter_vertical"] = OrderedDict([("src", self.cl_mem["input_data"].data),
                                                              ("dst", self.cl_mem["vector_vertical"].data),
                                                              ("width", numpy.uint32(self.npt_width)),
                                                              ("height", numpy.uint32(self.npt_height)),
                                                              ("dummy", numpy.float32(0)),
                                                              ("quantile", numpy.float32(0.5))])

        self.cl_kernel_args["filter_horizontal"] = OrderedDict([("src", self.cl_mem["input_data"].data),
                                                                ("dst", self.cl_mem["vector_horizontal"].data),
                                                                ("width", numpy.uint32(self.npt_width)),
                                                                ("height", numpy.uint32(self.npt_height)),
                                                                ("dummy", numpy.float32(0)),
                                                                ("quantile", numpy.float32(0.5))])

        self.cl_kernel_args["trimmed_mean_vertical"] = OrderedDict([("src", self.cl_mem["input_data"].data),
                                                                    ("dst", self.cl_mem["vector_vertical"].data),
                                                                    ("width", numpy.uint32(self.npt_width)),
                                                                    ("height", numpy.uint32(self.npt_height)),
                                                                    ("dummy", numpy.float32(0)),
                                                                    ("lower_quantile", numpy.float32(0.5)),
                                                                    ("upper_quantile", numpy.float32(0.5))])

        self.cl_kernel_args["trimmed_mean_horizontal"] = OrderedDict([("src", self.cl_mem["input_data"].data),
                                                                      ("dst", self.cl_mem["vector_horizontal"].data),
                                                                      ("width", numpy.uint32(self.npt_width)),
                                                                      ("height", numpy.uint32(self.npt_height)),
                                                                      ("dummy", numpy.float32(0)),
                                                                      ("lower_quantile", numpy.float32(0.5)),
                                                                      ("upper_quantile", numpy.float32(0.5))])

        self.cl_kernel_args["mean_std_vertical"] = OrderedDict([("src", self.cl_mem["input_data"].data),
                                                                ("mean", self.cl_mem["vector_vertical"].data),
                                                                ("std", self.cl_mem["vector_vertical_2"].data),
                                                                ("dummy", numpy.float32(0)),
                                                                ("l_data", None)])

        self.cl_kernel_args["mean_std_horizontal"] = OrderedDict([("src", self.cl_mem["input_data"].data),
                                                                  ("mean", self.cl_mem["vector_horizontal"].data),
                                                                  ("std", self.cl_mem["vector_horizontal_2"].data),
                                                                  ("dummy", numpy.float32(0)),
                                                                  ("l_data", None)])

        self.cl_kernel_args["sigma_clip_vertical"] = OrderedDict([("src", self.cl_mem["input_data"].data),
                                                                  ("mean", self.cl_mem["vector_vertical"].data),
                                                                  ("std", self.cl_mem["vector_vertical_2"].data),
                                                                  ("dummy", numpy.float32(0)),
                                                                  ("sigma_lo", numpy.float32(3.0)),
                                                                  ("sigma_hi", numpy.float32(3.0)),
                                                                  ("max_iter", numpy.int32(5)),
                                                                  ("l_data", None)])

        self.cl_kernel_args["sigma_clip_horizontal"] = OrderedDict([("src", self.cl_mem["input_data"].data),
                                                                    ("mean", self.cl_mem["vector_horizontal"].data),
                                                                    ("std", self.cl_mem["vector_horizontal_2"].data),
                                                                    ("dummy", numpy.float32(0)),
                                                                    ("sigma_lo", numpy.float32(3.0)),
                                                                    ("sigma_hi", numpy.float32(3.0)),
                                                                    ("max_iter", numpy.int32(5)),
                                                                    ("l_data", None)])

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

                kargs = self.cl_kernel_args["copy_pad"]
                kargs["src"] = data.data
                kargs["dummy"] = dummy
                kargs["src_size"] = numpy.int32(data.size),
                wg = min(32, self.block_size)
                size = ((self.npt_height * self.npt_width) + wg - 1) & ~(wg - 1)
                evt = self.kernels.copy_pad(self.queue, (size,), (wg,), *kargs.values())
                events.append(EventDescription("copy_pad", evt))
            else:
                data_big = numpy.zeros((self.npt_height, self.npt_width), dtype=numpy.float32) + dummy
                data_big[:data.shape[0], :] = data
                self.cl_mem["input_data"].set(data_big)
        else:
            if isinstance(data, pyopencl.array.Array):
                evt = pyopencl.enqueue(data.queue, self.cl_mem["input_data"].data, data.data)
                events.append(EventDescription("copy", evt))
            else:
                self.cl_mem["input_data"].set(data)
        ws = self.npt_height // 8
        if self.block_size < ws:
            raise RuntimeError("Requested a workgoup size of %s, maximum is %s" % (ws, self.block_size))

        kargs = self.cl_kernel_args["bsort_vertical"]
        local_mem = kargs["l_data"]
        if not local_mem or local_mem.size < ws * 32:
            kargs["l_data"] = pyopencl.LocalMemory(ws * 32)  # 2float4 = 2*4*4 bytes per workgroup size
        evt = self.kernels.bsort_vertical(self.queue, (ws, self.npt_width), (ws, 1), *kargs.values())
        events.append(EventDescription("bsort_vertical", evt))

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
            events.append(EventDescription("copy", evt))
        else:
            self.cl_mem["input_data"].set(data)
        ws = self.npt_width // 8
        if self.block_size < ws:
            raise RuntimeError("Requested a workgoup size of %s, maximum is %s" % (ws, self.block_size))
        kargs = self.cl_kernel_args["bsort_horizontal"]
        local_mem = kargs["l_data"]
        if not local_mem or local_mem.size < ws * 32:
            kargs["l_data"] = pyopencl.LocalMemory(ws * 32)  # 2float4 = 2*4*4 bytes per workgroup size
        evt = self.kernels.bsort_horizontal(self.queue, (self.npt_height, ws), (1, ws), *kargs.values())
        events.append(EventDescription("bsort_horizontal", evt))

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
            kargs = self.cl_kernel_args["filter_vertical"]
            kargs["dummy"] = dummy
            kargs['quantile'] = numpy.float32(quantile)
            evt = self.kernels.filter_vertical(self.queue, (ws,), (wg,), *kargs.values())
            self.events.append(EventDescription("filter_vertical", evt))
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
            kargs = self.cl_kernel_args["filter_horizontal"]
            kargs["dummy"] = dummy
            kargs["quantile"] = numpy.float32(quantile)
            evt = self.kernels.filter_horizontal(self.queue, (ws,), (wg,), *kargs.values())
            self.events.append(EventDescription("filter_horizontal", evt))
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
            kargs = self.cl_kernel_args["trimmed_mean_vertical"]
            kargs["dummy"] = dummy
            kargs["lower_quantile"] = numpy.float32(min(quantiles))
            kargs["upper_quantile"] = numpy.float32(max(quantiles))
            evt = self.kernels.trimmed_mean_vertical(self.queue, (ws,), (wg,), *kargs.values())
            self.events.append(EventDescription("trimmed_mean_vertical", evt))
        return self.cl_mem["vector_vertical"]

    def trimmed_mean_horizontal(self, data, dummy=None, quantiles=(0.5, 0.5)):
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
            kargs = self.cl_kernel_args["trimmed_mean_horizontal"]
            kargs["dummy"] = dummy
            kargs["lower_quantile"] = numpy.float32(min(quantiles))
            kargs["upper_quantile"] = numpy.float32(max(quantiles))
            evt = self.kernels.trimmed_mean_horizontal(self.queue, (ws,), (wg,), *kargs.values())
            self.events.append(EventDescription("trimmed_mean_horizontal", evt))
        return self.cl_mem["vector_horizontal"]

    def mean_std_vertical(self, data, dummy=None):
        """calculates the mean and std along a column,
        column size has to be multiple of 8 and <8192"""
        if dummy is None:
            dummy = self.DUMMY
        else:
            dummy = numpy.float32(dummy)
        assert data.shape[0] == self.npt_height
        assert data.shape[1] == self.npt_width
        wg = self.npt_height // 8
        ws = (wg, self.npt_width)
        with self.sem:
            if isinstance(data, pyopencl.array.Array):
                evt = pyopencl.enqueue_copy(data.queue, self.cl_mem["input_data"].data, data.data)
                events = [EventDescription("copy input", evt)]
            else:
                self.cl_mem["input_data"].set(data)
                events = []
            kargs = self.cl_kernel_args["mean_std_vertical"]
            kargs["dummy"] = dummy
            local_mem = kargs["l_data"]
            if not local_mem or local_mem.size < wg * 20:
                kargs["l_data"] = pyopencl.LocalMemory(wg * 20)  # 5 float per thread
            evt = self.kernels.mean_std_vertical(self.queue, ws, (wg, 1), *kargs.values())
            events.append(EventDescription("mean_std_vertical", evt))
        if self.profile:
            self.events += events
        return self.cl_mem["vector_vertical"], self.cl_mem["vector_vertical_2"]

    def mean_std_horizontal(self, data, dummy=None):
        "calculates the mean and std along a row"
        if dummy is None:
            dummy = self.DUMMY
        else:
            dummy = numpy.float32(dummy)
        assert data.shape[0] == self.npt_height
        assert data.shape[1] == self.npt_width
        wg = self.npt_width // 8
        ws = (self.npt_height, wg)
        with self.sem:
            if isinstance(data, pyopencl.array.Array):
                evt = pyopencl.enqueue_copy(data.queue, self.cl_mem["input_data"].data, data.data)
                events = [EventDescription("copy input", evt)]
            else:
                self.cl_mem["input_data"].set(data)
                events = []
            kargs = self.cl_kernel_args["mean_std_horizontal"]
            kargs["dummy"] = dummy
            local_mem = kargs["l_data"]
            if not local_mem or local_mem.size < wg * 20:
                kargs["l_data"] = pyopencl.LocalMemory(wg * 20)  # 5 float per thread
            evt = self.kernels.mean_std_horizontal(self.queue, ws, (1, wg), *kargs.values())
            events.append(EventDescription("mean_std_horizontal", evt))
        if self.profile:
            self.events += events
        return self.cl_mem["vector_horizontal"], self.cl_mem["vector_horizontal_2"]

    def sigma_clip_vertical(self, data, sigma_lo=3, sigma_hi=None, max_iter=5, dummy=None):
        """calculates iterative sigma-clipped mean and std per column.
        column size has to be multiple of 8 and <8192"""
        if dummy is None:
            dummy = self.DUMMY
        else:
            dummy = numpy.float32(dummy)
        if sigma_hi is None:
            sigma_hi = sigma_lo
        assert data.shape[0] == self.npt_height
        assert data.shape[1] == self.npt_width
        wg = self.npt_height // 8
        ws = (wg, self.npt_width)
        with self.sem:
            if isinstance(data, pyopencl.array.Array):
                evt = pyopencl.enqueue_copy(data.queue, self.cl_mem["input_data"].data, data.data)
                events = [EventDescription("copy input", evt)]
            else:
                self.cl_mem["input_data"].set(data)
                events = []
            kargs = self.cl_kernel_args["sigma_clip_vertical"]
            kargs["dummy"] = dummy
            kargs["sigma_lo"] = numpy.float32(sigma_lo)
            kargs["sigma_hi"] = numpy.float32(sigma_hi)
            kargs["max_iter"] = numpy.int32(max_iter)
            local_mem = kargs["l_data"]
            if not local_mem or local_mem.size < wg * 20:
                kargs["l_data"] = pyopencl.LocalMemory(wg * 20)  # 5 float per thread
            evt = self.kernels.sigma_clip_vertical(self.queue, ws, (wg, 1), *kargs.values())
            events.append(EventDescription("sigma_clip_vertical", evt))
        if self.profile:
            self.events += events
        return self.cl_mem["vector_vertical"], self.cl_mem["vector_vertical_2"]

    def sigma_clip_horizontal(self, data, sigma_lo=3, sigma_hi=None, max_iter=5, dummy=None):
        """calculates iterative sigma-clipped mean and std per row.
        column size has to be multiple of 8 and <8192"""
        if dummy is None:
            dummy = self.DUMMY
        else:
            dummy = numpy.float32(dummy)
        assert data.shape[0] == self.npt_height
        assert data.shape[1] == self.npt_width
        wg = self.npt_width // 8
        ws = (self.npt_height, wg)
        with self.sem:
            if isinstance(data, pyopencl.array.Array):
                evt = pyopencl.enqueue_copy(data.queue, self.cl_mem["input_data"].data, data.data)
                events = [EventDescription("copy input", evt)]
            else:
                self.cl_mem["input_data"].set(data)
                events = []
            kargs = self.cl_kernel_args["sigma_clip_horizontal"]
            kargs["dummy"] = dummy
            kargs["sigma_lo"] = numpy.float32(sigma_lo)
            kargs["sigma_hi"] = numpy.float32(sigma_hi)
            kargs["max_iter"] = numpy.int32(max_iter)
            local_mem = kargs["l_data"]
            if not local_mem or local_mem.size < wg * 20:
                kargs["l_data"] = pyopencl.LocalMemory(wg * 20)  # 5 float per thread
            evt = self.kernels.sigma_clip_horizontal(self.queue, ws, (1, wg), *kargs.values())
            events.append(EventDescription("sigma_clip_horizontal", evt))
        if self.profile:
            self.events += events
        return self.cl_mem["vector_horizontal"], self.cl_mem["vector_horizontal_2"]
