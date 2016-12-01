#!/usr/bin/env python
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

from __future__ import print_function
import utilstest
import numpy, time
import pyFAI, pyFAI.opencl
from pyFAI.opencl import pyopencl, ocl
import pyopencl.array

N = 1024
ws = N // 8

ctx = ocl.create_context("GPU")
queue = pyopencl.CommandQueue(ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)

h_data = numpy.random.random(N).astype("float32")
d_data = pyopencl.array.to_device(queue, h_data)
local_mem = pyopencl.LocalMemory(ws * 32)  # 2float4 = 2*4*4 bytes per workgroup size

src = pyFAI.utils.read_cl_file("bsort.cl")
prg = pyopencl.Program(ctx, src).build()

t0 = time.time()
hs_data = numpy.sort(h_data)
t1 = time.time()
time_sort = 1e3 * (t1 - t0)

print(time_sort)

evt = prg.bsort_init(queue, (ws,), (ws,), d_data.data, local_mem)
evt.wait()
err = abs(hs_data - d_data.get()).max()
print("Numpy sort on %s element took %s ms" % (N, time_sort))
print("Reference sort time: %s ms, err=%s " % (1e-6 * (evt.profile.end - evt.profile.start), err))
