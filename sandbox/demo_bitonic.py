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

"""
simple demonstrator for bitonic sort
"""

import numpy, pyopencl, pyopencl.array, time

N = 1024
ws = N // 8
h_data = numpy.random.random(N).astype("float32")

ctx = pyopencl.create_some_context()
queue = pyopencl.CommandQueue(ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
d_data = pyopencl.array.to_device(queue, h_data)
local_mem = pyopencl.LocalMemory(ws * 32)  # 2float4 = 2*4*4 bytes per workgroup size
src = open("../openCL/bitonic.cl").read().strip()
prg = pyopencl.Program(ctx, src).build()
evt = prg.bsort_file(queue, (ws,), (ws,), d_data.data, local_mem)
evt.wait()
t0 = time.time()
hs_data = numpy.sort(h_data)
t1 = time.time()
time_sort = 1e3 * (t1 - t0)

print("Numpy sort on %s element took %s ms" % (N, time_sort))
print("Reference Execution time: %s ms, err=%s " % (1e-6 * (evt.profile.end - evt.profile.start), abs(numpy.sort(h_data) - d_data.get()).max()))
d_data = pyopencl.array.to_device(queue, h_data)
evt = prg.bsort_all(queue, (ws,), (ws,), d_data.data, local_mem)
evt.wait()
print("Global Execution time: %s ms, err=%s " % (1e-6 * (evt.profile.end - evt.profile.start), abs(numpy.sort(h_data) - d_data.get()).max()))

print("")
print("*" * 80)
print("")

h2_data = numpy.random.random((N, N)).astype("float32").reshape((N, N))
d2_data = pyopencl.array.to_device(queue, h2_data)
t0 = time.time()
h2s_data = numpy.sort(h2_data, axis=-1)
t1 = time.time()
time_sort_hor = 1e3 * (t1 - t0)
print("Numpy horizontal sort on %sx%s elements took %s ms" % (N, N, time_sort_hor))

evt = prg.bsort_horizontal(queue, (N, ws), (1, ws), d2_data.data, local_mem)
evt.wait()
print("Horizontal Execution time: %s ms, err=%s " % (1e-6 * (evt.profile.end - evt.profile.start), abs(h2s_data - d2_data.get()).max()))

print("")
print("*" * 80)
print("")

d2_data = pyopencl.array.to_device(queue, h2_data)
t0 = time.time()
h2s_data = numpy.sort(h2_data, axis=0)
t1 = time.time()
time_sort_ver = 1e3 * (t1 - t0)
print("Numpy vertical sort on %sx%s elements took %s ms" % (N, N, time_sort_ver))

evt = prg.bsort_vertical(queue, (ws, N), (ws, 1), d2_data.data, local_mem)
evt.wait()
print("Vertical Execution time: %s ms, err=%s " % (1e-6 * (evt.profile.end - evt.profile.start), abs(h2s_data - d2_data.get()).max()))
