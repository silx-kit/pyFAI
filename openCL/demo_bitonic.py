#!/usr/bin/python
# simple demonstrator for bitonic sort

import numpy, pyopencl, pyopencl.array, time

N = 1024
ws = N // 8
h_data = numpy.random.random(N).astype("float32")

ctx = pyopencl.create_some_context()
queue = pyopencl.CommandQueue(ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
d_data = pyopencl.array.to_device(queue, h_data)
local_mem = pyopencl.LocalMemory(ws * 32)  # 2float4 = 2*4*4 bytes per workgroup size
src = open("bitonic.cl").read().strip()
prg = pyopencl.Program(ctx, src).build()
evt = prg.bsort(queue, (ws,), (ws,), d_data.data, local_mem)
evt.wait()
print("Reference Execution time: %s ms, err=%s " % (1e-6 * (evt.profile.end - evt.profile.start), abs(numpy.sort(h_data) - d_data.get()).max()))
d_data = pyopencl.array.to_device(queue, h_data)
evt = prg.bsort_all(queue, (ws,), (ws,), d_data.data, local_mem)
evt.wait()
print("Global Execution time: %s ms, err=%s " % (1e-6 * (evt.profile.end - evt.profile.start), abs(numpy.sort(h_data) - d_data.get()).max()))

print()
print("*"*80)
print()

h2_data = numpy.random.random((N, N)).astype("float32").reshape((N, N))
d2_data = pyopencl.array.to_device(queue, h2_data)
t0 = time.time()
h2s_data = numpy.sort(h2_data, axis=-1)
t1 = time.time()
time_sort_hor = 1e3 * (t1 - t0)
evt = prg.bsort_horizontal(queue, (N, ws), (1, ws), d2_data.data, local_mem)
evt.wait()
print("Horizontal Execution time: %s ms, err=%s " % (1e-6 * (evt.profile.end - evt.profile.start), abs(h2s_data - d2_data.get()).max()))

print()
print("*"*80)
print()
