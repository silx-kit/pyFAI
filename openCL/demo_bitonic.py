#!/usr/bin/python
# simple demonstrator for bitonic sort

import numpy, pyopencl, pyopencl.array

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
print("Execution time: %s ms " % (1e-6 * (evt.profile.end - evt.profile.start)))

evt = prg.bsort_all(queue, (ws,), (ws,), d_data.data, local_mem)
print("Execution time: %s ms " % (1e-6 * (evt.profile.end - evt.profile.start)))
