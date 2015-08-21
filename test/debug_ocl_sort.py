from __future__ import print_function
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
