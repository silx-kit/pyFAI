# -*- coding: utf-8 -*-
"""
Created on Fri Mar 07 09:52:51 2014

@author: ashiotis
"""

import sys, numpy, time
import utilstest
import fabio
import pyopencl as cl
from pylab import *
print "#"*50
pyFAI = sys.modules["pyFAI"]
from pyFAI import splitPixelFullLUT
from pyFAI import ocl_hist_pixelsplit
#from pyFAI import splitBBoxLUT
#from pyFAI import splitBBoxCSR
#logger = utilstest.getLogger("profile")


ai = pyFAI.load("testimages/halfccd.poni")
data = fabio.open("testimages/halfccd.edf").data

workgroup_size = 256
bins = 1000

pos_in = ai.array_from_unit(data.shape, "corner", unit="2th_deg")

pos = pos_in.reshape(pos_in.size/8,4,2)

pos_size = pos.size
#size = data.size
size = pos_size/8

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

d_pos     = cl.array.to_device(queue, pos)
d_preresult = cl.Buffer(ctx, mf.READ_WRITE, 4*4*workgroup_size)
d_minmax = cl.Buffer(ctx, mf.READ_WRITE, 4*4)

with open("../openCL/ocl_lut_pixelsplit.cl", "r") as kernelFile:
    kernel_src = kernelFile.read()

compile_options = "-D BINS=%i  -D NIMAGE=%i -D WORKGROUP_SIZE=%i -D EPS=%f" % \
                (bins, size, workgroup_size, numpy.finfo(numpy.float32).eps)

program = cl.Program(ctx, kernel_src).build(options=compile_options)

program.reduce1(queue, (workgroup_size*workgroup_size,), (workgroup_size,), d_pos.data,  numpy.uint32(pos_size), d_preresult)

program.reduce2(queue, (workgroup_size,), (workgroup_size,), d_preresult, d_minmax)

result = numpy.ndarray(4,dtype=numpy.float32)

cl.enqueue_copy(queue,result, d_minmax)


min0 = pos[:, :, 0].min()
max0 = pos[:, :, 0].max()
min1 = pos[:, :, 1].min()
max1 = pos[:, :, 1].max()
minmax=(min0,max0,min1,max1)

print minmax
print result


memset_size = (bins + workgroup_size - 1) & ~(workgroup_size - 1),

d_outMax  = cl.Buffer(ctx, mf.READ_WRITE, 4*bins)
program.memset_out_int(queue, memset_size, (workgroup_size,), d_outMax)


global_size = (size + workgroup_size - 1) & ~(workgroup_size - 1),

program.lut1(queue, global_size, (workgroup_size,), d_pos.data, d_minmax, numpy.uint32(size), d_outMax)


outMax_1  = numpy.ndarray(bins, dtype=numpy.int32)

cl.enqueue_copy(queue, outMax_1, d_outMax)


d_idx_ptr = cl.Buffer(ctx, mf.READ_WRITE, 4*(bins+1))

d_lutsize = cl.Buffer(ctx, mf.READ_WRITE, 4)

program.lut2(queue, (1,), (1,), d_outMax, d_idx_ptr, d_lutsize)

lutsize  = numpy.ndarray(1, dtype=numpy.int32)

cl.enqueue_copy(queue, lutsize, d_lutsize)

print lutsize

lut_size = int(lutsize[0])

d_indices  = cl.Buffer(ctx, mf.READ_WRITE, 4*lut_size)
d_data     = cl.Buffer(ctx, mf.READ_WRITE, 4*lut_size)

#d_check_atomics = cl.Buffer(ctx, mf.READ_WRITE, 4*lut_size)


program.memset_out_int(queue, memset_size, (workgroup_size,), d_outMax)

d_outData  = cl.Buffer(ctx, mf.READ_WRITE, 4*bins)
d_outCount = cl.Buffer(ctx, mf.READ_WRITE, 4*bins)
d_outMerge = cl.Buffer(ctx, mf.READ_WRITE, 4*bins)

program.lut3(queue, global_size, (workgroup_size,), d_pos.data, d_minmax, numpy.uint32(size), d_outMax, d_idx_ptr, d_indices, d_data)


#check_atomics = numpy.ndarray(lut_size, dtype=numpy.int32)

#cl.enqueue_copy(queue, check_atomics, d_check_atomics)


program.memset_out(queue, memset_size, (workgroup_size,), d_outData, d_outCount, d_outMerge)




d_image = cl.array.to_device(queue, data)
d_image_float = cl.Buffer(ctx, mf.READ_WRITE, 4*size)

#program.s32_to_float(queue, global_size, (workgroup_size,), d_image.data, d_image_float)  # Pilatus1M
program.u16_to_float(queue, global_size, (workgroup_size,), d_image.data, d_image_float)  # halfccd

program.csr_integrate(queue, (bins*workgroup_size,),(workgroup_size,), d_image_float, d_data, d_indices, d_idx_ptr, d_outData, d_outCount, d_outMerge)


#outData  = numpy.ndarray(bins, dtype=numpy.float32)
#outCount = numpy.ndarray(bins, dtype=numpy.float32)
outMerge = numpy.ndarray(bins, dtype=numpy.float32)


#cl.enqueue_copy(queue,outData, d_outData)
#cl.enqueue_copy(queue,outCount, d_outCount)
cl.enqueue_copy(queue,outMerge, d_outMerge)

#program.integrate2(queue, (1024,), (workgroup_size,), d_outData, d_outCount, d_outMerge)

#cl.enqueue_copy(queue,outData, d_outData)
#cl.enqueue_copy(queue,outCount, d_outCount)
#cl.enqueue_copy(queue,outMerge, d_outMerge)



ref = ai.xrpd_LUT(data, 1000)[1]


#assert(numpy.allclose(ref,outMerge))

plot(outMerge, label="ocl_hist")
plot(ref, label="ref")
##plot(abs(ref-outMerge)/outMerge, label="ocl_csr_fullsplit")
legend()
show()
raw_input()

