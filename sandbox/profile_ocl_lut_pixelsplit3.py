# -*- coding: utf-8 -*-
"""
Created on Fri Mar 07 09:52:51 2014

@author: ashiotis
"""
from __future__ import absolute_import
from __future__ import print_function

import sys, numpy, time
from . import utilstest
import fabio
import pyopencl as cl
from pylab import *
print("#"*50)
pyFAI = sys.modules["pyFAI"]
from pyFAI import splitPixelFullLUT
from pyFAI import ocl_hist_pixelsplit
try:
    from pyFAI.third_party import six
except (ImportError, Exception):
    import six
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

d_pos       = cl.array.to_device(queue, pos)
d_preresult = cl.array.empty(queue, (4*workgroup_size,), dtype=numpy.float32)
d_minmax    = cl.array.empty(queue, (4,), dtype=numpy.float32)

with open("../openCL/ocl_lut_pixelsplit.cl", "r") as kernelFile:
    kernel_src = kernelFile.read()

compile_options = "-D BINS=%i  -D NIMAGE=%i -D WORKGROUP_SIZE=%i -D EPS=%e" % \
                (bins, size, workgroup_size, numpy.finfo(numpy.float32).eps)

print(compile_options)

program = cl.Program(ctx, kernel_src).build(options=compile_options)

program.reduce1(queue, (workgroup_size*workgroup_size,), (workgroup_size,), d_pos.data,  numpy.uint32(pos_size), d_preresult.data)

program.reduce2(queue, (workgroup_size,), (workgroup_size,), d_preresult.data, d_minmax.data)


min0 = pos[:, :, 0].min()
max0 = pos[:, :, 0].max()
min1 = pos[:, :, 1].min()
max1 = pos[:, :, 1].max()
minmax=(min0,max0,min1,max1)

print(minmax)
print(d_minmax)


memset_size = (bins + workgroup_size - 1) & ~(workgroup_size - 1),

d_outMax  = cl.array.empty(queue, (bins,), dtype=numpy.int32)

program.memset_out_int(queue, memset_size, (workgroup_size,), d_outMax.data)

global_size = (size + workgroup_size - 1) & ~(workgroup_size - 1),

program.lut1(queue, global_size, (workgroup_size,), d_pos.data, d_minmax.data, numpy.uint32(size), d_outMax.data)


outMax_1  = numpy.copy(d_outMax)



d_idx_ptr = cl.array.empty(queue, (bins+1,), dtype=numpy.int32)

d_lutsize = cl.array.empty(queue, (1,), dtype=numpy.int32)

program.lut2(queue, (1,), (1,), d_outMax.data, d_idx_ptr.data, d_lutsize.data)

lutsize  = numpy.ndarray(1, dtype=numpy.int32)

cl.enqueue_copy(queue, lutsize, d_lutsize.data)

print(lutsize)

lut_size = int(lutsize[0])

d_indices  = cl.array.empty(queue, (lut_size,), dtype=numpy.int32)
d_data     = cl.array.empty(queue, (lut_size,), dtype=numpy.float32)

#d_check_atomics = cl.Buffer(ctx, mf.READ_WRITE, 4*lut_size)


program.memset_out_int(queue, memset_size, (workgroup_size,), d_outMax.data)

d_outData  = cl.array.empty(queue, (bins,), dtype=numpy.float32)
d_outCount = cl.array.empty(queue, (bins,), dtype=numpy.float32)
d_outMerge = cl.array.empty(queue, (bins,), dtype=numpy.float32)

program.lut3(queue, global_size, (workgroup_size,), d_pos.data, d_minmax.data, numpy.uint32(size), d_outMax.data, d_idx_ptr.data, d_indices.data, d_data.data)


outMax_2  = numpy.copy(d_outMax)

#check_atomics = numpy.ndarray(lut_size, dtype=numpy.int32)

#cl.enqueue_copy(queue, check_atomics, d_check_atomics)


program.memset_out(queue, memset_size, (workgroup_size,), d_outData.data, d_outCount.data, d_outMerge.data)




d_image = cl.array.to_device(queue, data)
d_image_float = cl.array.empty(queue, (size,), dtype=numpy.float32)

#program.s32_to_float(queue, global_size, (workgroup_size,), d_image.data, d_image_float)  # Pilatus1M
program.u16_to_float(queue, global_size, (workgroup_size,), d_image.data, d_image_float.data)  # halfccd

program.csr_integrate(queue, (bins*workgroup_size,),(workgroup_size,), d_image_float.data, d_data.data, d_indices.data, d_idx_ptr.data, d_outData.data, d_outCount.data, d_outMerge.data)


#outData  = numpy.ndarray(bins, dtype=numpy.float32)
#outCount = numpy.ndarray(bins, dtype=numpy.float32)
outMerge = numpy.ndarray(bins, dtype=numpy.float32)


#cl.enqueue_copy(queue,outData, d_outData)
#cl.enqueue_copy(queue,outCount, d_outCount)
cl.enqueue_copy(queue,outMerge, d_outMerge.data)

#program.integrate2(queue, (1024,), (workgroup_size,), d_outData, d_outCount, d_outMerge)

#cl.enqueue_copy(queue,outData, d_outData)
#cl.enqueue_copy(queue,outCount, d_outCount)
#cl.enqueue_copy(queue,outMerge, d_outMerge)



#ref = ai.integrate1d(data,bins,unit="2th_deg", correctSolidAngle=False, method="splitpixelfull")

ref = splitPixelFullLUT.HistoLUT1dFullSplit(pos,bins, unit="2th_deg")

#assert(numpy.allclose(ref,outMerge))

##plot(ref[0],outMerge, label="ocl_lut_merge")
###plot(ref[0],outData, label="ocl_lut_data")
###plot(ref[0],outCount, label="ocl_lut_count")
#plot(ref[1], label="ref_merge")
###plot(ref[0], ref[2], label="ref_data")
###plot(ref[0], ref[3], label="ref_count")
####plot(abs(ref-outMerge)/outMerge, label="ocl_csr_fullsplit")
#legend()
#show()
#six.moves.input()

