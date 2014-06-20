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


ai = pyFAI.load("testimages/Pilatus1M.poni")
data = fabio.open("testimages/Pilatus1M.edf").data

workgroup_size = 256
bins = 1000

pos_in = ai.array_from_unit(data.shape, "corner", unit="2th_deg")

pos = pos_in.reshape(pos_in.size/8,4,2)

pos_size = pos.size
size = pos_size/8


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

d_input     = cl.array.to_device(queue, pos)
d_preresult = cl.Buffer(ctx, mf.READ_WRITE, 4*4*workgroup_size)
d_result = cl.Buffer(ctx, mf.READ_WRITE, 4*4)

with open("../openCL/ocl_hist_pixelsplit.cl", "r") as kernelFile:
    kernel_src = kernelFile.read()

compile_options = "-D BINS=%i  -D NIMAGE=%i -D WORKGROUP_SIZE=%i -D EPS=%f" % \
                (bins, size, workgroup_size, numpy.finfo(numpy.float32).eps)

program = cl.Program(ctx, kernel_src).build(options=compile_options)

program.reduce1(queue, (workgroup_size*workgroup_size,), (workgroup_size,), d_input.data,  numpy.uint32(pos_size), d_preresult)

program.reduce2(queue, (workgroup_size,), (workgroup_size,), d_preresult, d_result)

result = numpy.ndarray(4,dtype=numpy.float32)

cl.enqueue_copy(queue,result, d_result)


min0 = pos[:, :, 0].min()
max0 = pos[:, :, 0].max()
min1 = pos[:, :, 1].min()
max1 = pos[:, :, 1].max()
minmax=(min0,max0,min1,max1)

print minmax
print result





#plot(ref, label="ocl_csr")
#plot(boo, label="csr_fullsplit")
#plot(boo2, label="ocl_csr_fullsplit")
#legend()
#show()
#raw_input()

