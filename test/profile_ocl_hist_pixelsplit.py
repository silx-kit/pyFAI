# -*- coding: utf-8 -*-
"""
Created on Fri Mar 07 09:52:51 2014

@author: ashiotis
"""
from __future__ import absolute_import
from __future__ import print_function

import sys, numpy, time, os

import fabio
import pyopencl as cl
from pylab import *
from six.moves import input
print("#"*50)
if __name__ == '__main__':
    import pkgutil
    __path__ = pkgutil.extend_path([os.path.dirname(__file__)], "pyFAI.test")
from .utilstest import UtilsTest, getLogger

pyFAI = sys.modules["pyFAI"]
from pyFAI import splitPixelFullLUT
from pyFAI import splitPixelFull
from pyFAI import ocl_hist_pixelsplit
# from pyFAI import splitBBoxLUT
# from pyFAI import splitBBoxCSR

os.chdir("testimages")
ai = pyFAI.load("halfccd.poni")
data = fabio.open("halfccd.edf").data

workgroup_size = 256
bins = 1000

pos_in = ai.array_from_unit(data.shape, "corner", unit="2th_deg")

pos = pos_in.reshape(pos_in.size / 8, 4, 2)

pos_size = pos.size
size = data.size


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

d_pos = cl.array.to_device(queue, pos)
d_preresult = cl.Buffer(ctx, mf.READ_WRITE, 4 * 4 * workgroup_size)
d_minmax = cl.Buffer(ctx, mf.READ_WRITE, 4 * 4)

with open("../../openCL/ocl_hist_pixelsplit.cl", "r") as kernelFile:
    kernel_src = kernelFile.read()

compile_options = "-D BINS=%i  -D NIMAGE=%i -D WORKGROUP_SIZE=%i -D EPS=%f" % \
                (bins, size, workgroup_size, numpy.finfo(numpy.float32).eps)

program = cl.Program(ctx, kernel_src).build(options=compile_options)

program.reduce1(queue, (workgroup_size * workgroup_size,), (workgroup_size,), d_pos.data, numpy.uint32(pos_size), d_preresult)

program.reduce2(queue, (workgroup_size,), (workgroup_size,), d_preresult, d_minmax)

result = numpy.ndarray(4, dtype=numpy.float32)

cl.enqueue_copy(queue, result, d_minmax)


min0 = pos[:, :, 0].min()
max0 = pos[:, :, 0].max()
min1 = pos[:, :, 1].min()
max1 = pos[:, :, 1].max()
minmax = (min0, max0, min1, max1)

print(minmax)
print(result)


d_outData = cl.Buffer(ctx, mf.READ_WRITE, 4 * bins)
d_outCount = cl.Buffer(ctx, mf.READ_WRITE, 4 * bins)
d_outMerge = cl.Buffer(ctx, mf.READ_WRITE, 4 * bins)

program.memset_out(queue, (1024,), (workgroup_size,), d_outData, d_outCount, d_outMerge)

outData = numpy.ndarray(bins, dtype=numpy.float32)
outCount = numpy.ndarray(bins, dtype=numpy.float32)
outMerge = numpy.ndarray(bins, dtype=numpy.float32)

cl.enqueue_copy(queue, outData, d_outData)
cl.enqueue_copy(queue, outCount, d_outCount)
cl.enqueue_copy(queue, outMerge, d_outMerge)

global_size = (data.size + workgroup_size - 1) & ~(workgroup_size - 1),

d_image = cl.array.to_device(queue, data)
d_image_float = cl.Buffer(ctx, mf.READ_WRITE, 4 * size)

# program.s32_to_float(queue, global_size, (workgroup_size,), d_image.data, d_image_float)  # Pilatus1M
program.u16_to_float(queue, global_size, (workgroup_size,), d_image.data, d_image_float)  # halfccd

program.integrate1(queue, global_size, (workgroup_size,), d_pos.data, d_image_float, d_minmax, numpy.int32(data.size), d_outData, d_outCount)

cl.enqueue_copy(queue, outData, d_outData)
cl.enqueue_copy(queue, outCount, d_outCount)
cl.enqueue_copy(queue, outMerge, d_outMerge)

program.integrate2(queue, (1024,), (workgroup_size,), d_outData, d_outCount, d_outMerge)

cl.enqueue_copy(queue, outData, d_outData)
cl.enqueue_copy(queue, outCount, d_outCount)
cl.enqueue_copy(queue, outMerge, d_outMerge)



ref = ai.xrpd_LUT(data, bins, correctSolidAngle=False)
test = splitPixelFull.fullSplit1D(pos, data, bins)

# assert(numpy.allclose(ref,outMerge))

# plot(outMerge, label="ocl_hist")
plot(ref[0], test[1], label="splitPixelFull")
plot(ref[0], ref[1], label="ref")
# plot(abs(ref-outMerge)/outMerge, label="ocl_csr_fullsplit")
legend()
show()
input()

