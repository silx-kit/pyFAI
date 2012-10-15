#!/usr/bin/python
import os, time, numpy
import pyFAI, fabio

root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test", "testimages")
spline = os.path.join(root, "halfccd.spline")
poni = os.path.join(root, "LaB6.poni")
bins = 2048
res = []
with open(poni, "r") as f:
    for l in f:
        if l.startswith("SplineFile"):
            res.append("SplineFile: %s%s" % (spline, os.linesep))
        else:
            res.append(l)
with open(poni, "w") as f:
    f.writelines(res)
edf = os.path.join(root, "LaB6_0020.edf")

img = fabio.open(edf)
ai = pyFAI.load(poni)
ai.xrpd(img.data, bins)
tth = ai._ttha.ravel().astype("float32")
dtth = ai._dttha.ravel().astype("float32")
data = img.data.ravel().astype("float32")

import splitBBox
t0 = time.time()
ra, rb, rc, rd = splitBBox.histoBBox1d(data, tth, dtth, bins=bins)
t1 = time.time()
ref_time = t1 - t0
print("ref time: %.3fs" % ref_time)

#import paraSplitBBox
#t0 = time.time()
#a, b, c, d = paraSplitBBox.histoBBox1d(data, tth, dtth, bins=2048)
#t1 = time.time()
#psbb_time = t1 - t0
#print("Parallel Split Bounding Box: %.3fs" % ref_time)
#print abs(ra - a).max(), abs(rb - b).max(), abs(rc - c).max(), abs(rd - d).max()

print "With LUT"
import splitBBoxLUT
#a, b, c, d, ee = splitBBoxLUT.histoBBox1d(data, tth, dtth, bins=2048)
#print "LUT max =", ee.max()
t0 = time.time()
integ = splitBBoxLUT.HistoBBox1d(tth, dtth, bins=bins)
t1 = time.time()
a, b, c, d = integ.integrate(data)
t2 = time.time()
print("LUT creation: %.3fs; integration %.3f" % (t1 - t0, t2 - t1))
print abs(ra - a).max(), abs(rb - b).max(), abs(rc - c).max(), abs(rd - d).max()
t1 = time.time()
a, b, c, d = integ.integrate(data)
t2 = time.time()
print "speed-up:", ref_time / (t2 - t1)
import pylab
#plot(ee)
pylab.plot(a, b, label="LUT")
pylab.plot(ra, rb, label="Original")

import pyopencl

mf = pyopencl.mem_flags
ctx = pyopencl.create_some_context()
q = pyopencl.CommandQueue(ctx)
program = pyopencl.Program(ctx, open("../openCL/ocl_azim_LUT.cl").read()).build()
t3 = time.time()
weights_buf = pyopencl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
lut_idx_buf = pyopencl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=integ.lut_idx.astype(numpy.uint32))
lut_coef_buf = pyopencl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=integ.lut_coef)
None_buf = pyopencl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=numpy.zeros(1, dtype=numpy.float32))
outData_buf = pyopencl.Buffer(ctx, mf.WRITE_ONLY, numpy.dtype("float32").itemsize * bins)
outCount_buf = pyopencl.Buffer(ctx, mf.WRITE_ONLY, numpy.dtype("float32").itemsize * bins)
outMerge_buf = pyopencl.Buffer(ctx, mf.WRITE_ONLY, numpy.dtype("float32").itemsize * bins)
args = (weights_buf,
                       numpy.uint32(2048),
                       numpy.uint32(integ.lut_size),
                       lut_idx_buf,
                       lut_coef_buf,
                       numpy.int32(0),
                       numpy.float32(0),
                       numpy.float32(0),
                       numpy.int32(0),
                       None_buf,
                       numpy.int32(0),
                       None_buf,
                       outData_buf,
                       outCount_buf,
                       outMerge_buf)
t4 = time.time()
program.lut_integrate(q, (bins,), (64,), *args)
b = numpy.empty(bins, dtype=numpy.float32)
c = numpy.empty(bins, dtype=numpy.float32)
d = numpy.empty(bins, dtype=numpy.float32)
pyopencl.enqueue_read_buffer(q, outData_buf, c).wait()
pyopencl.enqueue_read_buffer(q, outCount_buf, d).wait()
pyopencl.enqueue_read_buffer(q, outMerge_buf, b).wait()
t5 = time.time()
print "OpenCL speed-up: %s setup:%s exec %s" % (ref_time / (t5 - t3), t4 - t3, t5 - t4)
print abs(ra - a).max(), abs(rb - b).max(), abs(rc - c).max(), abs(rd - d).max()
for i in range(10):
    j = 2 ** i
    st = time.time()
    program.lut_integrate(q, (bins,), (j,), *args)
    pyopencl.enqueue_read_buffer(q, outMerge_buf, b).wait()
    print("Size: %s \ttime: %.2fms" % (j, 1000 * (time.time() - st)))

#plot(ee)
pylab.plot(a, b, label="OpenCL")
pylab.legend()
pylab.show()
raw_input("Enter")
