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
tth = ai._ttha.ravel().astype(numpy.float32)
dtth = ai._dttha.ravel().astype(numpy.float32)
data = img.data.ravel().astype(numpy.float32)

import splitBBox
t0 = time.time()
ra, rb, rc, rd = splitBBox.histoBBox1d(data, tth, dtth, bins=bins)
t1 = time.time()
ref_time = 1000 * (t1 - t0)
print("ref time: %.3fs" % ref_time)


try:
    from pyFAI import ocl_azim
    integr_OCL = ocl_azim.Integrator1d()
    integr_OCL.init()
    integr_OCL.getConfiguration(tth.size, bins)
    integr_OCL.configure()
    integr_OCL.loadTth(tth, dtth, max(0, (tth - dtth).min()) , (tth + dtth).max())
    t0o = time.time()
    a, b, c = integr_OCL.execute(data)
    t1o = time.time()
    print("OpenCL(fw) time: %.3fms" % (1000 * (t1o - t0o)))
except:
    print("Original implementation of OpenCL pyFAI failed")
else:
    print(abs(ra - a).max(), abs(rd - b).max())
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
ct = 1000 * (t1 - t0)
integ_time = 1000 * (t2 - t1)
print("LUT creation: %.3fms; integration %.3fms" % (ct, integ_time))
print abs(ra - a).max(), abs(rb - b).max(), abs(rc - c).max(), abs(rd - d).max()
t1 = time.time()
a, b, c, d = integ.integrate(data)
t2 = time.time()
print "LUT speed-up:", ref_time / (integ_time)
import pylab
#plot(ee)
pylab.plot(a, b, label="LUT")
pylab.plot(ra, rb, label="Original")

import pyopencl

mf = pyopencl.mem_flags
ct = pyopencl.channel_type
co = pyopencl.channel_order
ctx = pyopencl.create_some_context()
q = pyopencl.CommandQueue(ctx)
program = pyopencl.Program(ctx, open("../openCL/ocl_azim_LUT.cl").read()).build()
t3 = time.time()
weights_buf = pyopencl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
#weights_img = pyopencl.image_from_array(ctx, ary=img.data.astype(numpy.float32), mode="r", norm_int=False, num_channels=1)
#print co.INTENSITY, ct.FLOAT,
#imf = pyopencl.ImageFormat(numpy.uint32(co.INTENSITY), numpy.uint32(ct.FLOAT))
#weights_img = pyopencl.Image(ctx, flags=mf.READ_ONLY | mf.COPY_HOST_PTR,
#                             format=imf,
#                             hostbuf=data,
#                             pitches=(img.data.shape[-1],))
#image_from_array(ctx, ary=img.data.astype(numpy.float32), mode="r", norm_int=False, num_channels=1)

#lut_idx_buf = pyopencl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=integ.lut_idx.astype(numpy.uint32))
#lut_coef_buf = pyopencl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=integ.lut_coef)
lut_buf = pyopencl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=integ.lut)
None_buf = pyopencl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=numpy.zeros(1, dtype=numpy.float32))
outData_buf = pyopencl.Buffer(ctx, mf.WRITE_ONLY, numpy.dtype(numpy.float32).itemsize * bins)
outCount_buf = pyopencl.Buffer(ctx, mf.WRITE_ONLY, numpy.dtype(numpy.float32).itemsize * bins)
outMerge_buf = pyopencl.Buffer(ctx, mf.WRITE_ONLY, numpy.dtype(numpy.float32).itemsize * bins)
args = (#weights_img, numpy.uint32(img.dim1), numpy.uint32(img.dim0),
        weights_buf,
                       numpy.uint32(2048),
                       numpy.uint32(integ.lut_size),
#                       lut_idx_buf,
#                       lut_coef_buf,
                       lut_buf,
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
print len(args)
program.lut_integrate_single(q, (bins,), (16,), *args)
b = numpy.empty(bins, dtype=numpy.float32)
c = numpy.empty(bins, dtype=numpy.float32)
d = numpy.empty(bins, dtype=numpy.float32)
pyopencl.enqueue_copy(q, c, outData_buf)
pyopencl.enqueue_copy(q, d, outCount_buf)
pyopencl.enqueue_copy(q, b, outMerge_buf).wait()
t5 = time.time()
pylab.plot(a, b, label="OpenCL")

print "OpenCL speed-up: %s setup: %.2fms \texec: %.2fms" % (0.001 * ref_time / (t5 - t3), 1000 * (t4 - t3), 1000 * (t5 - t4))
print abs(ra - a).max(), abs(rb - b).max(), abs(rc - c).max(), abs(rd - d).max()
for i in range(10):
    j = 2 ** i
    st = time.time()
    program.lut_integrate_single(q, (bins,), (j,), * args)
    pyopencl.enqueue_copy(q, b, outMerge_buf).wait()
    print("Size: %s \ttime: %.2fms" % (j, 1000 * (time.time() - st)))

#plot(ee)
#pylab.plot(a, b, label="OpenCL")
pylab.legend()
pylab.show()
raw_input("Enter")
