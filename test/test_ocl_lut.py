#!/usr/bin/python
import sys, numpy, time
import utilstest
import fabio, pyopencl
from pylab import *
print "#"*50
pyFAI = sys.modules["pyFAI"]
from pyFAI import splitBBox
#splitBBox = sys.modules["pyFAI.splitBBox"]
ai = pyFAI.load("testimages/Pilatus1M.poni")
data = fabio.open("testimages/Pilatus1M.edf").data
ref = ai.xrpd_LUT(data, 1000)
obt = ai.xrpd_LUT_OCL(data, 1000)
print abs(obt[1] - ref[1]).max()
lut = ai._lut_integrator.lut
gpu = pyFAI.ocl_azim_lut.OCL_LUT_Integrator(lut, data.size, "GPU")
print gpu.device
img = numpy.zeros(data.shape, dtype="float32")
print "ref", (data == -2).sum(), (data == -1).sum()
pyopencl.enqueue_copy(gpu._queue, img, gpu._cl_mem["image"])#.wait()
print "obt", (img == -2).sum(), (img == -1).sum()

out_cyt = ai._lut_integrator.integrate(data)
out_ocl = gpu.integrate(data)[0]
print "NoCorr R=", utilstest.Rwp((out_cyt[0], out_ocl), out_cyt[:2], "no corrections")
nodummy = out_cyt[1]
plot(nodummy + 1, label="no_corr")
out_cyt = ai._lut_integrator.integrate(data, dummy= -2, delta_dummy=1.5)
out_ocl = gpu.integrate(data, dummy= -2, delta_dummy=1.5)[0]
print "Dummy  R=", utilstest.Rwp((out_cyt[0], out_ocl), out_cyt[:2], "Dummy")
#print "nodummy/Dummy", utilstest.Rwp((out_cyt[0], out_cyt[1]), (out_cyt[0], nodummy), "nodummy/Dummy")

dark = numpy.random.random(data.shape)
out_cyt = ai._lut_integrator.integrate(data, dark=dark)
out_ocl = gpu.integrate(data, dark=dark)[0]
print "Dark  R=", utilstest.Rwp((out_cyt[0], out_ocl), out_cyt[:2], "dark")

flat = 2 * numpy.ones_like(data)
out_cyt = ai._lut_integrator.integrate(data, flat=flat)
out_ocl = gpu.integrate(data, flat=flat)[0]
print "Flat  R=", utilstest.Rwp((out_cyt[0], out_ocl), out_cyt[:2], "flat")

solidAngle = ai.solidAngleArray(data.shape)
out_cyt = ai._lut_integrator.integrate(data, solidAngle=solidAngle)
out_ocl = gpu.integrate(data, solidAngle=solidAngle)[0]
print "SolidAngle  R=", utilstest.Rwp((out_cyt[0], out_ocl), out_cyt[:2], "SolidAngle")

polarization = ai.polarization(data.shape, 0.95)
out_cyt = ai._lut_integrator.integrate(data, polarization=polarization)
out_ocl = gpu.integrate(data, polarization=polarization)[0]
print "PolarizationR=", utilstest.Rwp((out_cyt[0], out_ocl), out_cyt[:2], "Polarization")

#pyopencl.enqueue_copy(gpu._queue, img, gpu._cl_mem["image"]).wait()
#print "SolidAngle", solidAngle
#print img
#xx = splitBBox.histoBBox1d(weights=data,
#                                                 pos0=ai._ttha,
#                                                 delta_pos0=ai._dttha,
#                                                 bins=1000,
#                                                 polarization=polarization)[1]
#plot(xx + 2, label="xrpd")
#print "Pol: lut/refR=", utilstest.Rwp((out_cyt[0], xx), out_cyt[:2], "Polarization")
#print "Pol: ocl/refR=", utilstest.Rwp((out_cyt[0], out_ocl), (out_cyt[0], xx), "Polarization")
#print "Pol: noc/refR=", utilstest.Rwp((out_cyt[0], nodummy), (out_cyt[0], xx), "Polarization")
#print out_ocl
plot(out_cyt[1], label="ref")
plot(out_ocl, label="obt")

#plot(out, label="out")
#outData = numpy.zeros(1000, "float32")
#outCount = numpy.zeros(1000, "float32")
#outMerge = numpy.zeros(1000, "float32")
#pyopencl.enqueue_copy(gpu._queue, outData, gpu._cl_mem["outData"])#.wait()
#pyopencl.enqueue_copy(gpu._queue, outCount, gpu._cl_mem["outCount"])#.wait()
#pyopencl.enqueue_copy(gpu._queue, outMerge, gpu._cl_mem["outMerge"])#.wait()
#plot(outData, label="outData")
#plot(outCount, label="outCount")
#plot(outMerge, label="outMerge")
legend()
t0 = time.time()
out = gpu.integrate(data, dummy= -2, delta_dummy=1.5)
print "Timings With dummy", 1000 * (time.time() - t0)
t0 = time.time()
out = gpu.integrate(data)
print "Timings Without dummy", 1000 * (time.time() - t0)
yscale("log")
show()
