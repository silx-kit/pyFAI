#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import absolute_import
from __future__ import print_function

import sys, numpy, time
from pyFAI.utils import mathutil
from . import utilstest
import fabio, pyopencl
from pylab import *
print("#"*50)
pyFAI = sys.modules["pyFAI"]
from pyFAI import splitBBox
#splitBBox = sys.modules["pyFAI.splitBBox"]
ai = pyFAI.load("testimages/Pilatus1M.poni")
data = fabio.open("testimages/Pilatus1M.edf").data
ref = ai.xrpd_LUT(data, 1000)
obt = ai.xrpd_LUT_OCL(data, 1000)
print(abs(obt[1] - ref[1]).max())
lut = ai._lut_integrator.lut
gpu = pyFAI.ocl_azim_lut.OCL_LUT_Integrator(lut, data.size, "GPU")
print(gpu.device)
img = numpy.zeros(data.shape, dtype="float32")
print("ref", (data == -2).sum(), (data == -1).sum())
pyopencl.enqueue_copy(gpu._queue, img, gpu._cl_mem["image"])#.wait()
print("obt", (img == -2).sum(), (img == -1).sum())

out_cyt = ai._lut_integrator.integrate(data)
out_ocl = gpu.integrate(data)[0]
print("NoCorr R=", mathutil.rwp((out_cyt[0], out_ocl), out_cyt[:2], "no corrections"))
nodummy = out_cyt[1]
plot(nodummy + 1, label="no_corr")
out_cyt = ai._lut_integrator.integrate(data, dummy= -2, delta_dummy=1.5)
out_ocl = gpu.integrate(data, dummy= -2, delta_dummy=1.5)[0]
print("Dummy  R=", mathutil.rwp((out_cyt[0], out_ocl), out_cyt[:2], "Dummy"))
#print("nodummy/Dummy", mathutil.rwp((out_cyt[0], out_cyt[1]), (out_cyt[0], nodummy), "nodummy/Dummy")

dark = numpy.random.random(data.shape)
out_cyt = ai._lut_integrator.integrate(data, dark=dark)
out_ocl = gpu.integrate(data, dark=dark)[0]
print("Dark  R=", mathutil.rwp((out_cyt[0], out_ocl), out_cyt[:2], "dark"))

flat = 2 * numpy.ones_like(data)
out_cyt = ai._lut_integrator.integrate(data, flat=flat)
out_ocl = gpu.integrate(data, flat=flat)[0]
print("Flat  R=", mathutil.rwp((out_cyt[0], out_ocl), out_cyt[:2], "flat"))

solidAngle = ai.solidAngleArray(data.shape)
out_cyt = ai._lut_integrator.integrate(data, solidAngle=solidAngle)
out_ocl = gpu.integrate(data, solidAngle=solidAngle)[0]
print("SolidAngle  R=", mathutil.rwp((out_cyt[0], out_ocl), out_cyt[:2], "SolidAngle"))

polarization = ai.polarization(data.shape, 0.95)
out_cyt = ai._lut_integrator.integrate(data, polarization=polarization)
out_ocl = gpu.integrate(data, polarization=polarization)[0]
print("PolarizationR=", mathutil.rwp((out_cyt[0], out_ocl), out_cyt[:2], "Polarization"))

#pyopencl.enqueue_copy(gpu._queue, img, gpu._cl_mem["image"]).wait()
#xx = splitBBox.histoBBox1d(weights=data,
#                                                 pos0=ai._ttha,
#                                                 delta_pos0=ai._dttha,
#                                                 bins=1000,
#                                                 polarization=polarization)[1]
#plot(xx + 2, label="xrpd")
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
print("Timings With dummy", 1000 * (time.time() - t0))
t0 = time.time()
out = gpu.integrate(data)
print("Timings Without dummy", 1000 * (time.time() - t0))
yscale("log")
show()
