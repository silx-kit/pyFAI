#!/usr/bin/python
import os, sys
import numpy as np
import ocl_azim
from matplotlib import pylab
import time
import pyFAI, fabio
data = fabio.open("../test/testimages/LaB6_0020.edf").data.astype("float32")
ai = pyFAI.load("../test/testimages/halfccd.poni")

tth = ai.twoThetaArray(data.shape)
dtth = ai.delta2Theta(data.shape)
N = 1024 * 2048
Nbins = 1024

solid = np.ones(N, dtype=np.float32)

fhistf = np.zeros(Nbins, dtype=np.float32)
fbinf = np.zeros(Nbins, dtype=np.float32)

tth_min = 0# max(0.0, float(tth.min()))
tth_max = 0.5#float(tth.max())
#print tth_max

mask = np.zeros(N, dtype=np.int)

a = ocl_azim.Integrator1d()
a.init("gpu", False)
a.getConfiguration(N, Nbins)
a.configure()
a.loadTth(tth, dtth)#, 0, tth_max)
a.setSolidAngle(solid)
a.setRange(tth_min, tth_max)
a.setMask(mask)
t0 = time.time()
a_tth, a_xrpd, a_count = a.execute(data)
print "opencl", (time.time() - t0) * 1e3, "ms"

pylab.plot(a_tth)
#pylab.plot(a_xrpd)
#pylab.plot(a_count)

import pyFAI.ocl_azim_pyocl
b = pyFAI.ocl_azim_pyocl.Integrator1d(sys.stdout)
b.init("gpu", False)
b.getConfiguration(N, Nbins)
b.configure()
b.loadTth(tth, dtth)#, tth_min, tth_max)
b.setRange(tth_min, tth_max)
b.setSolidAngle(solid)
b.setMask(mask)
t0 = time.time()
b_tth, b_xrpd, b_count = b.execute(data)
print "pyopencl", (time.time() - t0) * 1e3

print "Max error on 2theta", abs(a_tth - b_tth).max()
print "Max error on xrpd", abs(a_xrpd - b_xrpd).max()
print "Max error on count", abs(a_count - b_count).max()

#optionally
pylab.plot(b_tth)
pylab.plot(a_tth - b_tth)
#pylab.plot(b_xrpd)
#pylab.plot(b_count)
pylab.show()
