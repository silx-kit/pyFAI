#!/usr/bin/python
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

tth_min = max(0.0, float(tth.min()))
tth_max = float(tth.max())

mask = np.zeros(N, dtype=np.int)

a = ocl_azim.Integrator1d()
a.init("gpu", False)
a.getConfiguration(N, Nbins)
a.configure()
a.loadTth(tth, dtth, tth_min, tth_max)
a.setSolidAngle(solid)
a.setMask(mask)
t0 = time.time()
xrpd, count, weight = a.execute(data)
print "opencl", (time.time() - t0) * 1e3, "ms"

import pyFAI.ocl_azim_pyocl
b = pyFAI.ocl_azim_pyocl.Integrator1d()
b.init("gpu", False)
b.getConfiguration(N, Nbins)
b.configure()
b.loadTth(tth, dtth, tth_min, tth_max)
b.setSolidAngle(solid)
b.setMask(mask)
t0 = time.time()
xrpd, count, weight = a.execute(data)
print "pyopencl", (time.time() - t0) * 1e3

#optionally
pylab.plot(xrpd)
pylab.plot(count)
pylab.plot(weight)
pylab.show()
