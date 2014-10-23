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
import scipy
#logger = utilstest.getLogger("profile")


ai = pyFAI.load("testimages/halfccd.poni")
data = fabio.open("testimages/halfccd.edf").data

workgroup_size = 256
bins = (100,36)

pos_in = ai.array_from_unit(data.shape, "corner", unit="2th_deg")

pos = pos_in.reshape(pos_in.size/8,4,2)

pos_size = pos.size
#size = data.size
size = pos_size/8



boo = splitPixelFullLUT.HistoLUT2dFullSplit(pos,bins, unit="2th_deg")

foo = boo.integrate(data)

#ref = ai.integrate2d(data,bins=bins,unit="2th_deg", correctSolidAngle=False, method="lut")

#assert(numpy.allclose(ref[1],outMerge))
plot(foo[0])
#plot(ref[0],outMerge, label="ocl_lut_merge")
#plot(ref[0],outData, label="ocl_lut_data")
#plot(ref[0],outCount, label="ocl_lut_count")

#plot(out[0], out[1], label="ocl_lut_merge")
#plot(out[0], out[2], label="ocl_lut_data")
#plot(out[0], out[3], label="ocl_lut_count")

#plot(ref[0], ref[1], label="ref_merge")
#plot(ref[0], ref[2], label="ref_data")
#plot(ref[0], ref[3], label="ref_count")
####plot(abs(ref-outMerge)/outMerge, label="ocl_csr_fullsplit")
#legend()
show()



