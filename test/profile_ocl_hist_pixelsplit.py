# -*- coding: utf-8 -*-
"""
Created on Fri Mar 07 09:52:51 2014

@author: ashiotis
"""

import sys, numpy, time
import utilstest
import fabio, pyopencl
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
ref = ai.xrpd_LUT(data, 1000)[1]
#obt = ai.xrpd_LUT_OCL(data, 1000)[1]

#ref = ai.integrate1d(data, 1000, method="ocl_csr", unit="2th_deg")[0]
pos_in = ai.array_from_unit(data.shape, "corner", unit="2th_deg")

pos = pos_in.reshape(pos_in.size/8,4,2)

foo = splitPixelFullLUT.HistoLUT1dFullSplit(pos, 1000, unit="2th_deg")

boo = foo.integrate(data)[1]

foo2 = ocl_hist_pixelsplit.OCL_Hist_Pixelsplit(pos, 1000, data.size, devicetype="cpu", block_size=32)
boo2 = foo2.integrate(data)[2]

#plot(ref, label="ocl_csr")
#plot(boo, label="csr_fullsplit")
plot(boo2, label="ocl_csr_fullsplit")
legend()
show()
raw_input()

