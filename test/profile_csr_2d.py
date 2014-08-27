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
#from pyFAI import splitBBox
#from pyFAI import splitBBoxLUT
#from pyFAI import splitBBoxCSR
#logger = utilstest.getLogger("profile")


ai = pyFAI.load("testimages/Pilatus1M.poni")
data = fabio.open("testimages/Pilatus1M.edf").data
#ref = ai.xrpd_LUT(data, 1000)[1]
#obt = ai.xrpd_LUT_OCL(data, 1000)[1]

ref = ai.integrate2d(data, 100, 360, method="lut", unit="2th_deg")[0]
obt = ai.integrate2d(data, 100, 360, method="ocl_csr", unit="2th_deg")[0]
##logger.debug("check LUT basics: %s"%abs(obt[1] - ref[1]).max())
assert numpy.allclose(ref,obt)


plot(ref.ravel(), label="ocl_lut")
plot(obt.ravel(), label="ocl_csr")
legend()
show()
raw_input()

