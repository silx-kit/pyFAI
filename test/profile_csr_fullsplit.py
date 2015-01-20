# -*- coding: utf-8 -*-
"""
Created on Fri Mar 07 09:52:51 2014

@author: ashiotis
"""
from __future__ import absolute_import
from __future__ import print_function

import sys, numpy, time
from . import utilstest
import fabio, pyopencl
from pylab import *
from six.moves import input
print("#"*50)
pyFAI = sys.modules["pyFAI"]
from pyFAI import splitPixelFullLUT
from pyFAI import ocl_azim_csr
#from pyFAI import splitBBoxLUT
#from pyFAI import splitBBoxCSR
#logger = utilstest.getLogger("profile")


ai = pyFAI.load("testimages/Pilatus1M.poni")
data = fabio.open("testimages/Pilatus1M.edf").data
ref = ai.xrpd_LUT(data, 1000)[1]
#obt = ai.xrpd_LUT_OCL(data, 1000)[1]

#ref = ai.integrate1d(data, 1000, method="ocl_csr", unit="2th_deg")[0]

pos = ai.array_from_unit(data.shape, "corner", unit="2th_deg")
foo = splitPixelFullLUT.HistoLUT1dFullSplit(pos, 1000, unit="2th_deg")

boo = foo.integrate(data)[1]

foo2 = ocl_azim_csr.OCL_CSR_Integrator(foo.lut, data.size, "GPU", block_size=32)
boo2 = foo2.integrate(data)[0]

plot(ref, label="ocl_csr")
plot(boo, label="csr_fullsplit")
plot(boo2, label="ocl_csr_fullsplit")
legend()
show()
input()

