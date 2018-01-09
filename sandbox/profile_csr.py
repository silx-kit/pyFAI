# -*- coding: utf-8 -*-
"""
Created on Fri Mar 07 09:52:51 2014

@author: ashiotis
"""
from __future__ import absolute_import, division, print_function

import sys, numpy, time
import utilstest
import fabio, pyopencl
from pylab import *
print("#"*50)
pyFAI = sys.modules["pyFAI"]
from pyFAI import splitBBox
from pyFAI import splitBBoxLUT
from pyFAI import splitBBoxCSR
from pyFAI.third_party import six
logger = utilstest.getLogger("profile")


ai = pyFAI.load("testimages/Pilatus1M.poni")
data = fabio.open("testimages/Pilatus1M.edf").data
ref = ai.xrpd_LUT(data, 1000)[1]
obt = ai.xrpd_LUT_OCL(data, 1000)[1]
logger.debug("check LUT basics: %s"%abs(obt[1] - ref[1]).max())
assert numpy.allclose(ref,obt)

cyt_lut = pyFAI.splitBBoxLUT.HistoBBox1d(
                 ai._ttha,
                 ai._dttha,
                 bins=1000,
                 unit="2th_deg")

ocl_lut = pyFAI.ocl_azim_lut.OCL_LUT_Integrator(cyt_lut.lut, data.size, "GPU",profile=True)

print("OpenCL Device", ocl_lut.device)


cyt_csr = pyFAI.splitBBoxCSR.HistoBBox1d(
                 ai._ttha,
                 ai._dttha,
                 bins=1000,
                 unit="2th_deg")

out_cyt_lut = cyt_lut.integrate(data)[1]
out_ocl_lut = ocl_lut.integrate(data)[0]
#out_ocl_csr = ocl_csr.integrate(data)[0]
out_cyt_csr = cyt_csr.integrate(data)[1]
print("lut cpu vs lut gpu",abs(out_cyt_lut - out_ocl_lut).max())
assert numpy.allclose(out_cyt_lut, out_ocl_lut)
print("lut cpu vs csr cpu",abs(out_cyt_lut - out_cyt_csr).max())
#assert numpy.allclose(out_cyt_lut, out_cyt_csr)


ocl_lut.log_profile()

plot(out_cyt_lut, label="cyt_lut" )
plot(out_ocl_lut, label="ocl_lut")
plot(out_cyt_csr, label="cyt_csr" )
#plot(out_ocl-out_cyt, label="delta")
legend()
show()
six.moves.input()

