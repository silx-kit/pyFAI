# -*- coding: utf-8 -*-
"""
Created on Fri Mar 07 09:52:51 2014

@author: ashiotis
"""
from __future__ import absolute_import
from __future__ import print_function

import sys, numpy, time
from pyFAI.test import utilstest
import fabio, pyopencl
from pylab import *
from pyFAI.third_party import six
print("#"*50)
pyFAI = sys.modules["pyFAI"]
from pyFAI import splitBBox
from pyFAI import splitBBoxLUT
from pyFAI import splitBBoxCSR
from pyFAI import ocl_azim_csr
logger = utilstest.getLogger("profile")


ai = pyFAI.load("testimages/Pilatus1M.poni")
data = fabio.open("testimages/Pilatus1M.edf").data
ref = ai.xrpd_LUT(data, 1000)[1]
obt = ai.xrpd_LUT_OCL(data, 1000)[1]
logger.debug("check LUT basics: %s"%abs(obt[1] - ref[1]).max())
assert numpy.allclose(ref,obt)


workgroup_size = 128
print("Workgroup size = ", workgroup_size)


out_cyt_bb = pyFAI.splitBBox.histoBBox1d(data, ai._ttha, ai._dttha, bins=1000)[1]


t0 = time.time()
cyt_lut = pyFAI.splitBBoxLUT.HistoBBox1d(
                 ai._ttha,
                 ai._dttha,
                 bins=1000,
                 unit="2th_deg")
t1 = time.time()
print("Time to create cython lut: ", t1-t0)

t0 = time.time()
cyt_lut.generate_csr()
t1 = time.time()
print("Time to generate CSR from cython lut: ", t1-t0)

t0 = time.time()
cyt_lut.generate_csr_padded(workgroup_size)
t1 = time.time()
print("Time to generate CSR_Padded from cython lut: ", t1-t0)

t0 = time.time()
cyt_csr = pyFAI.splitBBoxCSR.HistoBBox1d(
                 ai._ttha,
                 ai._dttha,
                 bins=1000,
                 unit="2th_deg")
t1 = time.time()
print("Time to create cython CSR: ", t1-t0)

t0 = time.time()
cyt_csr_padded = pyFAI.splitBBoxCSR.HistoBBox1d(
                 ai._ttha,
                 ai._dttha,
                 bins=1000,
                 unit="2th_deg",
                 padding=workgroup_size)
t1 = time.time()
print("Time to create cython CSR_Padded: ", t1-t0)






out_cyt_lut = cyt_lut.integrate(data)[1]


ocl_lut = pyFAI.ocl_azim_lut.OCL_LUT_Integrator(cyt_lut.lut, data.size, "GPU",profile=True)
out_ocl_lut = ocl_lut.integrate(data)[0]
print("")
print("OpenCL LUT on: ", ocl_lut.device)
ocl_lut.log_profile()
print("")
print("================================================================================")
ocl_lut.__del__()


ocl_csr = ocl_azim_csr.OCL_CSR_Integrator(cyt_csr.lut, data.size, "GPU",profile=True, block_size=workgroup_size)
out_ocl_csr = ocl_csr.integrate(data)[0]
print("")
print("ÖpenCL CSR on: ", ocl_csr.device)
ocl_csr.log_profile()
print("")
print("================================================================================")
ocl_csr.__del__()


ocl_csr_padded = ocl_azim_csr.OCL_CSR_Integrator(cyt_csr_padded.lut, data.size, "GPU",profile=True, block_size=workgroup_size)
out_ocl_csr_padded = ocl_csr_padded.integrate(data)[0]
print("")
print("ÖpenCL CSR padded: ", ocl_csr_padded.device)
ocl_csr_padded.log_profile()
print("")
print("================================================================================")
ocl_csr_padded.__del__


#assert numpy.allclose(out_ocl_csr_padded,out_cyt_bb)



plot(out_cyt_bb, label="cyt_bb" )
plot(out_cyt_lut, label="cyt_lut" )
plot(out_ocl_lut, label="ocl_lut")
plot(out_ocl_csr, label="ocl_csr")
plot(out_ocl_csr_padded, label="ocl_csr_padded")
legend()
show()
six.moves.input()

