# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:      Giannis Ashiotis
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
try:
    from pyFAI.third_party import six
except (ImportError, Exception):
    import six
print("#"*50)
pyFAI = sys.modules["pyFAI"]
from pyFAI import splitPixelFullLUT
from pyFAI import ocl_azim_csr
# from pyFAI import splitBBoxLUT
# from pyFAI import splitBBoxCSR
# logger = utilstest.getLogger("profile")


ponifile = utilstest.UtilsTest.getimage("Pilatus1M.poni")
datafile = utilstest.UtilsTest.getimage("Pilatus1M.edf")
ai = pyFAI.load(ponifile)
data = fabio.open(datafile).data

ref = ai.xrpd_LUT(data, 1000)[1]
# obt = ai.xrpd_LUT_OCL(data, 1000)[1]

# ref = ai.integrate1d(data, 1000, method="ocl_csr", unit="2th_deg")[0]

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
six.moves.input()
