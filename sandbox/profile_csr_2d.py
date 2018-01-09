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
from pyFAI.third_party import six
print("#"*50)
pyFAI = sys.modules["pyFAI"]
# from pyFAI import splitBBox
# from pyFAI import splitBBoxLUT
# from pyFAI import splitBBoxCSR
# logger = utilstest.getLogger("profile")
ponifile = utilstest.UtilsTest.getimage("Pilatus1M.poni")
datafile = utilstest.UtilsTest.getimage("Pilatus1M.edf")
ai = pyFAI.load(ponifile)
data = fabio.open(datafile).data
# ref = ai.xrpd_LUT(data, 1000)[1]
# obt = ai.xrpd_LUT_OCL(data, 1000)[1]

ref = ai.integrate2d(data, 100, 360, method="lut", unit="2th_deg")[0]
obt = ai.integrate2d(data, 100, 360, method="ocl_csr", unit="2th_deg")[0]
# #logger.debug("check LUT basics: %s"%abs(obt[1] - ref[1]).max())
assert numpy.allclose(ref, obt)


plot(ref.ravel(), label="ocl_lut")
plot(obt.ravel(), label="ocl_csr")
legend()
show()
six.moves.input()

