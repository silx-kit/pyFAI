#!/usr/bin/python
# coding: utf-8
# author: Jérôme Kieffer
#
#    Project: Fast Azimuthal integration
#             https://github.com/pyFAI/pyFAI
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

import sys, os
import numpy
import fabio
from utilstest import  UtilsTest
pyFAI = UtilsTest.pyFAI
data = fabio.open(UtilsTest.getimage("1788/moke.tif")).data
ai = pyFAI.AzimuthalIntegrator.sload("moke.poni")
ai.xrpd(data, 1000)
tth = ai.twoThetaArray(data.shape)
dtth = ai.delta2Theta(data.shape)
o1 = ai.xrpd(data, 1000)
o2 = ai.xrpd(data, 1000, tthRange=[3.5, 12.5])
o3 = ai.xrpd(data, 1000, chiRange=[10, 80])
o4 = ai.xrpd2(data, 100, 36, tthRange=[3.5, 12.5], chiRange=[10, 80])
from pylab import  *
plot(o1[0], o1[1], "b")
plot(o2[0], o2[1], "r")
plot(o3[0], o3[1], "g")
imshow(o4[0])
show()
