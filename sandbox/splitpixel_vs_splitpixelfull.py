#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
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
import utilstest
import pyFAI
import numpy
from pyFAI import splitPixelFull, splitPixel

ai = pyFAI.AzimuthalIntegrator(detector="Fairchild")
shape = (2048, 2048)
data = numpy.zeros(shape)
data[100, 200] = 1

tth, I = ai.integrate1d(data, 10000, correctSolidAngle=False, method="splitpixel", unit="2th_deg")

res_splitPixelFull = splitPixelFull.fullSplit1D(ai._corner4Da, data, bins=10000)
res_splitPixel = splitPixel.fullSplit1D(ai._corner4Da, data, bins=10000)

for i, ary in enumerate(("tth", "I", "unweight", "weight")):
    print("Error on %s: %s" % (i, abs(res_splitPixelFull[i] - res_splitPixel[i]).max()))


