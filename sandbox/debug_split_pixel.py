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
import pyFAI
import numpy
from pyFAI.third_party import six
img = numpy.zeros((512,512))
for i in range(1,6):img[i*100,i*100]=1
det = pyFAI.detectors.Detector(1e-4,1e-4)
det.shape=(512,512)
ai=pyFAI.AzimuthalIntegrator(1,detector=det)
import pylab
from pyFAI.utils import mathutil
results = {}
for i, meth in enumerate(["cython", "splitbbox", "splitpixel", "csr_no", "csr_bbox", "csr_full"]):
    tth, I = ai.integrate1d(img, 10000, method=meth, unit="2th_deg")
    pylab.plot(tth, I + i * 1e-3, label=meth)
    ai.reset()
    results[meth]=tth, I
print("no_split R=%.3f" % mathutil.rwp(results["csr_no"], results["cython"]))
print("split_bbox R=%.3f" % mathutil.rwp(results["csr_bbox"], results["splitbbox"]))
print("split_full R=%.3f" % mathutil.rwp(results["csr_full"], results["splitpixel"]))
pylab.legend()
pylab.ion()
pylab.show()
six.moves.input("enter_to_quit")
