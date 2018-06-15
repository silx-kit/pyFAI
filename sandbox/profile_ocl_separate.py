#!/usr/bin/env python
# coding: utf-8
#
#    Project: Fast Azimuthal Integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2018 European Synchrotron Radiation Facility, Grenoble, France
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
import cProfile, pstats, StringIO
import unittest
import pyFAI, fabio
aimg = fabio.open("testimages/halfccd.edf").data
ai = pyFAI.load("testimages/halfccd.poni")
a, b = ai.separate(aimg)
pr = cProfile.Profile()
pr.enable()
a, b = ai.separate(aimg)
pr.disable()
pr.dump_stats(__file__ + ".numpy.log")
a, b = ai.separate(aimg, method="ocl_csr")
pr = cProfile.Profile()
pr.enable()
a, b = ai.separate(aimg, method="ocl_csr")
pr.disable()
pr.dump_stats(__file__ + ".opencl.log")
# s = StringIO.StringIO()
# sortby = 'cumulative'
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print s.getvalue()

from pylab import *
f = figure()
f.add_subplot(221)
a, b = ai.separate(aimg)
imshow(log(a))
f.add_subplot(222)
imshow(log(b))
f.add_subplot(223)
a, b = ai.separate(aimg, method="ocl_csr")
imshow(log(a))
f.add_subplot(224)
imshow(log(b))
show()
