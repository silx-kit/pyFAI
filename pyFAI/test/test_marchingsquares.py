#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015 European Synchrotron Radiation Facility, Grenoble, France
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

"test suite for marching_squares / isocontour"
from __future__ import absolute_import, division, print_function

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "19/07/2017"


import unittest
import numpy
import logging
from .utilstest import getLogger
logger = getLogger(__file__)
from ..ext.marchingsquares import isocontour
if logger.getEffectiveLevel() <= logging.INFO:
    import pylab


class TestMarchingSquares(unittest.TestCase):
    def test_isocontour(self):
            ref = 50
            y, x = numpy.ogrid[-100:100:0.1, -100:100:0.1]
            r = numpy.sqrt(x * x + y * y)

            c = isocontour(r, ref)
            self.assertNotEqual(0, len(c), "controur plot contains not point")
            i = numpy.round(c).astype(numpy.int32)
            self.assertTrue(abs(r[(i[:, 0], i[:, 1])] - ref).max() < 0.05, "contour plot not working correctly")
            if logger.getEffectiveLevel() <= logging.DEBUG:
                pylab.imshow(r)
                pylab.plot(c[:, 1], c[:, 0], ",")
                pylab.show()


def suite():
    testsuite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite.addTest(loader(TestMarchingSquares))

    return testsuite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
