#!/usr/bin/python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/pyFAI/pyFAI
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

from __future__ import absolute_import, division, print_function

__doc__ = "test suite for masked arrays"

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "29/01/2016"


import unittest
import numpy
import logging
import sys
import fabio
import time
from .utilstest import UtilsTest, getLogger
logger = getLogger(__file__)

if logger.getEffectiveLevel() <= logging.INFO:
    import pylab
from .. import spline
from ..ext import _bispev
try:
    import six
except ImportError:
    from pyFAI.third_party import six

try:
    from scipy.interpolate import fitpack
except:
    fitpack = None


class TestBispev(unittest.TestCase):
    spinefile = "1461/halfccd.spline"

    def setUp(self):
        """Download files"""
        self.splineFile = UtilsTest.getimage(self.__class__.spinefile)
        self.spline = spline.Spline(self.splineFile)
        self.spline.spline2array(timing=True)

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.spline = self.splineFile = None

    def test_bispev(self):
        x_1d_array = numpy.arange(self.spline.xmin, self.spline.xmax + 1)
        y_1d_array = numpy.arange(self.spline.ymin, self.spline.ymax + 1)
        t0 = time.time()
        dx_ref = fitpack.bisplev(
                x_1d_array, y_1d_array, [self.spline.xSplineKnotsX,
                                         self.spline.xSplineKnotsY,
                                         self.spline.xSplineCoeff,
                                         self.spline.splineOrder,
                                         self.spline.splineOrder],
                dx=0, dy=0)
        t1 = time.time()
        logger.debug(self.spline.xSplineKnotsX.dtype)
        logger.debug(self.spline.xSplineKnotsY.dtype)
        logger.debug(self.spline.xSplineCoeff.dtype)
        dx_loc = _bispev.bisplev(
                x_1d_array, y_1d_array, [self.spline.xSplineKnotsX,
                                         self.spline.xSplineKnotsY,
                                         self.spline.xSplineCoeff,
                                         self.spline.splineOrder,
                                         self.spline.splineOrder],
                )
        t2 = time.time()
        logger.debug("Scipy timings: %.3fs\t cython timings: %.3fs" % (t1 - t0, t2 - t1))
        logger.debug("%s, %s" % (dx_ref.shape, dx_loc.shape))
        logger.debug(dx_ref)
        logger.debug(dx_loc)
        logger.debug("delta = %s" % abs(dx_loc - dx_ref).max())
        if logger.getEffectiveLevel() == logging.DEBUG:
            fig = pylab.figure()
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax1.imshow(dx_ref)
            ax2.imshow(dx_loc)
            fig.show()
            six.moves.input()
        self.assert_(abs(dx_loc - dx_ref).max() < 2e-5, "Result are similar")


def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(TestBispev("test_bispev"))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
    UtilsTest.clean_up()
