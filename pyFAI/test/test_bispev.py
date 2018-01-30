#!/usr/bin/env python
# coding: utf-8
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

from __future__ import absolute_import, division, print_function

"""Test suite for masked arrays"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "10/01/2018"


import unittest
import numpy
import logging
import time
from .utilstest import UtilsTest
logger = logging.getLogger(__name__)

if logger.getEffectiveLevel() <= logging.INFO:
    import pylab
from .. import spline
from ..ext import _bispev
from pyFAI.third_party import six

try:
    from scipy.interpolate import fitpack
except ImportError:
    fitpack = None


class TestBispev(unittest.TestCase):
    spinefile = "halfccd.spline"

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
        dx_ref = fitpack.bisplev(x_1d_array, y_1d_array,
                                 [self.spline.xSplineKnotsX,
                                  self.spline.xSplineKnotsY,
                                  self.spline.xSplineCoeff,
                                  self.spline.splineOrder,
                                  self.spline.splineOrder],
                                 dx=0, dy=0)
        t1 = time.time()
        logger.debug(self.spline.xSplineKnotsX.dtype)
        logger.debug(self.spline.xSplineKnotsY.dtype)
        logger.debug(self.spline.xSplineCoeff.dtype)
        dx_loc = _bispev.bisplev(x_1d_array, y_1d_array,
                                 [self.spline.xSplineKnotsX,
                                  self.spline.xSplineKnotsY,
                                  self.spline.xSplineCoeff,
                                  self.spline.splineOrder,
                                  self.spline.splineOrder])
        t2 = time.time()
        logger.debug("Scipy timings: %.3fs\t cython timings: %.3fs", t1 - t0, t2 - t1)
        logger.debug("%s, %s", dx_ref.shape, dx_loc.shape)
        logger.debug(dx_ref)
        logger.debug(dx_loc)
        logger.debug("delta = %s", abs(dx_loc - dx_ref).max())
        if logger.getEffectiveLevel() == logging.DEBUG:
            fig = pylab.figure()
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax1.imshow(dx_ref)
            ax2.imshow(dx_loc)
            fig.show()
            six.moves.input()
        self.assertTrue(abs(dx_loc - dx_ref).max() < 2e-5, "Result are similar")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestBispev))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
    UtilsTest.clean_up()
