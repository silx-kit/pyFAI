#!/usr/bin/python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) 2013-2015 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  .
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  .
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

from __future__ import absolute_import, division, print_function

__doc__ = "test suite for Distortion correction class"
__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "04/05/2016"


import unittest
import numpy
import fabio
from .utilstest import UtilsTest, getLogger
logger = getLogger(__file__)
from .. import detectors
from .. import distortion


class TestHalfCCD(unittest.TestCase):
    """basic test"""
    halfFrelon = "1464/LaB6_0020.edf"
    splineFile = "1461/halfccd.spline"
    fit2d_cor = "2454/halfccd.fit2d.edf"

    def setUp(self):
        """Download files"""
        self.fit2dFile = UtilsTest.getimage(self.__class__.fit2d_cor)
        self.halfFrelon = UtilsTest.getimage(self.__class__.halfFrelon)
        self.splineFile = UtilsTest.getimage(self.__class__.splineFile)
        self.det = detectors.FReLoN(self.splineFile)
        self.dis = distortion.Distortion(self.det, self.det.shape, resize=False,
                                         mask=numpy.zeros(self.det.shape, "int8"))
        self.fit2d = fabio.open(self.fit2dFile).data
        self.raw = fabio.open(self.halfFrelon).data

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.fit2dFile = self.halfFrelon = self.splineFile = self.det = self.dis = self.fit2d = self.raw = None

    def test_size(self):
        self.dis.reset(prepare=False)
        ny = self.dis.calc_size(False)
        self.dis.reset(prepare)
        cy = self.dis.calc_size(True)
        self.assertEqual(abs(ny - cy).max(), 0, "equivalence of the cython and numpy model")

    def test_vs_fit2d(self):
        """
        Compare spline correction vs fit2d's code

        precision at 1e-3 : 90% of pixels
        """
        size = self.dis.calc_LUT_size()
        mem = size.max() * self.raw.nbytes * 4 / 2.0 ** 20
        logger.info("Memory expected for LUT: %.3f MBytes", mem)
        try:
            self.dis.calc_LUT()
        except MemoryError as error:
            logger.warning("TestHalfCCD.test_vs_fit2d failed because of MemoryError. This test tries to allocate %.3fMBytes and failed with %s", mem, error)
            return
        cor = self.dis.correct(self.raw)
        delta = abs(cor - self.fit2d)
        mask = numpy.where(self.fit2d == 0)
        denom = self.fit2d.copy()
        denom[mask] = 1
        ratio = delta / denom
        ratio[mask] = 0
        good_points_ratio = 1.0 * (ratio < 1e-3).sum() / self.raw.size
        logger.info("ratio of good points (less than 1/1000 relative error): %.4f" % good_points_ratio)
        self.assert_(good_points_ratio > 0.99, "99% of all points have a relative error below 1/1000")


class TestImplementations(unittest.TestCase):
    """Ensure equivalence of implementation between numpy & Cython"""
    halfFrelon = "1464/LaB6_0020.edf"
    splineFile = "1461/halfccd.spline"

    def setUp(self):
        """Download files"""
        self.halfFrelon = UtilsTest.getimage(self.__class__.halfFrelon)
        self.splineFile = UtilsTest.getimage(self.__class__.splineFile)
        self.det = detectors.FReLoN(self.splineFile)
        self.det.binning = 5, 8  # larger binning makes python loops faster
        self.dis = distortion.Distortion(self.det, self.det.shape, resize=False,
                                         mask=numpy.zeros(self.det.shape, "int8"))

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.fit2dFile = self.halfFrelon = self.splineFile = self.det = self.dis = self.fit2d = self.raw = None

    def test_size(self):
        self.dis.reset(prepare=False)
        ny = self.dis.calc_size(False)
        self.dis.reset(prepare=False)
        cy = self.dis.calc_size(True)
        delta = abs(ny - cy).sum()
        self.assertEqual(delta, 0, "equivalence of the cython and numpy model, summed error=%s" % delta)


def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(TestImplementations("test_size"))
    testsuite.addTest(TestHalfCCD("test_vs_fit2d"))
#    testsuite.addTest(test_azim_halfFrelon("test_numpy_vs_fit2d"))
#    testsuite.addTest(test_azim_halfFrelon("test_cythonSP_vs_fit2d"))
#    testsuite.addTest(test_azim_halfFrelon("test_cython_vs_numpy"))
#    testsuite.addTest(test_flatimage("test_splitPixel"))
#    testsuite.addTest(test_flatimage("test_splitBBox"))
# This test is known to be broken ...
#    testsuite.addTest(test_saxs("test_mask"))

    return testsuite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
