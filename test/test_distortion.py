#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/kif/pyFAI
#
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"test suite for Distortion correction class"

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "23/02/2013"


import unittest
# import os
import numpy
# import logging, time
import sys
import fabio

from utilstest import UtilsTest, getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]
from pyFAI import _distortion, detectors
#_distortion = sys.modules["pyFAI._distortion"]
#detectors = sys.modules["pyFAI.detectors"]

class test_halfccd(unittest.TestCase):
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
        self.dis = _distortion.Distortion(self.det)
        self.fit2d = fabio.open(self.fit2dFile).data
        self.raw = fabio.open(self.halfFrelon).data

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
            logger.warning("test_halfccd.test_vs_fit2d failed because of MemoryError. This test tries to allocate %.3fMBytes and failed with %s", mem, error)
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

def test_suite_all_distortion():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_halfccd("test_vs_fit2d"))
#    testSuite.addTest(test_azim_halfFrelon("test_numpy_vs_fit2d"))
#    testSuite.addTest(test_azim_halfFrelon("test_cythonSP_vs_fit2d"))
#    testSuite.addTest(test_azim_halfFrelon("test_cython_vs_numpy"))
#    testSuite.addTest(test_flatimage("test_splitPixel"))
#    testSuite.addTest(test_flatimage("test_splitBBox"))
# This test is known to be broken ...
#    testSuite.addTest(test_saxs("test_mask"))

    return testSuite

if __name__ == '__main__':

    mysuite = test_suite_all_distortion()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
