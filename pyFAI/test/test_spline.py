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

"""Test suites for sparse matrix multiplication modules"""

from __future__ import absolute_import, division, print_function

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "03/07/2018"


import unittest
import logging
logger = logging.getLogger(__name__)
import pyFAI.spline
from . import utilstest


class TestSpline(utilstest.ParametricTestCase):

    def test_tilt_coverage(self):
        """
        Some tests ....
        """
        spline_file = utilstest.UtilsTest.getimage("frelon.spline")

        default_center = (1000, 1500)
        default_distance = 500
        default_tilt = 1
        default_rotation_tilt = 45

        test_cases = [
            (default_center, default_distance, default_tilt, default_rotation_tilt),
        ]

        for center, distance, tilt, rotation_tilt in test_cases:
            with self.subTest(center=center, distance=distance, tilt=tilt, rotation_tilt=rotation_tilt):
                spline = pyFAI.spline.Spline()
                spline.read(spline_file)
                logger.info("Original Spline: %s", spline)
                spline.spline2array(timing=True)
                _tilted = spline.tilt(center, tilt, rotation_tilt, distance, timing=True)
                # As there is not assesement, just validate the execution of the code
                # tilted.writeEDF("%s-tilted-t%i-p%i-d%i" %
                #                  (os.path.splitext(spline_file)[0],
                #                  tilt, rotation_tilt, distance))

    def test_half_ccd(self):
        "Test the half_ccd back and forth"
        spline_file = utilstest.UtilsTest.getimage("halfccd.spline")
        spline = pyFAI.spline.Spline(spline_file)
        logger.debug("xmin %s, xmax %s, ymin %s, ymax",
                     spline.xmin, spline.xmax, spline.ymin, spline.ymax)
        spline.spline2array()
        logger.debug("delta_x: %s", spline.xDispArray.shape)
        logger.debug("delta_y: %s", spline.yDispArray.shape)
        new_spline = spline.flipud(False).fliplr(False).fliplrud(False)
        new_spline.array2spline(smoothing=0.1)
        self.assertLess(abs(new_spline.xDispArray - spline.xDispArray).max(), 1e-6, "X data are OK")
        self.assertLess(abs(new_spline.yDispArray - spline.yDispArray).max(), 1e-6, "Y data are OK")
        self.assertLess(abs(spline.xSplineKnotsX - new_spline.xSplineKnotsX).max(), 1e-6, "xSplineKnotsX data are OK")
        self.assertLess(abs(spline.xSplineKnotsY - new_spline.xSplineKnotsY).max(), 1e-6, "xSplineKnotsY data are OK")
        self.assertLess(abs(spline.xSplineCoeff - new_spline.xSplineCoeff).max(), 1e-6, "xSplineCoeff data are OK")
        self.assertLess(abs(spline.ySplineKnotsX - new_spline.ySplineKnotsX).max(), 1e-6, "ySplineKnotsX data are OK")
        self.assertLess(abs(spline.ySplineKnotsY - new_spline.ySplineKnotsY).max(), 1e-6, "ySplineKnotsY data are OK")
        self.assertLess(abs(spline.ySplineCoeff - new_spline.ySplineCoeff).max(), 1e-6, "ySplineCoeff data are OK")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestSpline))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
