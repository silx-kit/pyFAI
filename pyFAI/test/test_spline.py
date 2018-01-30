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
__date__ = "12/01/2018"


import unittest
import logging
logger = logging.getLogger(__name__)
import pyFAI.spline
from . import utilstest
import itertools


class TestSpline(utilstest.ParametricTestCase):

    def test_tilt_coverage(self):
        """
        Some tests ....
        """
        spline_file = utilstest.UtilsTest.getimage("frelon.spline")

        default_center = (0, 0)
        default_distance = 0
        default_tilt = 0
        default_rotation_tilt = 0

        test_cases = [
            (default_center, default_distance, default_tilt, default_rotation_tilt),
            ((1000, 1000), default_distance, default_tilt, default_rotation_tilt),
            (default_center, 1, default_tilt, default_rotation_tilt),
            (default_center, 10, default_tilt, default_rotation_tilt),
            (default_center, 1000, default_tilt, default_rotation_tilt),
            (default_center, default_distance, 1, default_rotation_tilt),
            (default_center, default_distance, 10, default_rotation_tilt),
            (default_center, default_distance, default_tilt, 10),
            (default_center, default_distance, default_tilt, 90),
            (default_center, default_distance, default_tilt, 180),
        ]

        for center, distance, tilt, rotation_tilt in test_cases:
            with self.subTest(center=center, distance=distance, tilt=tilt, rotation_tilt=rotation_tilt):
                print(center, distance, tilt, rotation_tilt)
                spline = pyFAI.spline.Spline()
                spline.read(spline_file)
                logger.info("Original Spline: %s", spline)
                spline.spline2array(timing=True)
                _tilted = spline.tilt(center, tilt, rotation_tilt, distance, timing=True)
                # tilted.writeEDF("%s-tilted-t%i-p%i-d%i" %
                #                  (os.path.splitext(spline_file)[0],
                #                  tilt, rotation_tilt, distance))


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestSpline))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
