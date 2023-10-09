#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2023 European Synchrotron Radiation Facility, Grenoble, France
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

"""Test suite for unit utils"""

__author__ = "Valentin Valls"
__contact__ = "valentin.valls@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "04/11/2023"

import unittest
import logging
import numpy

from .. import validators
from .... import units
from .. import unitutils

_logger = logging.getLogger(__name__)


class TestUnitUtils(unittest.TestCase):
    def testFrom2ThRad__Q_A(self):
        result = unitutils.from2ThRad(0.1, units.Q_A, wavelength=1.03321e-10)
        self.assertAlmostEqual(result, 0.6078, places=3)

    def testFrom2ThRad__D_A(self):
        result = unitutils.from2ThRad(0.1, units.D_A, wavelength=1.03321e-10)
        self.assertAlmostEqual(result, 10.3364, places=3)

    def testFrom2ThRad__Q_NM(self):
        result = unitutils.from2ThRad(0.1, units.Q_NM, wavelength=1.03321e-10)
        self.assertAlmostEqual(result, 6.0786, places=3)

    def testFrom2ThRad__RecD2_NM(self):
        result = unitutils.from2ThRad(0.1, units.RecD2_NM, wavelength=1.03321e-10)
        self.assertAlmostEqual(result, 0.9359, places=3)

    def testFrom2ThRad__RecD2_A(self):
        result = unitutils.from2ThRad(0.1, units.RecD2_A, wavelength=1.03321e-10)
        self.assertAlmostEqual(result, 0.0093596, places=6)

    def testFrom2ThRad__R_MM(self):
        result = unitutils.from2ThRad(
            0.1, units.R_MM, wavelength=1.03321e-10, directDist=1
        )
        self.assertAlmostEqual(result, 0.1003346, places=6)

    def testFrom2ThRad__R_M(self):
        result = unitutils.from2ThRad(
            0.1, units.R_M, wavelength=1.03321e-10, directDist=1
        )
        self.assertAlmostEqual(result, 0.0001003, places=6)


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestUnitUtils))
    return testsuite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
