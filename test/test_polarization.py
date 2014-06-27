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
"test suite for polarization corrections"

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "19/04/2013"


import unittest
import os
import numpy
import logging, time
import sys
import fabio
from utilstest import UtilsTest, Rwp, getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]

if logger.getEffectiveLevel() <= logging.INFO:
    import pylab

class TestPolarization(unittest.TestCase):
    shape = (13, 13)
    Y, X = numpy.ogrid[-6:7, -6:7]
    rotY =numpy.radians(30.0*Y)
    rotX =numpy.radians(30.0*X)
    tth = numpy.sqrt(rotY ** 2 + rotX ** 2)
    chi = numpy.arctan2(rotY, rotX)
#    print numpy.degrees(tth[6])
#    print numpy.degrees(chi[6])
#    print numpy.degrees(tth[:, 6])
#    print numpy.degrees(chi[:, 6])
    ai = pyFAI.AzimuthalIntegrator(dist=1, pixel1=0.1, pixel2=0.1)
    ai._ttha = tth
    ai._chia = chi

    def testNoPol(self):
        "without polarization correction should be 1"
        self.assert_(abs(self.ai.polarization(factor=None) - numpy.ones(self.shape)).max() == 0, "without polarization correction should be 1")

    def testCircularPol(self):
        "Circular polarization should decay in (1+(cos2θ)^2)/2"
        pol = (1.0 + numpy.cos(self.tth) ** 2) / 2.0
        self.assert_(abs(self.ai.polarization(factor=0) - pol).max() == 0, "with circular polarization correction is independent of chi")
        self.assert_(abs(self.ai.polarization(factor=0, axis_offset=1) - pol).max() == 0, "with circular polarization correction is independent of chi")

    def testHorizPol(self):
        "horizontal polarization should decay in (cos2θ)**2 in horizontal plane and no correction in vertical one"
        self.assert_(abs(self.ai.polarization(factor=1)[:, 6] - numpy.ones(13)).max() == 0, "No correction in the vertical plane")
        self.assert_(abs(self.ai.polarization(factor=1)[6] - numpy.cos(self.rotX) ** 2).max() < 1e-15, "cos(2th)^2 like in the horizontal plane")

    def testVertPol(self):
        "Vertical polarization should decay in (cos2θ)**2 in vertical plane and no correction in horizontal one"
        self.assert_(abs(self.ai.polarization(factor= -1)[6] - numpy.ones(13)).max() == 0, "No correction in the horizontal plane")
        self.assert_(abs(self.ai.polarization(factor= -1)[:, 6] - (numpy.cos((2 * self.rotX)) + 1) / 2).max() < 1e-15, "cos(2th)^2 like in the verical plane")

    def testoffsetPol(self):
        "test for the rotation of the polarization axis"
        self.assert_(abs(self.ai.polarization(factor=1, axis_offset=numpy.pi / 2)[6] - numpy.ones(13)).max() == 0, "No correction in the horizontal plane")
        self.assert_(abs(self.ai.polarization(factor=1, axis_offset=numpy.pi / 2)[:, 6] - (numpy.cos((2 * self.rotX)) + 1) / 2).max() < 1e-15, "cos(2th)^2 like in the verical plane")


def test_suite_all_Polarization():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestPolarization("testNoPol"))
    testSuite.addTest(TestPolarization("testCircularPol"))
    testSuite.addTest(TestPolarization("testHorizPol"))
    testSuite.addTest(TestPolarization("testVertPol"))
    testSuite.addTest(TestPolarization("testoffsetPol"))
#    testSuite.addTest(TestPolarization("test2th"))

    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_all_Polarization()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
