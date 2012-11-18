#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://forge.epn-campus.eu/projects/azimuthal
#
#    File: "$Id$"
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
"test suite for masked arrays"

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "16/11/2012"


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

class TestSaxs(unittest.TestCase):


    def setUp(self):
        self.img = UtilsTest.getimage("1883/Pilatus1M.edf")
        self.data = fabio.open(self.img).data
        self.ai = pyFAI.AzimuthalIntegrator(1.58323111834, 0.0334170169115, 0.0412277798782, 0.00648735642526, 0.00755810191106, 0.0, detector=pyFAI.detectors.Pilatus1M())
        self.ai.wavelength = 1e-10
#        self.ai.mask=None

    def tearDown(self):
        pass

    def testMask(self):
        assert self.ai.mask.sum() == 73533

    def testNumpy(self):
        qref, Iref, s = self.ai.saxs(self.data, 1000)
        q, I, s = self.ai.saxs(self.data, 1000, error_model="poisson", method="numpy")
        self.q = q
        self.I = I
        self.s = s
        assert q[0] > 0
        assert q[-1] < 8
        assert s.min() >= 0
        assert s.max() < 21
        assert I.max() < 52000
        assert I.min() >= 0
        R = Rwp((q, I), (qref, Iref))
        if R > 20: logger.error("Numpy has R=%s" % R)
        if logger.getEffectiveLevel() == logging.DEBUG:
            pylab.errorbar(q, I, s, label="Numpy R=%.1f" % R)
            pylab.yscale("log")
        self.assertEqual(R < 20, True, "Numpy: Measure R=%s<2" % R)

    def testCython(self):
        qref, Iref, s = self.ai.saxs(self.data, 1000)
        q, I, s = self.ai.saxs(self.data, 1000, error_model="poisson", method="cython")
        assert q[0] > 0
        assert q[-1] < 8
        assert s.min() >= 0
        assert s.max() < 21
        assert I.max() < 52000
        assert I.min() >= 0
        R = Rwp((q, I), (qref, Iref))
        if R > 20: logger.error("Cython has R=%s" % R)
        if logger.getEffectiveLevel() == logging.DEBUG:
            pylab.errorbar(q, I, s, label="Cython R=%.1f" % R)
            pylab.yscale("log")
        self.assertEqual(R < 20, True, "Cython: Measure R=%s<2" % R)

    def testSplitBBox(self):
        qref, Iref, s = self.ai.saxs(self.data, 1000)
        q, I, s = self.ai.saxs(self.data, 1000, error_model="poisson", method="splitbbox")
        assert q[0] > 0
        assert q[-1] < 8
        assert s.min() >= 0
        assert s.max() < 21
        assert I.max() < 52000
        assert I.min() >= 0
        R = Rwp((q, I), (qref, Iref))
        if R > 20: logger.error("SplitPixel has R=%s" % R)
        if logger.getEffectiveLevel() == logging.DEBUG:
            pylab.errorbar(q, I, s, label="SplitBBox R=%.1f" % R)
            pylab.yscale("log")
        self.assertEqual(R < 20, True, "SplitBBox: Measure R=%s<20" % R)

    def testSplitPixel(self):
        qref, Iref, s = self.ai.saxs(self.data, 1000)
        q, I, s = self.ai.saxs(self.data, 1000, error_model="poisson", method="splitpixel")
        assert q[0] > 0
        assert q[-1] < 8
        assert s.min() >= 0
        assert s.max() < 21
        assert I.max() < 52000
        assert I.min() >= 0
        R = Rwp((q, I), (qref, Iref))
        if R > 20: logger.error("SplitPixel has R=%s" % R)
        if logger.getEffectiveLevel() == logging.DEBUG:
            pylab.errorbar(q, I, s, label="SplitPixel R=%.1f" % R)
            pylab.yscale("log")
        self.assertEqual(R < 20, True, "SplitPixel: Measure R=%s<20" % R)

def test_suite_all_Saxs():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestSaxs("testMask"))
    testSuite.addTest(TestSaxs("testNumpy"))
    testSuite.addTest(TestSaxs("testCython"))
    testSuite.addTest(TestSaxs("testSplitBBox"))
    testSuite.addTest(TestSaxs("testSplitPixel"))
#    testSuite.addTest(TestSaxs("test_mask_splitBBox"))
#    testSuite.addTest(TestSaxs("test_mask_splitBBox"))
#    testSuite.addTest(TestSaxs("test_mask_splitBBox"))
#    testSuite.addTest(TestSaxs("test_mask_splitBBox"))

    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_all_Saxs()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
    if logger.getEffectiveLevel() == logging.DEBUG:
        pylab.legend()
        pylab.show()
        raw_input()
        pylab.clf()
