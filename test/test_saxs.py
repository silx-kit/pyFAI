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
    img = UtilsTest.getimage("1883/Pilatus1M.edf")
    data = fabio.open(img).data
    ai = pyFAI.AzimuthalIntegrator(1.58323111834, 0.0334170169115, 0.0412277798782, 0.00648735642526, 0.00755810191106, 0.0, detector=pyFAI.detectors.Pilatus1M())
    ai.wavelength = 1e-10
    npt = 1000
    def testMask(self):
        ss = self.ai.mask.sum()
        self.assertTrue(ss == 73533, "masked pixel = %s expected 73533" % ss)

    def testNumpy(self):
        qref, Iref, s = self.ai.saxs(self.data, self.npt)

        q, I, s = self.ai.saxs(self.data, self.npt, error_model="poisson", method="numpy")
        self.assertTrue(q[0] > 0, "q[0]>0 %s" % q[0])
        self.assertTrue(q[-1] < 8, "q[-1] < 8, got %s" % q[-1])
        self.assertTrue(s.min() >= 0, "s.min() >= 0 got %s" % (s.min()))
        self.assertTrue(s.max() < 21, "s.max() < 21 got %s" % (s.max()))
        self.assertTrue(I.max() < 52000, "I.max() < 52000 got %s" % (I.max()))
        self.assertTrue(I.min() >= 0, "I.min() >= 0 got %s" % (I.min()))
        R = Rwp((q, I), (qref, Iref))
        if R > 20: logger.error("Numpy has R=%s" % R)
        if logger.getEffectiveLevel() == logging.DEBUG:
            pylab.errorbar(q, I, s, label="Numpy R=%.1f" % R)
            pylab.yscale("log")
        self.assertTrue(R < 20, "Numpy: Measure R=%s<2" % R)

    def testCython(self):
        qref, Iref, s = self.ai.saxs(self.data, self.npt)
        q, I, s = self.ai.saxs(self.data, self.npt, error_model="poisson", method="cython")
        self.assertTrue(q[0] > 0, "q[0]>0 %s" % q[0])
        self.assertTrue(q[-1] < 8, "q[-1] < 8, got %s" % q[-1])
        self.assertTrue(s.min() >= 0, "s.min() >= 0 got %s" % (s.min()))
        self.assertTrue(s.max() < 21, "s.max() < 21 got %s" % (s.max()))
        self.assertTrue(I.max() < 52000, "I.max() < 52000 got %s" % (I.max()))
        self.assertTrue(I.min() >= 0, "I.min() >= 0 got %s" % (I.min()))
        R = Rwp((q, I), (qref, Iref))
        if R > 20: logger.error("Cython has R=%s" % R)
        if logger.getEffectiveLevel() == logging.DEBUG:
            pylab.errorbar(q, I, s, label="Cython R=%.1f" % R)
            pylab.yscale("log")
        self.assertTrue(R < 20, "Cython: Measure R=%s<2" % R)

    def testSplitBBox(self):
        qref, Iref, s = self.ai.saxs(self.data, self.npt)
        q, I, s = self.ai.saxs(self.data, self.npt, error_model="poisson", method="splitbbox")
        self.assertTrue(q[0] > 0, "q[0]>0 %s" % q[0])
        self.assertTrue(q[-1] < 8, "q[-1] < 8, got %s" % q[-1])
        self.assertTrue(s.min() >= 0, "s.min() >= 0 got %s" % (s.min()))
        self.assertTrue(s.max() < 21, "s.max() < 21 got %s" % (s.max()))
        self.assertTrue(I.max() < 52000, "I.max() < 52000 got %s" % (I.max()))
        self.assertTrue(I.min() >= 0, "I.min() >= 0 got %s" % (I.min()))
        R = Rwp((q, I), (qref, Iref))
        if R > 20: logger.error("SplitPixel has R=%s" % R)
        if logger.getEffectiveLevel() == logging.DEBUG:
            pylab.errorbar(q, I, s, label="SplitBBox R=%.1f" % R)
            pylab.yscale("log")
        self.assertEqual(R < 20, True, "SplitBBox: Measure R=%s<20" % R)

    def testSplitPixel(self):
        qref, Iref, s = self.ai.saxs(self.data, self.npt)
        q, I, s = self.ai.saxs(self.data, self.npt, error_model="poisson", method="splitpixel")
        self.assertTrue(q[0] > 0, "q[0]>0 %s" % q[0])
        self.assertTrue(q[-1] < 8, "q[-1] < 8, got %s" % q[-1])
        self.assertTrue(s.min() >= 0, "s.min() >= 0 got %s" % (s.min()))
        self.assertTrue(s.max() < 21, "s.max() < 21 got %s" % (s.max()))
        self.assertTrue(I.max() < 52000, "I.max() < 52000 got %s" % (I.max()))
        self.assertTrue(I.min() >= 0, "I.min() >= 0 got %s" % (I.min()))
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
#    testSuite.addTest(TestSaxs("testCython"))
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
