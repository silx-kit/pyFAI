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
"test suite for dark_current / flat_field correction"

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "05/04/2013"


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

class TestFlat1D(unittest.TestCase):
    shape = 640, 480
    flat = 1 + numpy.random.random(shape)
    dark = numpy.random.random(shape)
    raw = flat + dark
    eps = 1e-6
    ai = pyFAI.AzimuthalIntegrator()
    ai.setFit2D(directDist=1, centerX=shape[1] // 2, centerY=shape[0] // 2, pixelX=1, pixelY=1)
    bins = 500

    def test_no_correct(self):
        r, I = self.ai.integrate1d(self.raw, self.bins, unit="r_mm", correctSolidAngle=False)
        logger.info("1D Without correction Imin=%s Imax=%s <I>=%s std=%s" % (I.min(), I.max(), I.mean(), I.std()))
        self.assertNotAlmostEqual(I.mean(), 1, 2, "Mean should not be 1")
        self.assertFalse(I.max() - I.min() < self.eps, "deviation shaould be large")
    def test_correct(self):
        for meth in ["numpy", "cython", "splitbbox", "splitpix", "lut", "lut_ocl" ]:
            r, I = self.ai.integrate1d(self.raw, self.bins, unit="r_mm", method=meth, correctSolidAngle=False, dark=self.dark, flat=self.flat)
            logger.info("1D method:%s Imin=%s Imax=%s <I>=%s std=%s" % (meth, I.min(), I.max(), I.mean(), I.std()))
            self.assertAlmostEqual(I.mean(), 1, 2, "Mean should be 1 in %s" % meth)
            self.assert_(I.max() - I.min() < self.eps, "deviation should be small with meth %s, got %s" % (meth, I.max() - I.min()))
        for meth in ["xrpd_numpy", "xrpd_cython", "xrpd_splitBBox", "xrpd_splitPixel"]:  # , "xrpd_OpenCL" ]: bug with 32 bit GPU and request 64 bit integration
            r, I = self.ai.__getattribute__(meth)(self.raw, self.bins, correctSolidAngle=False, dark=self.dark, flat=self.flat)
            logger.info("1D method:%s Imin=%s Imax=%s <I>=%s std=%s" % (meth, I.min(), I.max(), I.mean(), I.std()))
            self.assertAlmostEqual(I.mean(), 1, 2, "Mean should be 1 in %s" % meth)
            self.assert_(I.max() - I.min() < self.eps, "deviation should be small with meth %s, got %s" % (meth, I.max() - I.min()))
        if pyFAI.opencl.ocl and pyFAI.opencl.ocl.select_device("gpu", extensions=["cl_khr_fp64"]):
            meth = "xrpd_OpenCL"
            r, I = self.ai.__getattribute__(meth)(self.raw, self.bins, correctSolidAngle=False, dark=self.dark, flat=self.flat)
            logger.info("1D method:%s Imin=%s Imax=%s <I>=%s std=%s" % (meth, I.min(), I.max(), I.mean(), I.std()))
            self.assertAlmostEqual(I.mean(), 1, 2, "Mean should be 1 in %s" % meth)
            self.assert_(I.max() - I.min() < self.eps, "deviation should be small with meth %s, got %s" % (meth, I.max() - I.min()))

class TestFlat2D(unittest.TestCase):
    shape = 640, 480
    flat = 1 + numpy.random.random(shape)
    dark = numpy.random.random(shape)
    raw = flat + dark
    eps = 1e-6
    ai = pyFAI.AzimuthalIntegrator()
    ai.setFit2D(directDist=1, centerX=shape[1] // 2, centerY=shape[0] // 2, pixelX=1, pixelY=1)
    bins = 500
    azim = 360
    def test_no_correct(self):
        I, _ , _ = self.ai.integrate2d(self.raw, self.bins, self.azim, unit="r_mm", correctSolidAngle=False)
        I = I[numpy.where(I > 0)]
        logger.info("2D Without correction Imin=%s Imax=%s <I>=%s std=%s" % (I.min(), I.max(), I.mean(), I.std()))

        self.assertNotAlmostEqual(I.mean(), 1, 2, "Mean should not be 1")
        self.assertFalse(I.max() - I.min() < self.eps, "deviation should be large")

    def test_correct(self):
        test2d = {"numpy":self.eps,
                  "cython":self.eps, 
                  "splitbbox":self.eps, 
                  "splitpix":self.eps, 
                  "lut":self.eps, 
                  "lut_ocl":self.eps}
        test2d_direct = {"xrpd2_numpy":0.3,#histograms are very noisy in 2D
                  "xrpd2_histogram":0.3,   #histograms are very noisy in 2D
                  "xrpd2_splitBBox":self.eps, 
                  "xrpd2_splitPixel":self.eps}
        for meth in test2d:
            I, _, _ = self.ai.integrate2d(self.raw, self.bins, self.azim, unit="r_mm", method=meth, correctSolidAngle=False, dark=self.dark, flat=self.flat)
            I = I[numpy.where(I > 0)]
            logger.info("2D method:%s Imin=%s Imax=%s <I>=%s std=%s" % (meth, I.min(), I.max(), I.mean(), I.std()))
            self.assertAlmostEqual(I.mean(), 1, 2, "Mean should be 1 in %s" % meth)
            self.assert_(I.max() - I.min() < test2d[meth], "deviation should be small with meth %s, got %s" % (meth, I.max() - I.min()))
        for meth in test2d_direct:
            I, _, _ = self.ai.__getattribute__(meth)(self.raw, self.bins, self.azim, correctSolidAngle=False, dark=self.dark, flat=self.flat)
            I = I[numpy.where(I > 0)]
            logger.info("1D method:%s Imin=%s Imax=%s <I>=%s std=%s" % (meth, I.min(), I.max(), I.mean(), I.std()))
            self.assert_(abs(I.mean() - 1) < test2d_direct[meth], "Mean should be 1 in %s" % meth)
            self.assert_(I.max() - I.min() < test2d_direct[meth], "deviation should be small with meth %s, got %s" % (meth, I.max() - I.min()))

def test_suite_all_Flat():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestFlat1D("test_no_correct"))
    testSuite.addTest(TestFlat1D("test_correct"))
    testSuite.addTest(TestFlat2D("test_no_correct"))
    testSuite.addTest(TestFlat2D("test_correct"))

    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_all_Flat()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)



