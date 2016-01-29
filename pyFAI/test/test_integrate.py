#!/usr/bin/python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) 2015 European Synchrotron Radiation Facility, Grenoble, France
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

__doc__ = "test suite for masked arrays"

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "29/01/2016"


import unittest
import os
import numpy
import logging
import time
import sys
import fabio
from .utilstest import UtilsTest, Rwp, getLogger
logger = getLogger(__file__)
from ..azimuthalIntegrator import AzimuthalIntegrator
from ..detectors import Pilatus1M
if logger.getEffectiveLevel() <= logging.INFO:
    import pylab


class TestIntegrate1D(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.npt = 1000
        self.img = UtilsTest.getimage("1883/Pilatus1M.edf")
        self.data = fabio.open(self.img).data
        self.ai = AzimuthalIntegrator(1.58323111834, 0.0334170169115, 0.0412277798782, 0.00648735642526, 0.00755810191106, 0.0, detector=Pilatus1M())
        self.ai.wavelength = 1e-10
        self.Rmax = 3

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.npt = self.img = self.data = self.ai = self.Rmax

    def testQ(self):
        res = {}
        for m in ("numpy", "cython", "BBox" , "splitpixel", "lut", "lut_ocl"):
            res[m] = self.ai.integrate1d(self.data, self.npt, method=m, radial_range=(0.5, 5.8))
        for a in res:
            for b in res:
                R = Rwp(res[a], res[b])
                mesg = "testQ: %s vs %s measured R=%s<%s" % (a, b, R, self.Rmax)
                if R > self.Rmax:
                    logger.error(mesg)
                else:
                    logger.info(mesg)
                self.assertTrue(R <= self.Rmax, mesg)

    def testR(self):
        res = {}
        for m in ("numpy", "cython", "BBox" , "splitpixel", "lut", "lut_ocl"):
            res[m] = self.ai.integrate1d(self.data, self.npt, method=m, unit="r_mm", radial_range=(20, 150))
        for a in res:
            for b in res:
                R = Rwp(res[a], res[b])
                mesg = "testR: %s vs %s measured R=%s<%s" % (a, b, R, self.Rmax)
                if R > self.Rmax:
                    logger.error(mesg)
                else:
                    logger.info(mesg)
                self.assertTrue(R <= self.Rmax, mesg)
    def test2th(self):
        res = {}
        for m in ("numpy", "cython", "BBox" , "splitpixel", "lut", "lut_ocl"):
            res[m] = self.ai.integrate1d(self.data, self.npt, method=m, unit="2th_deg", radial_range=(0.5, 5.5))
        for a in res:
            for b in res:
                R = Rwp(res[a], res[b])
                mesg = "test2th: %s vs %s measured R=%s<%s" % (a, b, R, self.Rmax)
                if R > self.Rmax:
                    logger.error(mesg)
                else:
                    logger.info(mesg)
                self.assertTrue(R <= self.Rmax, mesg)


class TestIntegrate2D(unittest.TestCase):
    npt = 500
    img = UtilsTest.getimage("1883/Pilatus1M.edf")
    data = fabio.open(img).data
    ai = AzimuthalIntegrator(1.58323111834, 0.0334170169115, 0.0412277798782, 0.00648735642526, 0.00755810191106, 0.0, detector=Pilatus1M())
    ai.wavelength = 1e-10
    Rmax = 20
    delta_pos_azim_max = 0.28

    def testQ(self):
        res = {}
        for m in ("numpy", "cython", "BBox" , "splitpixel"):  # , "lut", "lut_ocl"):
            res[m] = self.ai.integrate2d(self.data, self.npt, method=m)
        mask = (res["numpy"][0] != 0)
        self.assertTrue(mask.sum() > 36 * self.npt, "10%% of the pixels are valid at least")
        for a in res:
            for b in res:
                delta_pos_rad = abs(res[a][1] - res[b][1]).max()
                delta_pos_azim = abs(res[a][2] - res[b][2]).max()
                R = abs((res[a][0][mask] - res[b][0][mask]) / numpy.maximum(1, res[a][0][mask])).mean() * 100
                mesg = "testQ 2D: %s vs %s measured delta rad=%s azim=%s R=%s<%s" % (a, b, delta_pos_rad, delta_pos_azim, R, self.Rmax)
                if R > self.Rmax:
                    logger.error(mesg)
                else:
                    logger.info(mesg)
                self.assertTrue(delta_pos_rad <= 0.01, mesg)
                self.assertTrue(delta_pos_azim <= self.delta_pos_azim_max, mesg)
                self.assertTrue(R <= self.Rmax, mesg)

    def testR(self):
        res = {}
        for m in ("numpy", "cython", "BBox" , "splitpixel"):  # , "lut", "lut_ocl"):
            res[m] = self.ai.integrate2d(self.data, self.npt, method=m, unit="r_mm")  # , radial_range=(20, 150))
        mask = (res["numpy"][0] != 0)
        self.assertTrue(mask.sum() > 36 * self.npt, "10%% of the pixels are valid at least")
        for a in res:
            for b in res:
                delta_pos_rad = abs(res[a][1] - res[b][1]).max()
                delta_pos_azim = abs(res[a][2] - res[b][2]).max()
                R = abs((res[a][0][mask] - res[b][0][mask]) / numpy.maximum(1, res[a][0][mask])).mean() * 100
                mesg = "testR 2D: %s vs %s measured delta rad=%s azim=%s R=%s<%s" % (a, b, delta_pos_rad, delta_pos_azim, R, self.Rmax)
                if R > self.Rmax:
                    logger.error(mesg)
                else:
                    logger.info(mesg)
                self.assertTrue(delta_pos_rad <= 0.28, mesg)
                self.assertTrue(delta_pos_azim <= self.delta_pos_azim_max, mesg)
                self.assertTrue(R <= self.Rmax, mesg)
    def test2th(self):
        res = {}
        for m in ("numpy", "cython", "BBox" , "splitpixel"):  # , "lut", "lut_ocl"):
            res[m] = self.ai.integrate2d(self.data, self.npt, method=m, unit="2th_deg")  # , radial_range=(0.5, 5.5))
        mask = (res["numpy"][0] != 0)
        self.assertTrue(mask.sum() > 36 * self.npt, "10%% of the pixels are valid at least")
        for a in res:
            for b in res:
                if a == b:
                    continue
                delta_pos_rad = abs(res[a][1] - res[b][1]).max()
                delta_pos_azim = abs(res[a][2] - res[b][2]).max()
                R = abs((res[a][0][mask] - res[b][0][mask]) / numpy.maximum(1, res[a][0][mask])).mean() * 100
                mesg = "test2th 2D: %s vs %s measured delta rad=%s azim=%s R=%s<%s" % (a, b, delta_pos_rad, delta_pos_azim, R, self.Rmax)
                if R > self.Rmax:
                    logger.error(mesg)
                else:
                    logger.info(mesg)
                self.assertTrue(delta_pos_rad <= 0.01, mesg)
                self.assertTrue(R <= self.Rmax, mesg)


def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(TestIntegrate1D("testQ"))
    testsuite.addTest(TestIntegrate1D("testR"))
    testsuite.addTest(TestIntegrate1D("test2th"))
    testsuite.addTest(TestIntegrate2D("testQ"))
    testsuite.addTest(TestIntegrate2D("testR"))
    testsuite.addTest(TestIntegrate2D("test2th"))

    return testsuite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
#    if logger.getEffectiveLevel() == logging.DEBUG:
#        pylab.legend()
#        pylab.show()
#        raw_input()
#        pylab.clf()
