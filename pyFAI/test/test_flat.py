#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
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

__doc__ = "test suite for dark_current / flat_field correction"
__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "19/07/2017"


import unittest
import numpy
import sys
from .utilstest import getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]
from ..opencl import ocl


class TestFlat1D(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.shape = 640, 480
        self.flat = 1.0 + numpy.random.random(self.shape)
        self.dark = numpy.random.random(self.shape)
        self.raw = self.flat + self.dark
        self.eps = 1e-6
        self.ai = pyFAI.AzimuthalIntegrator()
        self.ai.setFit2D(directDist=1, centerX=self.shape[1] // 2, centerY=self.shape[0] // 2, pixelX=1, pixelY=1)
        self.bins = 500

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.shape = None
        self.flat = None
        self.dark = None
        self.raw = None
        self.eps = None
        self.ai = None
        self.bins = None

    def test_no_correct(self):
        result = self.ai.integrate1d(self.raw, self.bins, unit="r_mm", correctSolidAngle=False)
        I = result.intensity
        logger.info("1D Without correction Imin=%s Imax=%s <I>=%s std=%s", I.min(), I.max(), I.mean(), I.std())
        self.assertNotAlmostEqual(I.mean(), 1, 2, "Mean should not be 1")
        self.assertFalse(I.max() - I.min() < self.eps, "deviation should be large")

    def test_correct(self):
        all_methods = ["numpy", "cython", "splitbbox", "splitpix", "lut", "csr"]
        if ocl:
            for device in ["cpu", "gpu", "acc"]:
                if ocl.select_device(dtype=device):
                    all_methods.append("lut_ocl_%s" % device)
                    all_methods.append("csr_ocl_%s" % device)

        for meth in all_methods:
            _, I = self.ai.integrate1d(self.raw, self.bins, unit="r_mm", method=meth, correctSolidAngle=False, dark=self.dark, flat=self.flat)
            logger.info("1D method:%s Imin=%s Imax=%s <I>=%s std=%s", meth, I.min(), I.max(), I.mean(), I.std())
            self.assertAlmostEqual(I.mean(), 1, 2, "Mean should be 1 in %s" % meth)
            self.assertTrue(I.max() - I.min() < self.eps, "deviation should be small with meth %s, got %s" % (meth, I.max() - I.min()))
        for meth in ["xrpd_numpy", "xrpd_cython", "xrpd_splitBBox", "xrpd_splitPixel"]:  # , "xrpd_OpenCL" ]: bug with 32 bit GPU and request 64 bit integration
            _, I = self.ai.__getattribute__(meth)(self.raw, self.bins, correctSolidAngle=False, dark=self.dark, flat=self.flat)
            logger.info("1D method:%s Imin=%s Imax=%s <I>=%s std=%s", meth, I.min(), I.max(), I.mean(), I.std())
            self.assertAlmostEqual(I.mean(), 1, 2, "Mean should be 1 in %s" % meth)
            self.assertTrue(I.max() - I.min() < self.eps, "deviation should be small with meth %s, got %s" % (meth, I.max() - I.min()))
        if ocl and pyFAI.opencl.ocl.select_device("gpu", extensions=["cl_khr_fp64"]):
            meth = "xrpd_OpenCL"
            _, I = self.ai.__getattribute__(meth)(self.raw, self.bins, correctSolidAngle=False, dark=self.dark, flat=self.flat)
            logger.info("1D method:%s Imin=%s Imax=%s <I>=%s std=%s", meth, I.min(), I.max(), I.mean(), I.std())
            self.assertAlmostEqual(I.mean(), 1, 2, "Mean should be 1 in %s" % meth)
            self.assertTrue(I.max() - I.min() < self.eps, "deviation should be small with meth %s, got %s" % (meth, I.max() - I.min()))


class TestFlat2D(unittest.TestCase):

    def setUp(self):
        self.shape = 640, 480
        self.flat = 1 + numpy.random.random(self.shape)
        self.dark = numpy.random.random(self.shape)
        self.raw = self.flat + self.dark
        self.eps = 1e-6
        self.ai = pyFAI.AzimuthalIntegrator()
        self.ai.setFit2D(directDist=1, centerX=self.shape[1] // 2, centerY=self.shape[0] // 2, pixelX=1, pixelY=1)
        self.bins = 500
        self.azim = 360

    def tearDown(self):
        self.shape = None
        self.flat = None
        self.dark = None
        self.raw = None
        self.eps = None
        self.ai = None
        self.bins = None
        self.azim = None

    def test_no_correct(self):
        I, _, _ = self.ai.integrate2d(self.raw, self.bins, self.azim, unit="r_mm", correctSolidAngle=False)
        I = I[numpy.where(I > 0)]
        logger.info("2D Without correction Imin=%s Imax=%s <I>=%s std=%s", I.min(), I.max(), I.mean(), I.std())

        self.assertNotAlmostEqual(I.mean(), 1, 2, "Mean should not be 1")
        self.assertFalse(I.max() - I.min() < self.eps, "deviation should be large")

    def test_correct(self):
        test2d = {"numpy": self.eps,
                  "cython": self.eps,
                  "splitbbox": self.eps,
                  "splitpix": self.eps,
                  "lut": self.eps,
                  }
        if ocl:
            for device in ["cpu", "gpu", "acc"]:
                if ocl.select_device(dtype=device):
                    test2d["lut_ocl_%s" % device] = self.eps
                    test2d["csr_ocl_%s" % device] = self.eps
        test2d_direct = {"xrpd2_numpy": 0.3,  # histograms are very noisy in 2D
                         "xrpd2_histogram": 0.3,  # histograms are very noisy in 2D
                         "xrpd2_splitBBox": self.eps,
                         "xrpd2_splitPixel": self.eps}

        for meth in test2d:
            logger.info("About to test2d %s", meth)
            try:
                I, _, _ = self.ai.integrate2d(self.raw, self.bins, self.azim, unit="r_mm", method=meth, correctSolidAngle=False, dark=self.dark, flat=self.flat)
            except (MemoryError, pyFAI.opencl.pyopencl.MemoryError):
                logger.warning("Got MemoryError from OpenCL device")
                continue
            I = I[numpy.where(I > 0)]
            logger.info("2D method:%s Imin=%s Imax=%s <I>=%s std=%s", meth, I.min(), I.max(), I.mean(), I.std())
            self.assertAlmostEqual(I.mean(), 1, 2, "Mean should be 1 in %s" % meth)
            self.assertTrue(I.max() - I.min() < test2d[meth], "deviation should be small with meth %s, got %s" % (meth, I.max() - I.min()))
        for meth in test2d_direct:
            logger.info("About to test2d_direct %s", meth)
            I, _, _ = self.ai.__getattribute__(meth)(self.raw, self.bins, self.azim, correctSolidAngle=False, dark=self.dark, flat=self.flat)
            I = I[numpy.where(I > 0)]
            logger.info("1D method:%s Imin=%s Imax=%s <I>=%s std=%s", meth, I.min(), I.max(), I.mean(), I.std())
            self.assertTrue(abs(I.mean() - 1) < test2d_direct[meth], "Mean should be 1 in %s" % meth)
            self.assertTrue(I.max() - I.min() < test2d_direct[meth], "deviation should be small with meth %s, got %s" % (meth, I.max() - I.min()))


def suite():
    testsuite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite.addTest(loader(TestFlat1D))
    testsuite.addTest(loader(TestFlat2D))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
