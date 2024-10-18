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

"test suite for dark_current / flat_field correction"

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "31/01/2024"

import unittest
import numpy
import sys
import logging
logger = logging.getLogger(__name__)
pyFAI = sys.modules["pyFAI"]
from ..opencl import ocl
from .utilstest import UtilsTest
from ..integrator.azimuthal import AzimuthalIntegrator
from ..method_registry import IntegrationMethod


class TestFlat1D(unittest.TestCase):

    @classmethod
    def setUpClass(cls)->None:
        super(TestFlat1D, cls).setUpClass()
        cls.rng = UtilsTest.get_rng()
        cls.shape = 640, 480
        cls.flat = 1.0 + cls.rng.random(cls.shape)
        cls.dark = cls.rng.random(cls.shape)
        cls.raw = cls.flat + cls.dark
        cls.eps = 1e-6
        cls.ai = AzimuthalIntegrator()
        # 100mm distance and 100µm pixel size
        cls.ai.setFit2D(directDist=100, centerX=cls.shape[1] // 2, centerY=cls.shape[0] // 2, pixelX=100, pixelY=100)
        cls.bins = 500

    @classmethod
    def tearDownClass(cls)->None:
        super(TestFlat1D, cls).tearDownClass()
        cls.shape = None
        cls.flat = None
        cls.dark = None
        cls.raw = None
        cls.eps = None
        cls.ai = None
        cls.bins = None
        cls.rng = None

    def test_no_correct(self):
        result = self.ai.integrate1d_ng(self.raw, self.bins, unit="r_mm", correctSolidAngle=False)
        I = result.intensity
        logger.info("1D Without correction Imin=%s Imax=%s <I>=%s std=%s", I.min(), I.max(), I.mean(), I.std())
        self.assertNotAlmostEqual(I.mean(), 1, 2, "Mean should not be 1")
        self.assertFalse(I.max() - I.min() < self.eps, "deviation should be large")

    def test_correct(self):
        methods = { k.method[1:4]:k for k in  IntegrationMethod.select_method(dim=1)}
        logger.info("testing %s methods with 1d integration", len(methods))
        for meth in methods.values():
            if meth.dimension != 1: continue
            res = self.ai.integrate1d_ng(self.raw, self.bins, unit="r_mm", method=meth, correctSolidAngle=False, dark=self.dark, flat=self.flat)

            if meth.target_name and meth.algo_lower == "histogram":
                "OpenCL atomic are not that good !"
                eps = 3 * self.eps
            else:
                eps = self.eps
            # print(meth)
            # print(res.sum_signal)
            # print(res.sum_normalization)
            # print(res.intensity)
            _, I = res
            logger.info("1D method:%s Imin=%s Imax=%s <I>=%s std=%s", str(meth), I.min(), I.max(), I.mean(), I.std())
            self.assertAlmostEqual(I.mean(), 1, 2, "Mean should be 1 in %s" % meth)
            self.assertLess(I.max() - I.min(), eps, "deviation should be small with meth %s, got %s" % (meth, I.max() - I.min()))


class TestFlat2D(unittest.TestCase):

    @classmethod
    def setUpClass(cls)->None:
        super(TestFlat2D, cls).setUpClass()
        cls.rng = UtilsTest.get_rng()
        cls.shape = 640, 480
        cls.flat = 1 + cls.rng.random(cls.shape)
        cls.dark = cls.rng.random(cls.shape)
        cls.raw = cls.flat + cls.dark
        cls.eps = 1e-6
        cls.ai = AzimuthalIntegrator()
        cls.ai.setFit2D(directDist=1, centerX=cls.shape[1] // 2, centerY=cls.shape[0] // 2, pixelX=1, pixelY=1)
        cls.bins = 500
        cls.azim = 360

    @classmethod
    def tearDownClass(cls)->None:
        super(TestFlat2D, cls).tearDownClass()
        cls.shape = None
        cls.flat = None
        cls.dark = None
        cls.raw = None
        cls.eps = None
        cls.ai = None
        cls.bins = None
        cls.azim = None
        cls.rng = None

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
        if ocl and UtilsTest.opencl:
            for device in ["cpu", "gpu", "acc"]:
                if ocl.select_device(dtype=device):
                    test2d["lut_ocl_%s" % device] = self.eps
                    test2d["csr_ocl_%s" % device] = self.eps

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


def suite():
    testsuite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite.addTest(loader(TestFlat1D))
    testsuite.addTest(loader(TestFlat2D))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
