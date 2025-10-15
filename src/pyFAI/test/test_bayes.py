#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2025 European Synchrotron Radiation Facility, Grenoble, France
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

"""Test suites for bayesian background estimation"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "10/10/2025"

import unittest
import numpy
import logging
from ..utils import bayes
from .utilstest import UtilsTest
from scipy import interpolate
logger = logging.getLogger(__name__)


class TestBayes(unittest.TestCase):
    """Test Azimuthal integration based sparse matrix multiplication methods
    Bounding box pixel splitting
    """

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.size = 1000
        self.target = 1000
        self.outliers = 10
        self.big = 10000
        self.x = numpy.arange(self.size)
        rng = UtilsTest.get_rng()
        self.noise = rng.poisson(self.target, self.size)
        self.sigma = numpy.sqrt(self.noise)
        self.peaks = rng.uniform(0, self.size, self.outliers).astype(numpy.int64)

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.size = None
        self.target = None
        self.x = None
        self.noise = None
        self.sigma = None
        self.peaks = None

    def test_llk(self):
        x, y = bayes.BayesianBackground.test_bayes_llk()
        f = interpolate.interp1d(x, y)
        self.assertAlmostEqual(f(0), 0, msg="llk 0")
        self.assertAlmostEqual(f(-1), -1, msg="llk -1")
        self.assertAlmostEqual(f(-2), -4, msg="llk -2")
        self.assertAlmostEqual(f(-3), -9, msg="llk -1")
        self.assertAlmostEqual(f(-4), -16, msg="llk -2")
        self.assertAlmostEqual(f(1), -0.836596197557, msg="llk 1: %s" % (f(1)))
        self.assertAlmostEqual(f(0.01), -1e-4, msg="llk 1e-2")
        self.assertAlmostEqual(f(8), -4.62302437387, msg="llk 8: %s" % f(8))

    def test_background1d(self):
        mean = self.noise.mean()
        std = self.noise.std()
        bg = bayes.background(self.x, self.noise, self.sigma, k=1, npt=2)
        self.assertLess(mean - std, bg.min(), "background min")
        self.assertGreater(mean + std, bg.max(), "background max")

        noise = self.noise[:]
        noise[self.outliers] = self.big
        bg = bayes.background(self.x, noise, self.sigma, k=1, npt=2)
        self.assertLess(mean - std, bg.min(), "background min")
        self.assertGreater(mean + std, bg.max(), "background max")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestBayes))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
