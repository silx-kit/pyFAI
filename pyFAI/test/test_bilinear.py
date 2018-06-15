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

from __future__ import absolute_import, division, print_function

"""Test suite for bilinear interpolator class"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "10/01/2018"


import unittest
import numpy
import logging
from .utilstest import UtilsTest
logger = logging.getLogger(__name__)
from ..ext import bilinear


class TestBilinear(unittest.TestCase):
    """basic maximum search test"""
    N = 10000

    def test_max_search_round(self):
        """test maximum search using random points: maximum is at the pixel center"""
        a = numpy.arange(100) - 40.
        b = numpy.arange(100) - 60.
        ga = numpy.exp(-a * a / 4000)
        gb = numpy.exp(-b * b / 6000)
        gg = numpy.outer(ga, gb)
        b = bilinear.Bilinear(gg)
        ok = 0
        for _s in range(self.N):
            i, j = numpy.random.randint(100), numpy.random.randint(100)
            k, l = b.local_maxi((i, j))
            if abs(k - 40) > 1e-4 or abs(l - 60) > 1e-4:
                logger.warning("Wrong guess maximum (%i,%i) -> (%.1f,%.1f)", i, j, k, l)
            else:
                logger.debug("Good guess maximum (%i,%i) -> (%.1f,%.1f)", i, j, k, l)
                ok += 1
        logger.info("Success rate: %.1f", 100.0 * ok / self.N)
        self.assertEqual(ok, self.N, "Maximum is always found")

    def test_max_search_half(self):
        """test maximum search using random points: maximum is at a pixel edge"""
        a = numpy.arange(100) - 40.5
        b = numpy.arange(100) - 60.5
        ga = numpy.exp(-a * a / 4000)
        gb = numpy.exp(-b * b / 6000)
        gg = numpy.outer(ga, gb)
        b = bilinear.Bilinear(gg)
        ok = 0
        for _s in range(self.N):
            i, j = numpy.random.randint(100), numpy.random.randint(100)
            k, l = b.local_maxi((i, j))
            if abs(k - 40.5) > 0.5 or abs(l - 60.5) > 0.5:
                logger.warning("Wrong guess maximum (%i,%i) -> (%.1f,%.1f)", i, j, k, l)
            else:
                logger.debug("Good guess maximum (%i,%i) -> (%.1f,%.1f)", i, j, k, l)
                ok += 1
        logger.info("Success rate: %.1f", 100.0 * ok / self.N)
        self.assertEqual(ok, self.N, "Maximum is always found")


class TestConversion(unittest.TestCase):
    """basic 2d -> 4d transformation and vice-versa"""
    def test4d(self):
        Nx = 1000
        Ny = 1024
        y, x = numpy.mgrid[:Ny + 1, :Nx + 1]
        y = y.astype(float)
        x = x.astype(float)
        # print(y.dtype, x.dtype)
        pos = bilinear.convert_corner_2D_to_4D(3, numpy.ascontiguousarray(y), numpy.ascontiguousarray(x))
        y1, x1, z1 = bilinear.calc_cartesian_positions(y.ravel(), x.ravel(), pos)
        self.assertTrue(numpy.allclose(y.ravel(), y1), "Maximum error on y is %s" % (abs(y.ravel() - y1).max()))
        self.assertTrue(numpy.allclose(x.ravel(), x1), "Maximum error on x is %s" % (abs(x.ravel() - x1).max()))
        self.assertEqual(z1, None, "flat detector")
        x = x[:-1, :-1] + 0.5
        y = y[:-1, :-1] + 0.5
        y1, x1, z1 = bilinear.calc_cartesian_positions((y).ravel(), (x).ravel(), pos)

        self.assertTrue(numpy.allclose(y.ravel(), y1), "Maximum error on y_center is %s" % (abs(y.ravel() - y1).max()))
        self.assertTrue(numpy.allclose(x.ravel(), x1), "Maximum error on x_center is %s" % (abs(x.ravel() - x1).max()))
        self.assertEqual(z1, None, "flat detector")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestBilinear))
    testsuite.addTest(loader(TestConversion))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
    UtilsTest.clean_up()
