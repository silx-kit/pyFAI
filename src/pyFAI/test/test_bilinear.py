#!/usr/bin/env python
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

"""Test suite for bilinear interpolator class"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "08/09/2026"

import logging
import unittest

import numpy

from ..ext import bilinear
from .utilstest import UtilsTest

logger = logging.getLogger(__name__)


class TestBilinear(unittest.TestCase):
    """basic maximum search test"""
    @classmethod
    def setUpClass(cls)->None:
        super().setUpClass()
        cls.N = 10000
        cls.rng = UtilsTest.get_rng()
    @classmethod
    def tearDownClass(cls)->None:
        super().tearDownClass()
        cls.rng = None

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
            i, j = int(self.rng.uniform(0, 100)), int(self.rng.uniform(0, 100))
            p0, p1 = b.local_maxi((i, j))
            if abs(p0 - 40) > 1e-4 or abs(p1 - 60) > 1e-4:
                logger.warning("Wrong guess maximum (%i,%i) -> (%.1f,%.1f)", i, j, p0, p1)
            else:
                logger.debug("Good guess maximum (%i,%i) -> (%.1f,%.1f)", i, j, p0, p1)
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
            i, j = int(self.rng.uniform(0,100)), int(self.rng.uniform(0,100))
            p0, p1 = b.local_maxi((i, j))
            if abs(p0 - 40.5) > 0.5 or abs(p1 - 60.5) > 0.5:
                logger.warning("Wrong guess maximum (%i,%i) -> (%.1f,%.1f)", i, j, p0, p1)
            else:
                logger.debug("Good guess maximum (%i,%i) -> (%.1f,%.1f)", i, j, p0, p1)
                ok += 1
        logger.info("Success rate: %.1f", 100.0 * ok / self.N)
        self.assertEqual(ok, self.N, "Maximum is always found")


    def test_subpixel_quadratic(self):
        """The second order Taylor expansion is exact on a quadratic surface

        The sub-pixel position of the maximum should be found at the precision of the
        float32 storage, even when the quadratic form has a cross-term.
        """
        shape = (25, 25)
        pos1, pos2 = numpy.ogrid[:shape[0], :shape[1]]
        for _s in range(100):
            center = 12 + self.rng.uniform(-0.4, 0.4, 2)
            # negative definite quadratic form: a>0, b>0 and a*b > c**2
            a, b = self.rng.uniform(0.02, 0.1, 2)
            c = self.rng.uniform(-0.02, 0.02)
            d1 = pos1 - center[0]
            d2 = pos2 - center[1]
            data = (1.0 - a * d1 * d1 - b * d2 * d2 - 2.0 * c * d1 * d2).astype(numpy.float32)
            p0, p1 = bilinear.Bilinear(data).local_maxi((12, 12))
            err = numpy.sqrt((p0 - center[0]) ** 2 + (p1 - center[1]) ** 2)
            self.assertLess(err, 1e-4, f"quadratic maximum {center} found at ({p0}, {p1})")

    def test_subpixel_gaussian(self):
        """Sub-pixel refinement of a 2D Gaussian sitting at a known position"""
        shape = (25, 25)
        pos1, pos2 = numpy.ogrid[:shape[0], :shape[1]]
        # sigma along both axes, rotation of the Gaussian, tolerance in pixel
        for sigma1, sigma2, angle, tol in ((2.0, 2.0, 0.0, 0.06),
                                           (3.0, 1.5, 30.0, 0.30),
                                           (3.0, 1.5, 60.0, 0.30)):
            cos_a = numpy.cos(numpy.deg2rad(angle))
            sin_a = numpy.sin(numpy.deg2rad(angle))
            for _s in range(100):
                center = 12 + self.rng.uniform(-0.5, 0.5, 2)
                d1 = pos1 - center[0]
                d2 = pos2 - center[1]
                u = cos_a * d1 + sin_a * d2
                v = -sin_a * d1 + cos_a * d2
                data = numpy.exp(-0.5 * ((u / sigma1) ** 2 + (v / sigma2) ** 2)).astype(numpy.float32)
                idx = numpy.unravel_index(numpy.argmax(data), shape)
                p0, p1 = bilinear.Bilinear(data).local_maxi((int(idx[0]), int(idx[1])))
                err = numpy.sqrt((p0 - center[0]) ** 2 + (p1 - center[1]) ** 2)
                self.assertLess(err, tol, f"Gaussian ({sigma1}, {sigma2}, {angle}°) centered "
                                          f"on {center} found at ({p0}, {p1})")


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
        self.assertTrue(numpy.allclose(y.ravel(), y1), f"Maximum error on y is {abs(y.ravel() - y1).max()}")
        self.assertTrue(numpy.allclose(x.ravel(), x1), f"Maximum error on x is {abs(x.ravel() - x1).max()}")
        self.assertEqual(z1, None, "flat detector")
        x = x[:-1, :-1] + 0.5
        y = y[:-1, :-1] + 0.5
        y1, x1, z1 = bilinear.calc_cartesian_positions((y).ravel(), (x).ravel(), pos)

        self.assertTrue(numpy.allclose(y.ravel(), y1), f"Maximum error on y_center is {abs(y.ravel() - y1).max()}")
        self.assertTrue(numpy.allclose(x.ravel(), x1), f"Maximum error on x_center is {abs(x.ravel() - x1).max()}")
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
