#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2025-2025 European Synchrotron Radiation Facility, Grenoble, France
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

"""Test suite for container module"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jérôme.Kieffer@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "02/12/2025"

import unittest
import copy
import numpy
import logging
import fabio
from .utilstest import UtilsTest,TestLogging
from .. import load as pyFAI_load
from .. import containers
from ..utils.decorators import depreclog

logger = logging.getLogger(__name__)


class TestContainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img = fabio.open(UtilsTest.getimage("moke.tif")).data
        cls.ai = pyFAI_load(
            {
                "poni_version": 2.1,
                "detector": "Detector",
                "detector_config": {
                    "pixel1": 1e-4,
                    "pixel2": 1e-4,
                    "max_shape": [500, 600],
                    "orientation": 3,
                },
                "dist": 0.1,
                "poni1": 0.03,
                "poni2": 0.03,
                "rot1": 0.0,
                "rot2": 0.0,
                "rot3": 0.0,
                "wavelength": 1.0178021533473887e-10,
            }
        )

    @classmethod
    def tearDownClass(cls):
        cls.img = cls.ai = None

    def test_rebin1d(self):
        method = ("no", "histogram", "cython")
        res2d = self.ai.integrate2d(
            self.img, 500, 360, method=method, error_model="poisson"
        )
        ref1d = self.ai.integrate1d(self.img, 500, method=method, error_model="poisson")
        with TestLogging(logger=depreclog, warning=1):
            res1d = containers.rebin1d(res2d)
        self.assertTrue(numpy.allclose(res1d[0], ref1d[0]), "radial matches")
        self.assertTrue(numpy.allclose(res1d[1], ref1d[1]), "intensity matches")
        self.assertTrue(numpy.allclose(res1d[2], ref1d[2]), "sem matches")

    def test_symmetrize(self):
        res2d = self.ai.integrate2d(
            self.img,
            500,
            360,
            error_model="poisson",
            radial_range=(0, 12),
            unit="2th_deg",
        )
        sym = containers.symmetrize(res2d)
        self.assertAlmostEqual(res2d.intensity.mean(), sym.intensity.mean(), places=0)

    def test_spottiness(self):
        python = self.ai.integrate1d(self.img, 100, method=("no","csr", "python"), error_model="azimuth").calc_spottiness()
        cython = self.ai.integrate1d(self.img, 100, method=("no","csr", "cython"), error_model="azimuth").calc_spottiness()
        self.assertAlmostEqual(python, 0.06396, msg="python", places=4)
        self.assertAlmostEqual(cython, 0.06396, msg="cython", places=4)

    def test_maths(self):
        method = ("no", "histogram", "cython")
        a1d = self.ai.integrate1d(self.img, 10, method=method, error_model="poisson")
        b1d = self.ai.integrate1d(
            numpy.ones_like(self.img), 10, method=method, error_model="poisson"
        )

        c1d = a1d + b1d
        self.assertTrue(numpy.allclose(c1d.sum_signal, a1d.sum_signal + b1d.sum_signal))
        self.assertTrue(
            numpy.allclose(c1d.sum_variance, a1d.sum_variance + b1d.sum_variance)
        )
        self.assertTrue(numpy.allclose(c1d.sum_normalization, b1d.sum_normalization))
        self.assertTrue(numpy.allclose(c1d.sum_normalization2, b1d.sum_normalization2))
        self.assertTrue(numpy.allclose(c1d.count, b1d.count))
        self.assertTrue(numpy.allclose(c1d.radial, b1d.radial))
        self.assertTrue(numpy.all(c1d.intensity > a1d.intensity))
        self.assertTrue(numpy.all(c1d.intensity >= b1d.intensity))
        self.assertTrue(numpy.all(c1d.std > a1d.std))
        self.assertTrue(numpy.all(c1d.std > b1d.std))
        self.assertTrue(numpy.all(c1d.sem > a1d.sem))
        self.assertTrue(numpy.all(c1d.sem > b1d.sem))
        self.assertTrue(numpy.all(c1d.sigma > a1d.sigma))
        self.assertTrue(numpy.all(c1d.sigma > b1d.sigma))

        d1d = c1d - b1d
        self.assertTrue(numpy.allclose(d1d.sum_signal, a1d.sum_signal))
        self.assertTrue(
            numpy.allclose(d1d.sum_variance, a1d.sum_variance + 2 * b1d.sum_variance)
        )
        self.assertTrue(numpy.allclose(d1d.sum_normalization, b1d.sum_normalization))
        self.assertTrue(numpy.allclose(d1d.sum_normalization2, b1d.sum_normalization2))
        self.assertTrue(numpy.allclose(d1d.count, b1d.count))
        self.assertTrue(numpy.allclose(d1d.radial, b1d.radial))
        self.assertTrue(numpy.allclose(d1d.intensity, a1d.intensity))
        self.assertTrue(numpy.all(d1d.std > a1d.std))
        self.assertTrue(numpy.all(d1d.std > b1d.std))
        self.assertTrue(numpy.all(d1d.sem > a1d.sem))
        self.assertTrue(numpy.all(d1d.sem > b1d.sem))
        self.assertTrue(numpy.all(d1d.sigma > a1d.sigma))
        self.assertTrue(numpy.all(d1d.sigma > b1d.sigma))

        e1d = copy.deepcopy(a1d)
        e1d += b1d
        self.assertTrue(numpy.allclose(e1d.sum_signal, c1d.sum_signal))
        self.assertTrue(numpy.allclose(e1d.sum_variance, c1d.sum_variance))
        self.assertTrue(numpy.allclose(e1d.sum_normalization, c1d.sum_normalization))
        self.assertTrue(numpy.allclose(e1d.sum_normalization2, c1d.sum_normalization2))
        self.assertTrue(numpy.allclose(e1d.count, c1d.count))
        self.assertTrue(numpy.allclose(e1d.radial, c1d.radial))
        self.assertTrue(numpy.allclose(e1d.intensity, c1d.intensity))
        self.assertTrue(numpy.allclose(e1d.std, c1d.std))
        self.assertTrue(numpy.allclose(e1d.sem, c1d.sem))
        self.assertTrue(numpy.allclose(e1d.sigma, c1d.sigma))

        f1d = copy.deepcopy(c1d)
        f1d -= b1d
        self.assertTrue(numpy.allclose(f1d.sum_signal, a1d.sum_signal))
        self.assertTrue(numpy.allclose(f1d.sum_variance, d1d.sum_variance))
        self.assertTrue(numpy.allclose(f1d.sum_normalization, a1d.sum_normalization))
        self.assertTrue(numpy.allclose(f1d.sum_normalization2, a1d.sum_normalization2))
        self.assertTrue(numpy.allclose(f1d.count, a1d.count))
        self.assertTrue(numpy.allclose(f1d.radial, a1d.radial))
        self.assertTrue(numpy.allclose(f1d.intensity, a1d.intensity))
        self.assertTrue(numpy.allclose(f1d.std, d1d.std))
        self.assertTrue(numpy.allclose(f1d.sem, d1d.sem))
        self.assertTrue(numpy.allclose(f1d.sigma, d1d.sigma))

        g1d = a1d.union(b1d)
        self.assertTrue(numpy.allclose(g1d.sum_signal, a1d.sum_signal+b1d.sum_signal))
        self.assertTrue(numpy.allclose(g1d.sum_variance, a1d.sum_variance+b1d.sum_variance))
        self.assertTrue(numpy.allclose(g1d.sum_normalization, a1d.sum_normalization+b1d.sum_normalization))
        self.assertTrue(numpy.allclose(g1d.sum_normalization2, a1d.sum_normalization2+b1d.sum_normalization2))
        self.assertTrue(numpy.allclose(g1d.count, a1d.count+b1d.count))
        self.assertTrue(numpy.allclose(g1d.radial, a1d.radial))
        self.assertFalse(numpy.allclose(g1d.intensity, a1d.intensity))
        self.assertFalse(numpy.allclose(g1d.std, a1d.std))
        self.assertFalse(numpy.allclose(g1d.sem, a1d.sem))
        self.assertFalse(numpy.allclose(g1d.sigma, a1d.sigma))

        # same with 2D arrays
        a2d = self.ai.integrate2d(
            self.img, 40, 36, method=method, error_model="poisson"
        )
        b2d = self.ai.integrate2d(
            numpy.ones_like(self.img), 40, 36, method=method, error_model="poisson"
        )

        c2d = a2d + b2d
        self.assertTrue(numpy.allclose(c2d.sum_signal, a2d.sum_signal + b2d.sum_signal))
        self.assertTrue(
            numpy.allclose(c2d.sum_variance, a2d.sum_variance + b2d.sum_variance)
        )
        self.assertTrue(numpy.allclose(c2d.sum_normalization, b2d.sum_normalization))
        self.assertTrue(numpy.allclose(c2d.sum_normalization2, b2d.sum_normalization2))
        self.assertTrue(numpy.allclose(c2d.count, b2d.count))
        self.assertTrue(numpy.allclose(c2d.radial, b2d.radial))
        self.assertTrue(numpy.all(c2d.intensity >= a2d.intensity))
        self.assertTrue(numpy.all(c2d.intensity >= b2d.intensity))
        self.assertTrue(numpy.all(c2d.std >= a2d.std))
        self.assertTrue(numpy.all(c2d.std >= b2d.std))
        self.assertTrue(numpy.all(c2d.sem >= a2d.sem))
        self.assertTrue(numpy.all(c2d.sem >= b2d.sem))
        self.assertTrue(numpy.all(c2d.sigma >= a2d.sigma))
        self.assertTrue(numpy.all(c2d.sigma >= b2d.sigma))

        d2d = c2d - b2d
        self.assertTrue(numpy.allclose(d2d.sum_signal, a2d.sum_signal))
        self.assertTrue(
            numpy.allclose(d2d.sum_variance, a2d.sum_variance + 2 * b2d.sum_variance)
        )
        self.assertTrue(numpy.allclose(d2d.sum_normalization, b2d.sum_normalization))
        self.assertTrue(numpy.allclose(d2d.sum_normalization2, b2d.sum_normalization2))
        self.assertTrue(numpy.allclose(d2d.count, b2d.count))
        self.assertTrue(numpy.allclose(d2d.radial, b2d.radial))
        self.assertTrue(numpy.allclose(d2d.intensity, a2d.intensity))
        self.assertTrue(numpy.all(d2d.std >= a2d.std))
        self.assertTrue(numpy.all(d2d.std >= b2d.std))
        self.assertTrue(numpy.all(d2d.sem >= a2d.sem))
        self.assertTrue(numpy.all(d2d.sem >= b2d.sem))
        self.assertTrue(numpy.all(d2d.sigma >= a2d.sigma))
        self.assertTrue(numpy.all(d2d.sigma >= b2d.sigma))

        e2d = copy.deepcopy(a2d)
        e2d += b2d
        self.assertTrue(numpy.allclose(e2d.sum_signal, c2d.sum_signal))
        self.assertTrue(numpy.allclose(e2d.sum_variance, c2d.sum_variance))
        self.assertTrue(numpy.allclose(e2d.sum_normalization, c2d.sum_normalization))
        self.assertTrue(numpy.allclose(e2d.sum_normalization2, c2d.sum_normalization2))
        self.assertTrue(numpy.allclose(e2d.count, c2d.count))
        self.assertTrue(numpy.allclose(e2d.radial, c2d.radial))
        self.assertTrue(numpy.allclose(e2d.intensity, c2d.intensity))
        self.assertTrue(numpy.allclose(e2d.std, c2d.std))
        self.assertTrue(numpy.allclose(e2d.sem, c2d.sem))
        self.assertTrue(numpy.allclose(e2d.sigma, c2d.sigma))

        f2d = copy.deepcopy(c2d)
        f2d -= b2d
        self.assertTrue(numpy.allclose(f2d.sum_signal, a2d.sum_signal))
        self.assertTrue(numpy.allclose(f2d.sum_variance, d2d.sum_variance))
        self.assertTrue(numpy.allclose(f2d.sum_normalization, a2d.sum_normalization))
        self.assertTrue(numpy.allclose(f2d.sum_normalization2, a2d.sum_normalization2))
        self.assertTrue(numpy.allclose(f2d.count, a2d.count))
        self.assertTrue(numpy.allclose(f2d.radial, a2d.radial))
        self.assertTrue(numpy.allclose(f2d.intensity, a2d.intensity))
        self.assertTrue(numpy.allclose(f2d.std, d2d.std))
        self.assertTrue(numpy.allclose(f2d.sem, d2d.sem))
        self.assertTrue(numpy.allclose(f2d.sigma, d2d.sigma))

        g2d = a2d.union(b2d)
        self.assertTrue(numpy.allclose(g2d.sum_signal, a2d.sum_signal+b2d.sum_signal))
        self.assertTrue(numpy.allclose(g2d.sum_variance, a2d.sum_variance+b2d.sum_variance))
        self.assertTrue(numpy.allclose(g2d.sum_normalization, a2d.sum_normalization+b2d.sum_normalization))
        self.assertTrue(numpy.allclose(g2d.sum_normalization2, a2d.sum_normalization2+b2d.sum_normalization2))
        self.assertTrue(numpy.allclose(g2d.count, a2d.count+b2d.count))
        self.assertTrue(numpy.allclose(g2d.radial, a2d.radial))
        self.assertFalse(numpy.allclose(g2d.intensity, a2d.intensity))
        self.assertFalse(numpy.allclose(g2d.std, a2d.std))
        self.assertFalse(numpy.allclose(g2d.sem, a2d.sem))
        self.assertFalse(numpy.allclose(g2d.sigma, a2d.sigma))


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestContainer))
    return testsuite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
