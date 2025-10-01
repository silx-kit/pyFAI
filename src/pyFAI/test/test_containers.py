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
__date__ = "01/10/2025"

import unittest
import numpy
import logging
logger = logging.getLogger(__name__)
import fabio
from .utilstest import UtilsTest
from .. import load as pyFAI_load
from .. import containers


class TestContainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.img = fabio.open(UtilsTest.getimage("moke.tif")).data
        cls.ai = pyFAI_load({
    "poni_version": 2.1,
    "detector": "Detector",
    "detector_config": {
      "pixel1": 1e-4,
      "pixel2": 1e-4,
      "max_shape": [
        500,
        600
      ],
      "orientation": 3
    },
    "dist": 0.1,
    "poni1": 0.03,
    "poni2": 0.03,
    "rot1": 0.0,
    "rot2": 0.0,
    "rot3": 0.0,
    "wavelength": 1.0178021533473887e-10
  })
    @classmethod

    def tearDownClass(cls):
        cls.img = cls.ai = None

    def test_rebin1d(self):
        method = ("no", "histogram", "cython")
        res2d = self.ai.integrate2d(self.img, 500, 360, method=method, error_model="poisson")
        ref1d = self.ai.integrate1d(self.img, 500, method=method, error_model="poisson")
        res1d = containers.rebin1d(res2d)
        self.assertTrue(numpy.allclose(res1d[0], ref1d[0]), "radial matches")
        self.assertTrue(numpy.allclose(res1d[1], ref1d[1]), "intensity matches")
        self.assertTrue(numpy.allclose(res1d[2], ref1d[2]), "sem matches")

    def test_symmetrize(self):
        res2d = self.ai.integrate2d(self.img, 500, 360, error_model="poisson", radial_range=(0,12), unit="2th_deg")
        sym = containers.symmetrize(res2d)
        self.assertAlmostEqual(res2d.intensity.mean(), sym.intensity.mean(), places=0)

    def test_maths(self):
        method = ("no", "histogram", "cython")
        a1d = self.ai.integrate1d(self.img, 10, method=method, error_model="poisson")
        b1d = self.ai.integrate1d(numpy.ones_like(self.img), 10, method=method, error_model="poisson")
        print("a1d", a1d.intensity)
        print("b1d", b1d.intensity)

        c1d = a1d + b1d
        print("c1d", c1d.intensity)
        print("a1d", a1d.intensity)

        print(type(c1d), type(b1d), type(c1d)==type(b1d), type(c1d)==type(c1d))
        print(c1d.unit, b1d.unit,c1d.unit== b1d.unit)
        print(c1d.sum_normalization, b1d.sum_normalization, numpy.allclose(c1d.sum_normalization, b1d.sum_normalization))
        
        d1d = c1d - b1d
        
        self.assertTrue(numpy.allclose(c1d.sum_signal, a1d.sum_signal+b1d.sum_signal))
        self.assertTrue(numpy.allclose(c1d.sum_variance, a1d.sum_variance+b1d.sum_variance))
        self.assertTrue(numpy.allclose(c1d.sum_normalization, b1d.sum_normalization))
        self.assertTrue(numpy.allclose(c1d.sum_normalization2, b1d.sum_normalization2))
        self.assertTrue(numpy.allclose(c1d.count, b1d.count))
        self.assertTrue(numpy.allclose(c1d.radial, b1d.radial))
        self.assertTrue(numpy.all(c1d.intensity>a1d.intensity))
        self.assertTrue(numpy.all(c1d.intensity>=b1d.intensity))
        self.assertTrue(numpy.all(c1d.std>=a1d.std))
        self.assertTrue(numpy.all(c1d.std>=b1d.std))
        self.assertTrue(numpy.all(c1d.sem>=a1d.sem))
        self.assertTrue(numpy.all(c1d.sem>=b1d.sem))


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestContainer))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
