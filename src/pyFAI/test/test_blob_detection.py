#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2024 European Synchrotron Radiation Facility, Grenoble, France
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

"""Test suite for blob detection cython accelerated code"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "08/04/2024"

import unittest
import numpy
import logging
logger = logging.getLogger(__name__)
from ..detectors import detector_factory
from ..integrator.azimuthal import AzimuthalIntegrator
from ..blob_detection import BlobDetection, local_max
from ..ext import _blob
from ..ext import morphology
from scipy import ndimage


def image_test_rings():
    rings = 10
    mod = 50
    detector = detector_factory("Titan")
    sigma = detector.pixel1 * 4
    shape = detector.max_shape
    ai = AzimuthalIntegrator(detector=detector)
    ai.setFit2D(1000, 1000, 1000)
    r = ai.rArray(shape)
    r_max = r.max()
    chi = ai.chiArray(shape)
    img = numpy.zeros(shape)
    modulation = (1 + numpy.sin(5 * r + chi * mod))
    for radius in numpy.linspace(0, r_max, rings):
        img += numpy.exp(-(r - radius) ** 2 / (2 * (sigma * sigma)))
    return img * modulation


class TestBlobDetection(unittest.TestCase):
    img = None

    def setUp(self):
        if self.img is None:
            self.img = image_test_rings()

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.img = None

    def test_local_max(self):
        bd = BlobDetection(self.img)
        bd._one_octave(shrink=False, refine=False, n_5=False)
        self.assertTrue(numpy.all(_blob.local_max(bd.dogs, bd.cur_mask, False) ==
                                      local_max(bd.dogs, bd.cur_mask, False)), "max test, 3x3x3")
        self.assertTrue(numpy.all(_blob.local_max(bd.dogs, bd.cur_mask, True) ==
                                      local_max(bd.dogs, bd.cur_mask, True)), "max test, 3x5x5")

class TestMorphology(unittest.TestCase):
    shape = (101, 103)
    img = None

    @classmethod
    def setUpClass(cls):
        yc = cls.shape[0] // 2
        xc = cls.shape[0] // 2
        y, x = numpy.ogrid[:cls.shape[0],:cls.shape[1]]
        r = 0.4 * min(cls.shape)
        r2 = r * r
        y -= yc
        x -= xc
        cls.img = (x * x + y * y < r2).astype(numpy.int8)

    @classmethod
    def tearDownClass(cls):
        cls.img = None

    def test_morphology(self):
        ref_d = ndimage.binary_dilation(self.img.astype(bool)).astype("int8")
        res_d = morphology.binary_dilation(self.img)
        self.assertTrue(numpy.allclose(ref_d, res_d), "binary dilation")
        res_e = morphology.binary_erosion(res_d)
        self.assertTrue(numpy.allclose(self.img, res_e), "binary erosion")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestBlobDetection))
    testsuite.addTest(loader(TestMorphology))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
