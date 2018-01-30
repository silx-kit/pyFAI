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


from __future__ import division, print_function, absolute_import

"""Test suite for math utilities library"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "11/01/2018"

import unittest
import numpy
import os
import logging
from . import utilstest
logger = logging.getLogger(__name__)
from .. import utils

import scipy.ndimage


_ROUND_FFT_VALUES = [
    (2, 2), (3, 3), (5, 5), (7, 7), (11, 11), (13, 13), (17, 18), (19, 20),
    (23, 24), (29, 30), (31, 32), (37, 39), (41, 42), (43, 44), (47, 48),
    (53, 54), (59, 60), (61, 63), (67, 70), (71, 72), (73, 75), (79, 80),
    (83, 84), (89, 90), (97, 98), (101, 104), (103, 104), (107, 108),
    (109, 110), (113, 117), (127, 128), (131, 132), (137, 140), (139, 140),
    (149, 150), (151, 154), (157, 160), (163, 165), (167, 168), (173, 175),
    (179, 180), (181, 182), (191, 192), (193, 195), (197, 198), (199, 200),
    (211, 216), (223, 224), (227, 231), (229, 231), (233, 234), (239, 240),
    (241, 243), (251, 252), (257, 260), (263, 264), (269, 270), (271, 273),
    (277, 280), (281, 288), (283, 288), (293, 294), (307, 308), (311, 312),
    (313, 315), (317, 320), (331, 336), (337, 343), (347, 350), (349, 350),
    (353, 360), (359, 360), (367, 375), (373, 375), (379, 384), (383, 384),
    (389, 390), (397, 400), (401, 405), (409, 416), (419, 420), (421, 432),
    (431, 432), (433, 440), (439, 440), (443, 448), (449, 450), (457, 462),
    (461, 462), (463, 468), (467, 468), (479, 480), (487, 490), (491, 495),
    (499, 500), (503, 504), (509, 512), (521, 525), (523, 525), (541, 546),
    (547, 550), (557, 560), (563, 567), (569, 576), (571, 576), (577, 585),
    (587, 588), (593, 594), (599, 600), (601, 616), (607, 616), (613, 616),
    (617, 624), (619, 624), (631, 637), (641, 648), (643, 648), (647, 648),
    (653, 660), (659, 660), (661, 672), (673, 675), (677, 686), (683, 686),
    (691, 693), (701, 702), (709, 720), (719, 720), (727, 728), (733, 735),
    (739, 750), (743, 750), (751, 756), (757, 768), (761, 768), (769, 770),
    (773, 780), (787, 792), (797, 800), (809, 810), (811, 819), (821, 825),
    (823, 825), (827, 832), (829, 832), (839, 840), (853, 864), (857, 864),
    (859, 864), (863, 864), (877, 880), (881, 882), (883, 891), (887, 891),
    (907, 910), (911, 924), (919, 924), (929, 936), (937, 945), (941, 945),
    (947, 960), (953, 960), (967, 972), (971, 972), (977, 980), (983, 990),
    (991, 1000), (997, 1000)]


class TestMathUtil(utilstest.ParametricTestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.unbinned = numpy.random.random((64, 32))
        self.dark = self.unbinned.astype("float32")
        self.flat = 1 + numpy.random.random((64, 32))
        self.raw = self.flat + self.dark
        self.tmp_file = os.path.join(utilstest.UtilsTest.tempdir, "testUtils_average.edf")

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.dark = self.flat = self.raw = self.tmp_file = None

    def test_round_fft(self):
        """Test some rounding values."""
        for value, expected in _ROUND_FFT_VALUES:
            with self.subTest(value=value, expected=expected):
                self.assertEqual(utils.round_fft(value), expected)

    def test_binning(self):
        """
        test the binning and unbinning functions
        """
        binned = utils.binning(self.unbinned, (4, 2))
        self.assertEqual(binned.shape, (64 // 4, 32 // 2), "binned size is OK")
        unbinned = utils.unbinning(binned, (4, 2))
        self.assertEqual(unbinned.shape, self.unbinned.shape, "unbinned size is OK")
        self.assertAlmostEqual(unbinned.sum(), self.unbinned.sum(), 2, "content is the same")

    def test_shift(self):
        """
        Some testing for image shifting and offset measurement functions.
        """
        ref = numpy.ones((11, 12))
        ref[2, 3] = 5
        res = numpy.ones((11, 12))
        res[5, 7] = 5
        delta = (5 - 2, 7 - 3)
        self.assertTrue(abs(utils.shift(ref, delta) - res).max() < 1e-12, "shift with integers works")
        self.assertTrue(abs(utils.shift_fft(ref, delta) - res).max() < 1e-12, "shift with FFTs works")
        self.assertTrue(utils.measure_offset(res, ref) == delta, "measure offset works")

    def test_gaussian_filter(self):
        """
        Check gaussian filters applied via FFT
        """
        for sigma in [2, 9.0 / 8.0]:
            for mode in ["wrap", "reflect", "constant", "nearest", "mirror"]:
                blurred1 = scipy.ndimage.filters.gaussian_filter(self.flat, sigma, mode=mode)
                blurred2 = utils.gaussian_filter(self.flat, sigma, mode=mode, use_scipy=False)
                delta = abs((blurred1 - blurred2) / (blurred1)).max()
                logger.info("Error for gaussian blur sigma: %s with mode %s is %s", sigma, mode, delta)
                self.assertTrue(delta < 6e-5, "Gaussian blur sigma: %s  in %s mode are the same, got %s" % (sigma, mode, delta))

    def test_expand2d(self):
        vect = numpy.arange(10.)
        size2 = 11
        self.assertTrue((numpy.outer(vect, numpy.ones(size2)) == utils.expand2d(vect, size2, False)).all(), "horizontal vector expand")
        self.assertTrue((numpy.outer(numpy.ones(size2), vect) == utils.expand2d(vect, size2, True)).all(), "vertical vector expand")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestMathUtil))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
    utilstest.UtilsTest.clean_up()
