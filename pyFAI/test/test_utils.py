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


from __future__ import division, print_function, absolute_import

__doc__ = "test suite for utilities library"
__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "19/07/2017"

import unittest
import numpy
import os
from .utilstest import UtilsTest, getLogger
logger = getLogger(__file__)
from .. import utils
from .. import _version

import scipy.ndimage

# TODO Test:
# relabel
# DONE:
# # gaussian_filter
# # binning
# # unbinning
# # shift
# # shiftFFT
# # measure_offset


class TestUtils(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.unbinned = numpy.random.random((64, 32))
        self.dark = self.unbinned.astype("float32")
        self.flat = 1 + numpy.random.random((64, 32))
        self.raw = self.flat + self.dark
        self.tmp_file = os.path.join(UtilsTest.tempdir, "testUtils_average.edf")

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.dark = self.flat = self.raw = self.tmp_file = None

    def test_binning(self):
        """
        test the binning and unbinning functions
        """
        binned = utils.binning(self.unbinned, (4, 2))
        self.assertEqual(binned.shape, (64 // 4, 32 // 2), "binned size is OK")
        unbinned = utils.unBinning(binned, (4, 2))
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
        self.assertTrue(abs(utils.shiftFFT(ref, delta) - res).max() < 1e-12, "shift with FFTs works")
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

    def test_set(self):
        s = utils.FixedParameters()
        self.assertEqual(len(s), 0, "initial set is empty")
        s.add_or_discard("a", True)
        self.assertEqual(len(s), 1, "a is in set")
        s.add_or_discard("a", None)
        self.assertEqual(len(s), 1, "set is untouched")
        s.add_or_discard("a", False)
        self.assertEqual(len(s), 0, "set is empty again")
        s.add_or_discard("a", None)
        self.assertEqual(len(s), 0, "set is untouched")
        s.add_or_discard("a", False)
        self.assertEqual(len(s), 0, "set is still empty")

    def test_expand2d(self):
        vect = numpy.arange(10.)
        size2 = 11
        self.assertTrue((numpy.outer(vect, numpy.ones(size2)) == utils.expand2d(vect, size2, False)).all(), "horizontal vector expand")
        self.assertTrue((numpy.outer(numpy.ones(size2), vect) == utils.expand2d(vect, size2, True)).all(), "vertical vector expand")

    def test_hexversion(self):
        # print(_version, type(_version))
        self.assertEqual(_version.calc_hexversion(1), 1 << 24, "Major is OK")
        self.assertEqual(_version.calc_hexversion(0, 1), 1 << 16, "Minor is OK")
        self.assertEqual(_version.calc_hexversion(0, 0, 1), 1 << 8, "Micro is OK")
        self.assertEqual(_version.calc_hexversion(0, 0, 0, 1), 1 << 4, "Release level is OK")
        self.assertEqual(_version.calc_hexversion(0, 0, 0, 0, 1), 1, "Serial is OK")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestUtils))
    return testsuite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
    UtilsTest.clean_up()
