#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

from __future__ import division, print_function, absolute_import
"test suite for utilities library"
__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "07/05/2015"


import unittest
import numpy
import sys
import os
import fabio
import tempfile
if __name__ == '__main__':
    import pkgutil
    __path__ = pkgutil.extend_path([os.path.dirname(__file__)], "pyFAI.test")
from .utilstest import UtilsTest, getLogger, recursive_delete
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]
import pyFAI.utils

# if logger.getEffectiveLevel() <= logging.INFO:
#    from pyFAI.gui_utils import pylab
import scipy.ndimage

# TODO Test:
# gaussian_filter
# relabel
# boundingBox
# removeSaturatedPixel
# DONE:
# # binning
# # unbinning
# # shift
# # shiftFFT
# # measure_offset
# # averageDark
# # averageImages


class TestUtils(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.unbinned = numpy.random.random((64, 32))
        self.dark = unbinned.astype("float32")
        self.flat = 1 + numpy.random.random((64, 32))
        self.raw = flat + dark
        self.tmp_file = os.path.join(UtilsTest.tempdir, "testUtils_average.edf")

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.dark = self.flat = self.raw = self.tmp_file = None

    def test_binning(self):
        """
        test the binning and unbinning functions
        """
        binned = pyFAI.utils.binning(self.unbinned, (4, 2))
        self.assertEqual(binned.shape, (64 // 4, 32 // 2), "binned size is OK")
        unbinned = pyFAI.utils.unBinning(binned, (4, 2))
        self.assertEqual(unbinned.shape, self.unbinned.shape, "unbinned size is OK")
        self.assertAlmostEqual(unbinned.sum(), self.unbinned.sum(), 2, "content is the same")

    def test_averageDark(self):
        """
        Some testing for dark averaging
        """
        one = pyFAI.utils.averageDark([self.dark])
        self.assertEqual(abs(self.dark - one).max(), 0, "data are the same")

        two = pyFAI.utils.averageDark([self.dark, self.dark])
        self.assertEqual(abs(self.dark - two).max(), 0, "data are the same: mean test")

        three = pyFAI.utils.averageDark([numpy.ones_like(self.dark), self.dark, numpy.zeros_like(self.dark)], "median")
        self.assertEqual(abs(self.dark - three).max(), 0, "data are the same: median test")

        four = pyFAI.utils.averageDark([numpy.ones_like(self.dark), self.dark, numpy.zeros_like(self.dark)], "min")
        self.assertEqual(abs(numpy.zeros_like(self.dark) - four).max(), 0, "data are the same: min test")

        five = pyFAI.utils.averageDark([numpy.ones_like(self.dark), self.dark, numpy.zeros_like(self.dark)], "max")
        self.assertEqual(abs(numpy.ones_like(self.dark) - five).max(), 0, "data are the same: max test")

        six = pyFAI.utils.averageDark([numpy.ones_like(self.dark), self.dark, numpy.zeros_like(self.dark), self.dark, self.dark], "median", .001)
        self.assert_(abs(self.dark - six).max() < 1e-4, "data are the same: test threshold")

        seven = pyFAI.utils.averageImages([self.raw], darks=[self.dark], flats=[self.flat], threshold=0, output=self.tmp_file)
        self.assert_(abs(numpy.ones_like(self.dark) - fabio.open(seven).data).mean() < 1e-2, "averageImages")

    def test_shift(self):
        """
        Some testing for image shifting and offset measurement functions.
        """
        ref = numpy.ones((11, 12))
        ref[2, 3] = 5
        res = numpy.ones((11, 12))
        res[5, 7] = 5
        delta = (5 - 2, 7 - 3)
        self.assert_(abs(pyFAI.utils.shift(ref, delta) - res).max() < 1e-12, "shift with integers works")
        self.assert_(abs(pyFAI.utils.shiftFFT(ref, delta) - res).max() < 1e-12, "shift with FFTs works")
        self.assert_(pyFAI.utils.measure_offset(res, ref) == delta, "measure offset works")

    def test_gaussian_filter(self):
        """
        Check gaussian filters applied via FFT
        """
        for sigma in [2, 9.0 / 8.0]:
            for mode in ["wrap", "reflect", "constant", "nearest", "mirror"]:
                blurred1 = scipy.ndimage.filters.gaussian_filter(self.flat, sigma, mode=mode)
                blurred2 = pyFAI.utils.gaussian_filter(self.flat, sigma, mode=mode)
                delta = abs((blurred1 - blurred2) / (blurred1)).max()
                logger.info("Error for gaussian blur sigma: %s with mode %s is %s" % (sigma, mode, delta))
                self.assert_(delta < 6e-5, "Gaussian blur sigma: %s  in %s mode are the same, got %s" % (sigma, mode, delta))

    def test_set(self):
        s = pyFAI.utils.FixedParameters()
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
        self.assert_((numpy.outer(vect, numpy.ones(size2)) == pyFAI.utils.expand2d(vect, size2, False)).all(), "horizontal vector expand")
        self.assert_((numpy.outer(numpy.ones(size2), vect) == pyFAI.utils.expand2d(vect, size2, True)).all(), "vertical vector expand")


def test_suite_all_Utils():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestUtils("test_binning"))
    testSuite.addTest(TestUtils("test_averageDark"))
    testSuite.addTest(TestUtils("test_shift"))
    testSuite.addTest(TestUtils("test_gaussian_filter"))
    testSuite.addTest(TestUtils("test_set"))
    testSuite.addTest(TestUtils("test_expand2d"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_all_Utils()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
    UtilsTest.clean_up()
