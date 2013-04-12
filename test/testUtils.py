#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: FasT Azimuthal integration
#             https://github.com/kif/pyFAI
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
"test suite for utilities library"

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "11/04/2013"


import unittest
import numpy
import logging
import sys
import fabio
from utilstest import UtilsTest, getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]
import pyFAI.utils

if logger.getEffectiveLevel() <= logging.INFO:
    import pylab


# TODO Test:
# gaussian_filter
# shift
# relabel
# averageDark
# averageImages
# boundingBox
# removeSaturatedPixel
# DONE:
# # binning
# # unbinning

class test_utils(unittest.TestCase):
    unbinned = numpy.random.random((64, 32))
    dark = unbinned.astype("float32")
    flat = 1 + numpy.random.random((64, 32))
    raw = flat + dark
    def setUp(self):
        """Download files"""
        pass
    def test_binning(self):
        """
        test the binning and unbinning functions
        """
        binned = pyFAI.utils.binning(self.unbinned, (4, 2))
        self.assertEqual(binned.shape, (64 / 4, 32 / 2), "binned size is OK")
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

        three = pyFAI.utils.averageDark([numpy.ones_like(self.dark), self.dark, numpy.zeros_like(self.dark) ], "median")
        self.assertEqual(abs(self.dark - three).max(), 0, "data are the same: median test")

        four = pyFAI.utils.averageDark([numpy.ones_like(self.dark), self.dark, numpy.zeros_like(self.dark) ], "min")
        self.assertEqual(abs(numpy.zeros_like(self.dark) - four).max(), 0, "data are the same: min test")

        five = pyFAI.utils.averageDark([numpy.ones_like(self.dark), self.dark, numpy.zeros_like(self.dark) ], "max")
        self.assertEqual(abs(numpy.ones_like(self.dark) - five).max(), 0, "data are the same: max test")

        six = pyFAI.utils.averageDark([numpy.ones_like(self.dark), self.dark, numpy.zeros_like(self.dark), self.dark, self.dark ], "median", .001)
        self.assert_(abs(self.dark - six).max() < 1e-4, "data are the same: test threshold")

        seven = pyFAI.utils.averageImages([self.raw], darks=[self.dark], flats=[self.flat], threshold=0)
        self.assert_(abs(numpy.ones_like(self.dark) - fabio.open(seven).data).mean() < 1e-2, "averageImages")

    def test_shift(self):
        """
        Some testing for image shifting and offset measurment functions.
        """
        pass


def test_suite_all_Utils():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_utils("test_binning"))
    testSuite.addTest(test_utils("test_averageDark"))
    testSuite.addTest(test_utils("test_shift"))
    return testSuite

if __name__ == '__main__':

    mysuite = test_suite_all_Utils()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
