#!/usr/bin/python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/pyFAI/pyFAI
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

__doc__ = "test suite for average library"
__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "27/07/2016"

import unittest
import numpy
import os
import fabio
from .utilstest import UtilsTest, getLogger
from .. import average

logger = getLogger(__file__)


class TestAverage(unittest.TestCase):
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

    def test_average_dark(self):
        """
        Some testing for dark averaging
        """
        one = average.average_dark([self.dark])
        self.assertEqual(abs(self.dark - one).max(), 0, "data are the same")

        two = average.average_dark([self.dark, self.dark])
        self.assertEqual(abs(self.dark - two).max(), 0, "data are the same: mean test")

        three = average.average_dark([numpy.ones_like(self.dark), self.dark, numpy.zeros_like(self.dark)], "median")
        self.assertEqual(abs(self.dark - three).max(), 0, "data are the same: median test")

        four = average.average_dark([numpy.ones_like(self.dark), self.dark, numpy.zeros_like(self.dark)], "min")
        self.assertEqual(abs(numpy.zeros_like(self.dark) - four).max(), 0, "data are the same: min test")

        five = average.average_dark([numpy.ones_like(self.dark), self.dark, numpy.zeros_like(self.dark)], "max")
        self.assertEqual(abs(numpy.ones_like(self.dark) - five).max(), 0, "data are the same: max test")

        six = average.average_dark([numpy.ones_like(self.dark), self.dark, numpy.zeros_like(self.dark), self.dark, self.dark], "median", .001)
        self.assert_(abs(self.dark - six).max() < 1e-4, "data are the same: test threshold")
        if fabio.hexversion < 262147:
            logger.error("Error: the version of the FabIO library is too old: %s, please upgrade to 0.4+. Skipping test for now", fabio.version)
            return
        seven = average.average_images([self.raw], darks=[self.dark], flats=[self.flat], threshold=0, output=self.tmp_file)
        self.assert_(abs(numpy.ones_like(self.dark) - fabio.open(seven).data).mean() < 1e-2, "average_images")


def suite():
    testsuite = unittest.TestSuite()
    test_names = unittest.getTestCaseNames(TestAverage, "test")
    for test in test_names:
        testsuite.addTest(TestAverage(test))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
    UtilsTest.clean_up()
