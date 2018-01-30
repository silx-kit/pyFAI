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

"""Test suite for convolution cython code"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jérôme.Kieffer@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "10/01/2018"

import unittest
import numpy
import logging
logger = logging.getLogger(__name__)
from ..ext import _convolution
import scipy.ndimage
import scipy.misc
import scipy.signal


class TestConvolution(unittest.TestCase):
    def setUp(self):
        self.sigma = 1
        self.width = 8 * self.sigma + 1
        if self.width % 2 == 0:
            self.width += 1
        self.gauss = scipy.signal.gaussian(self.width, self.sigma)
        self.gauss /= self.gauss.sum()
        if "ascent" in dir(scipy.misc):
            self.lena = scipy.misc.ascent().astype("float32")
        else:
            self.lena = scipy.misc.lena().astype("float32")

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.lena = self.gauss = self.sigma = self.width = None

    def test_gaussian(self):
        gauss = _convolution.gaussian(self.sigma)
        self.assertTrue(numpy.allclose(gauss, self.gauss), "gaussian curves are the same")

    def test_horizontal_convolution(self):
        gauss = self.gauss.astype(numpy.float32)
        ref = scipy.ndimage.filters.convolve1d(self.lena, self.gauss, axis=-1)
        obt = _convolution.horizontal_convolution(self.lena, gauss)
        self.assertTrue(numpy.allclose(ref, obt), "horizontal filtered images are the same")

    def test_vertical_convolution(self):
        gauss = self.gauss.astype(numpy.float32)
        ref = scipy.ndimage.filters.convolve1d(self.lena, self.gauss, axis=0)
        obt = _convolution.vertical_convolution(self.lena, gauss)
        self.assertTrue(numpy.allclose(ref, obt), "vertical filtered images are the same")

    def test_gaussian_filter(self):
        ref = scipy.ndimage.filters.gaussian_filter(self.lena, self.sigma)
        obt = _convolution.gaussian_filter(self.lena, self.sigma)
        self.assertTrue(numpy.allclose(ref, obt), "gaussian filtered images are the same")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestConvolution))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
