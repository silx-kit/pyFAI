#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: PyFAI: https://github.com/kif/pyFAI
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France

"test suite for convolution cython code"

__author__ = "Jérôme Kieffer"
__contact__ = "Jérôme Kieffer"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "06/03/2014"

import sys
import unittest
import numpy
from utilstest import getLogger  # UtilsTest, Rwp, getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]
from pyFAI import _convolution
import scipy.ndimage, scipy.misc, scipy.signal

class TestConvolution(unittest.TestCase):
    def setUp(self):
        self.sigma = 1
        self.width = 8*self.sigma+1
        if self.width%2==0:
            self.width+=1
        self.gauss = scipy.signal.gaussian(self.width, self.sigma)
        self.gauss/=self.gauss.sum()
        self.lena = scipy.misc.lena().astype("float32")

    def test_gaussian(self):
        gauss = _convolution.gaussian(self.sigma)
        self.assert_(numpy.allclose(gauss,self.gauss), "gaussian curves are the same")

    def test_horizontal_convolution(self):
        gauss = self.gauss.astype(numpy.float32)
        ref = scipy.ndimage.filters.convolve1d(self.lena, self.gauss, axis= -1)
        obt = _convolution.horizontal_convolution(self.lena, gauss)
        self.assert_(numpy.allclose(ref, obt), "horizontal filtered images are the same")

    def test_vertical_convolution(self):
        gauss = self.gauss.astype(numpy.float32)
        ref = scipy.ndimage.filters.convolve1d(self.lena, self.gauss, axis=0)
        obt = _convolution.vertical_convolution(self.lena, gauss)
        self.assert_(numpy.allclose(ref, obt), "vertical filtered images are the same")

    def test_gaussian_filter(self):
        ref = scipy.ndimage.filters.gaussian_filter(self.lena, self.sigma)
        obt = _convolution.gaussian_filter(self.lena, self.sigma)
        self.assert_(numpy.allclose(ref, obt), "gaussian filtered images are the same")
        

def test_suite_all_convolution():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestConvolution("test_horizontal_convolution"))
    testSuite.addTest(TestConvolution("test_vertical_convolution"))
    testSuite.addTest(TestConvolution("test_gaussian"))
    testSuite.addTest(TestConvolution("test_gaussian_filter"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_all_convolution()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
