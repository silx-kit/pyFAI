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

"""test suite for average library
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "09/07/2025"

import unittest
import numpy
import logging
from .utilstest import UtilsTest
from ..crystallography import resolution

logger = logging.getLogger(__name__)


class TestCrystallography(unittest.TestCase):

    def test_constant(self):
        ref = [1] * 11
        c = resolution.Constant(180/numpy.pi)
        self.assertTrue(isinstance(c.__repr__(), str))
        self.assertTrue(numpy.allclose(c.fwhm(numpy.linspace(0,1,11)), ref))
        self.assertTrue(isinstance(c.sigma(1), float))

    def test_caglioti(self):
        ref = [0.04246609, 0.05619075, 0.07367654, 0.09299528, 0.11344279,
       0.1347813 , 0.15696523, 0.18004253, 0.20411724, 0.22933564,
       0.255883  ]
        c = resolution.Caglioti(1,1e-1,1e-2)
        self.assertTrue(isinstance(c.__repr__(), str))
        self.assertTrue(numpy.allclose(c.sigma(numpy.linspace(0,1,11)), ref))
        self.assertTrue(isinstance(c.fwhm(1), float))

    def test_chernyshov(self):
        ref = [0.44740802, 0.4452937 , 0.43897227, 0.4285082 , 0.41400832,
       0.39562113, 0.37353566, 0.34798047, 0.31922269, 0.28756785,
       0.25336168]
        c = resolution.Chernyshov(1,1e-1,1e-2)
        self.assertTrue(isinstance(c.__repr__(), str))
        self.assertTrue(numpy.allclose(c.sigma(numpy.linspace(0,1,11)), ref))
        self.assertTrue(isinstance(c.fwhm(1), float))

    def test_langford(self):
        ref = [8.48615349, 4.23249535, 2.80994112, 2.09520315, 1.6636391 ,
       1.37371653, 1.16479474, 1.00657162, 0.8822346 , 0.7817234 ]
        c = resolution.Langford(1e-3, 1e-2, 1e-1, 1)
        self.assertTrue(isinstance(c.__repr__(), str))
        self.assertTrue(numpy.allclose(c.sigma(numpy.linspace(0.1,1,10)), ref))
        self.assertTrue(isinstance(c.fwhm(1), float))


def suite():
    testsuite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite.addTest(loader(TestCrystallography))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
    UtilsTest.clean_up()
