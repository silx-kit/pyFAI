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
import os
import ast
import logging
import fabio
from .utilstest import UtilsTest
from ..crystallography import resolution

logger = logging.getLogger(__name__)


class TestCrystallography(unittest.TestCase):
    def test_caglioti(self):
        ref = [0.4246609 , 0.42890762, 0.43749603, 0.4507456 , 0.46925191,
       0.4939733 , 0.52638373, 0.56874106, 0.62457202, 0.69959768,
       0.80363747]
        c = resolution.Caglioti(1,1e-1,1e-2)
        str(c)
        self.assertTrue(numpy.allclose(c.sigma(numpy.linspace(0,1,11)), ref))
        c.fwhm(1)

    def test_chernyshov(self):
        ref = [0.44740802, 0.4452937 , 0.43897227, 0.4285082 , 0.41400832,
       0.39562113, 0.37353566, 0.34798047, 0.31922269, 0.28756785,
       0.25336168]
        c = resolution.Chernyshov(1,1e-1,1e-2)
        str(c)
        self.assertTrue(numpy.allclose(c.sigma(numpy.linspace(0,1,11)), ref))
        c.fwhm(1)



def suite():
    testsuite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite.addTest(loader(TestCrystallography))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
    UtilsTest.clean_up()
