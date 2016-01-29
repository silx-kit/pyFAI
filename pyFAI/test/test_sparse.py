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

from __future__ import absolute_import, division, print_function

__doc__ = """Test suites for sparse matrix multiplication modules"""
__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "29/01/2016"


import unittest
import numpy
import os
import sys
import time
from .utilstest import UtilsTest, getLogger
logger = getLogger(__file__)
from .. import load
from ..ext import splitBBox
from ..ext import splitBBoxCSR
from ..ext import splitBBoxLUT
import fabio


class TestSparseBBox(unittest.TestCase):
    """Test Azimuthal integration based sparse matrix multiplication methods
    Bounding box pixel splitting
    """

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.N = 1000
        self.unit = "2th_deg"
        self.ai = load(UtilsTest.getimage("1893/Pilatus1M.poni"))
        self.data = fabio.open(UtilsTest.getimage("1883/Pilatus1M.edf")).data
        self.ref = self.ai.integrate1d(self.data, self.N, correctSolidAngle=False, unit=self.unit, method="splitBBox")[1]
        self.cython = splitBBox.histoBBox1d(self.data, self.ai._ttha, self.ai._dttha, bins=self.N)

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.N = self.unit = self.ai = None
        self.data = self.ref = self.cython = None

    def test_LUT(self):
        obt = self.ai.integrate1d(self.data, self.N, correctSolidAngle=False, unit=self.unit, method="LUT")[1]
        logger.debug("delta on global result: %s" % (abs(obt - self.ref) / self.ref).max())
        self.assert_(numpy.allclose(obt, self.ref))

        cython = self.ai._lut_integrator.integrate(self.data)
        for ref, obt in zip(self.cython, cython):
            logger.debug("delta on cython result: %s" % (abs(obt - ref) / ref).max())
            self.assert_(numpy.allclose(obt, ref))

    def test_CSR(self):
        obt = self.ai.integrate1d(self.data, self.N, correctSolidAngle=False, unit=self.unit, method="CSR")[1]
        logger.debug("delta on global result: %s" % (abs(obt - self.ref) / self.ref).max())
        self.assert_(numpy.allclose(obt, self.ref))

        cython = self.ai._csr_integrator.integrate(self.data)
        for ref, obt in zip(self.cython, cython):
            logger.debug("delta on cython result: %s" % (abs(obt - ref) / ref).max())
            self.assert_(numpy.allclose(obt, ref))


def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(TestSparseBBox("test_LUT"))
    testsuite.addTest(TestSparseBBox("test_CSR"))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
