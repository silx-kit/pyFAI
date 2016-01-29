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

__doc__ = """Test suites for pixel splitting scheme validation

see sandbox/debug_split_pixel.py for visual validation
"""
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
from .utilstest import UtilsTest, getLogger, Rwp
logger = getLogger(__file__)
from ..azimuthalIntegrator import AzimuthalIntegrator
from ..detectors import Detector


class TestSplitPixel(unittest.TestCase):
    """

    """
    def setUp(self):
        unittest.TestCase.setUp(self)
        N = 10000
        img = numpy.zeros((512, 512))
        for i in range(1, 6):
            img[i * 100, i * 100] = 1
        det = Detector(1e-4, 1e-4)
        det.shape = (512, 512)
        ai = AzimuthalIntegrator(1, detector=det)
        self.results = {}
        for i, meth in enumerate(["numpy", "cython", "splitbbox", "splitpixel", "csr_no", "csr_bbox", "csr_full"]):
            self.results[meth] = ai.integrate1d(img, 10000, method=meth, unit="2th_deg")
            ai.reset()

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.results = None

    def test_no_split(self):
        """
        Validate that all non splitting algo give the same result...
        """
        thres = 7
        self.assert_(Rwp(self.results["numpy"], self.results["cython"]) < thres, "Cython/Numpy")
        self.assert_(Rwp(self.results["csr_no"], self.results["cython"]) < thres, "Cython/CSR")
        self.assert_(Rwp(self.results["csr_no"], self.results["numpy"]) < thres, "CSR/numpy")
        self.assert_(Rwp(self.results["splitbbox"], self.results["numpy"]) > thres, "splitbbox/Numpy")
        self.assert_(Rwp(self.results["splitpixel"], self.results["numpy"]) > thres, "splitpixel/Numpy")
        self.assert_(Rwp(self.results["csr_bbox"], self.results["numpy"]) > thres, "csr_bbox/Numpy")
        self.assert_(Rwp(self.results["csr_full"], self.results["numpy"]) > thres, "csr_full/Numpy")
        self.assert_(Rwp(self.results["splitbbox"], self.results["cython"]) > thres, "splitbbox/cython")
        self.assert_(Rwp(self.results["splitpixel"], self.results["cython"]) > thres, "splitpixel/cython")
        self.assert_(Rwp(self.results["csr_bbox"], self.results["cython"]) > thres, "csr_bbox/cython")
        self.assert_(Rwp(self.results["csr_full"], self.results["cython"]) > thres, "csr_full/cython")
        self.assert_(Rwp(self.results["splitbbox"], self.results["csr_no"]) > thres, "splitbbox/csr_no")
        self.assert_(Rwp(self.results["splitpixel"], self.results["csr_no"]) > thres, "splitpixel/csr_no")
        self.assert_(Rwp(self.results["csr_bbox"], self.results["csr_no"]) > thres, "csr_bbox/csr_no")
        self.assert_(Rwp(self.results["csr_full"], self.results["csr_no"]) > thres, "csr_full/csr_no")

    def test_split_bbox(self):
        """
        Validate that all bbox splitting algo give all the same result...
        """
        thres = 7
        self.assert_(Rwp(self.results["csr_bbox"], self.results["splitbbox"]) < thres, "csr_bbox/splitbbox")
        self.assert_(Rwp(self.results["numpy"], self.results["splitbbox"]) > thres, "numpy/splitbbox")
        self.assert_(Rwp(self.results["cython"], self.results["splitbbox"]) > thres, "cython/splitbbox")
        self.assert_(Rwp(self.results["splitpixel"], self.results["splitbbox"]) > thres, "splitpixel/splitbbox")
        self.assert_(Rwp(self.results["csr_no"], self.results["splitbbox"]) > thres, "csr_no/splitbbox")
        self.assert_(Rwp(self.results["csr_full"], self.results["splitbbox"]) > thres, "csr_full/splitbbox")
        self.assert_(Rwp(self.results["numpy"], self.results["csr_bbox"]) > thres, "numpy/csr_bbox")
        self.assert_(Rwp(self.results["cython"], self.results["csr_bbox"]) > thres, "cython/csr_bbox")
        self.assert_(Rwp(self.results["splitpixel"], self.results["csr_bbox"]) > thres, "splitpixel/csr_bbox")
        self.assert_(Rwp(self.results["csr_no"], self.results["csr_bbox"]) > thres, "csr_no/csr_bbox")
        self.assert_(Rwp(self.results["csr_full"], self.results["csr_bbox"]) > thres, "csr_full/csr_bbox")

    def test_split_full(self):
        """
        Validate that all full splitting algo give all the same result...
        """
        thres = 7
        self.assert_(Rwp(self.results["csr_full"], self.results["splitpixel"]) < thres, "csr_full/splitpixel")
        self.assert_(Rwp(self.results["numpy"], self.results["splitpixel"]) > thres, "numpy/splitpixel")
        self.assert_(Rwp(self.results["cython"], self.results["splitpixel"]) > thres, "cython/splitpixel")
        self.assert_(Rwp(self.results["splitbbox"], self.results["splitpixel"]) > thres, "splitpixel/splitpixel")
        self.assert_(Rwp(self.results["csr_no"], self.results["splitpixel"]) > thres, "csr_no/splitpixel")
        self.assert_(Rwp(self.results["csr_bbox"], self.results["splitpixel"]) > thres, "csr_full/splitpixel")
        self.assert_(Rwp(self.results["numpy"], self.results["csr_full"]) > thres, "numpy/csr_full")
        self.assert_(Rwp(self.results["cython"], self.results["csr_full"]) > thres, "cython/csr_full")
        self.assert_(Rwp(self.results["splitbbox"], self.results["csr_full"]) > thres, "splitpixel/csr_full")
        self.assert_(Rwp(self.results["csr_no"], self.results["csr_full"]) > thres, "csr_no/csr_full")
        self.assert_(Rwp(self.results["csr_bbox"], self.results["csr_full"]) > thres, "csr_full/csr_full")


def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(TestSplitPixel("test_no_split"))
    testsuite.addTest(TestSplitPixel("test_split_bbox"))
    testsuite.addTest(TestSplitPixel("test_split_full"))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
    UtilsTest.clean_up()
