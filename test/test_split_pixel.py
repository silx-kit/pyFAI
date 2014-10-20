#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal Integration
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


"""
Test suites for pixel splitting scheeme balidation

see debug_split_pixel.py for visual validation
"""

import unittest, numpy, os, sys, time, numpy
from utilstest import UtilsTest, getLogger, Rwp
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]


class TestSplitPixel(unittest.TestCase):
    """

    """
    N = 10000
    import pyFAI, numpy
    img = numpy.zeros((512, 512))
    for i in range(1, 6):img[i * 100, i * 100] = 1
    det = pyFAI.detectors.Detector(1e-4, 1e-4)
    det.shape = (512, 512)
    ai = pyFAI.AzimuthalIntegrator(1, detector=det)
    results = {}
    for i, meth in enumerate(["numpy", "cython", "splitbbox", "splitpixel", "csr_no", "csr_bbox", "csr_full"]):
        results[meth] = ai.integrate1d(img, 10000, method=meth, unit="2th_deg")
        ai.reset()

    def test_no_split(self):
        """
        Validate that all non splitting algo give the same result...
        """
        thres=7
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


def test_suite_all_split():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestSplitPixel("test_no_split"))
    testSuite.addTest(TestSplitPixel("test_split_bbox"))
    testSuite.addTest(TestSplitPixel("test_split_full"))
    return testSuite


if __name__ == '__main__':
    mysuite = test_suite_all_split()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
