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

"""Test suites for pixel splitting scheme validation

see sandbox/debug_split_pixel.py for visual validation
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "30/11/2018"

import unittest
import numpy
import logging
from .utilstest import UtilsTest
logger = logging.getLogger(__name__)
from ..azimuthalIntegrator import AzimuthalIntegrator
from ..detectors import Detector
from ..utils import mathutil
from ..ext import splitBBox, splitPixel


class TestSplitPixel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestSplitPixel, cls).setUpClass()
        img = numpy.zeros((512, 512))
        for i in range(1, 6):
            img[i * 100, i * 100] = 1
        det = Detector(1e-4, 1e-4)
        det.shape = (512, 512)
        ai = AzimuthalIntegrator(1, detector=det)
        cls.results = {}
        for i, meth in enumerate(["numpy", "cython", "splitbbox", "splitpixel", "csr_no", "csr_bbox", "csr_full"]):
            cls.results[meth] = ai.integrate1d(img, 10000, method=meth, unit="2th_deg")
            ai.reset()

    @classmethod
    def tearDownClass(cls):
        super(TestSplitPixel, cls).tearDownClass()
        cls.results = None

    def test_no_split(self):
        """
        Validate that all non splitting algo give the same result...
        """
        thres = 7
        self.assertTrue(mathutil.rwp(self.results["numpy"], self.results["cython"]) < thres, "Cython/Numpy")
        self.assertTrue(mathutil.rwp(self.results["csr_no"], self.results["cython"]) < thres, "Cython/CSR")
        self.assertTrue(mathutil.rwp(self.results["csr_no"], self.results["numpy"]) < thres, "CSR/numpy")
        self.assertTrue(mathutil.rwp(self.results["splitbbox"], self.results["numpy"]) > thres, "splitbbox/Numpy")
        self.assertTrue(mathutil.rwp(self.results["splitpixel"], self.results["numpy"]) > thres, "splitpixel/Numpy")
        self.assertTrue(mathutil.rwp(self.results["csr_bbox"], self.results["numpy"]) > thres, "csr_bbox/Numpy")
        self.assertTrue(mathutil.rwp(self.results["csr_full"], self.results["numpy"]) > thres, "csr_full/Numpy")
        self.assertTrue(mathutil.rwp(self.results["splitbbox"], self.results["cython"]) > thres, "splitbbox/cython")
        self.assertTrue(mathutil.rwp(self.results["splitpixel"], self.results["cython"]) > thres, "splitpixel/cython")
        self.assertTrue(mathutil.rwp(self.results["csr_bbox"], self.results["cython"]) > thres, "csr_bbox/cython")
        self.assertTrue(mathutil.rwp(self.results["csr_full"], self.results["cython"]) > thres, "csr_full/cython")
        self.assertTrue(mathutil.rwp(self.results["splitbbox"], self.results["csr_no"]) > thres, "splitbbox/csr_no")
        self.assertTrue(mathutil.rwp(self.results["splitpixel"], self.results["csr_no"]) > thres, "splitpixel/csr_no")
        self.assertTrue(mathutil.rwp(self.results["csr_bbox"], self.results["csr_no"]) > thres, "csr_bbox/csr_no")
        self.assertTrue(mathutil.rwp(self.results["csr_full"], self.results["csr_no"]) > thres, "csr_full/csr_no")

    def test_split_bbox(self):
        """
        Validate that all bbox splitting algo give all the same result...
        """
        thres = 7
        self.assertTrue(mathutil.rwp(self.results["csr_bbox"], self.results["splitbbox"]) < thres, "csr_bbox/splitbbox")
        self.assertTrue(mathutil.rwp(self.results["numpy"], self.results["splitbbox"]) > thres, "numpy/splitbbox")
        self.assertTrue(mathutil.rwp(self.results["cython"], self.results["splitbbox"]) > thres, "cython/splitbbox")
        self.assertTrue(mathutil.rwp(self.results["splitpixel"], self.results["splitbbox"]) > thres, "splitpixel/splitbbox")
        self.assertTrue(mathutil.rwp(self.results["csr_no"], self.results["splitbbox"]) > thres, "csr_no/splitbbox")
        self.assertTrue(mathutil.rwp(self.results["csr_full"], self.results["splitbbox"]) > thres, "csr_full/splitbbox")
        self.assertTrue(mathutil.rwp(self.results["numpy"], self.results["csr_bbox"]) > thres, "numpy/csr_bbox")
        self.assertTrue(mathutil.rwp(self.results["cython"], self.results["csr_bbox"]) > thres, "cython/csr_bbox")
        self.assertTrue(mathutil.rwp(self.results["splitpixel"], self.results["csr_bbox"]) > thres, "splitpixel/csr_bbox")
        self.assertTrue(mathutil.rwp(self.results["csr_no"], self.results["csr_bbox"]) > thres, "csr_no/csr_bbox")
        self.assertTrue(mathutil.rwp(self.results["csr_full"], self.results["csr_bbox"]) > thres, "csr_full/csr_bbox")

    def test_split_full(self):
        """
        Validate that all full splitting algo give all the same result...
        """
        thres = 7
        self.assertTrue(mathutil.rwp(self.results["csr_full"], self.results["splitpixel"]) < thres, "csr_full/splitpixel")
        self.assertTrue(mathutil.rwp(self.results["numpy"], self.results["splitpixel"]) > thres, "numpy/splitpixel")
        self.assertTrue(mathutil.rwp(self.results["cython"], self.results["splitpixel"]) > thres, "cython/splitpixel")
        self.assertTrue(mathutil.rwp(self.results["splitbbox"], self.results["splitpixel"]) > thres, "splitpixel/splitpixel")
        self.assertTrue(mathutil.rwp(self.results["csr_no"], self.results["splitpixel"]) > thres, "csr_no/splitpixel")
        self.assertTrue(mathutil.rwp(self.results["csr_bbox"], self.results["splitpixel"]) > thres, "csr_full/splitpixel")
        self.assertTrue(mathutil.rwp(self.results["numpy"], self.results["csr_full"]) > thres, "numpy/csr_full")
        self.assertTrue(mathutil.rwp(self.results["cython"], self.results["csr_full"]) > thres, "cython/csr_full")
        self.assertTrue(mathutil.rwp(self.results["splitbbox"], self.results["csr_full"]) > thres, "splitpixel/csr_full")
        self.assertTrue(mathutil.rwp(self.results["csr_no"], self.results["csr_full"]) > thres, "csr_no/csr_full")
        self.assertTrue(mathutil.rwp(self.results["csr_bbox"], self.results["csr_full"]) > thres, "csr_full/csr_full")


class TestSplitBBoxNg(unittest.TestCase):
    """Test the equivalence of the historical SplitBBox with the one propagating 
    the variance"""
    @classmethod
    def setUpClass(cls):
        super(TestSplitBBoxNg, cls).setUpClass()
        det = Detector.factory("Pilatus 100k")
        shape = det.shape
        # The randomness of the image is not correlated to bug #1021
        img = numpy.random.randint(0, 65000, numpy.prod(shape))

        ai = AzimuthalIntegrator(1, detector=det)
        ai.wavelength = 1e-10
        tth = ai.center_array(shape, unit="2th_rad", scale=False).ravel()
        dtth = ai.delta_array(shape, unit="2th_rad").ravel()
        chi = ai.chiArray(shape).ravel()
        dchi = ai.deltaChi(shape).ravel()
        pos = ai.corner_array(shape, unit="2th_deg", use_cython=True, scale=False)
        cls.results = {}
        # Legacy implementation:
        cls.results["histoBBox2d_legacy"] = splitBBox.histoBBox2d(img,
                                                                  tth,
                                                                  dtth,
                                                                  chi,
                                                                  dchi,
                                                                  empty=-1)
        cls.results["histoBBox2d_ng"] = splitBBox.histoBBox2d_ng(img,
                                                                 tth,
                                                                 dtth,
                                                                 chi,
                                                                 dchi,
                                                                 variance=img,
                                                                 empty=-1)
        # Legacy implementation:
        cls.results["fullSplit2D_legacy"] = splitPixel.fullSplit2D(pos,
                                                                   img,
                                                                   bins=(100, 36),
                                                                   empty=-1)
        cls.results["fullSplit2D_ng"] = splitPixel.pseudoSplit2D_ng(pos,
                                                                    img,
                                                                    bins=(100, 36),
                                                                    variance=img,
                                                                    empty=-1)
        cls.img = img

    @classmethod
    def tearDownClass(cls):
        super(TestSplitBBoxNg, cls).tearDownClass()
        cls.results = None
        cls.img = None

    def test_split_bbox_2d(self):
        # radial position:
        tth_legacy = self.results["histoBBox2d_legacy"][1]
        tth_ng = self.results["histoBBox2d_ng"][2]
        self.assertEqual(abs(tth_legacy - tth_ng).max(), 0, "radial position is the same")

        # azimuthal position:
        chi_legacy = self.results["histoBBox2d_legacy"][2]
        chi_ng = self.results["histoBBox2d_ng"][3]
        self.assertEqual(abs(chi_legacy - chi_ng).max(), 0, "azimuthal position is the same")

        # pixel count:
        count_legacy = self.results["histoBBox2d_legacy"][4]
        count_ng = self.results["histoBBox2d_ng"][4]["count"]

        if abs(count_ng).max() == 0:
            print(splitBBox)
            print(count_legacy)
            print(count_ng)
            print("prop", self.results["histoBBox2d_ng"][4])
            print("pos1", self.results["histoBBox2d_ng"][3])
            print("pos0", self.results["histoBBox2d_ng"][2])
            print("err", self.results["histoBBox2d_ng"][1])
            print("int", self.results["histoBBox2d_ng"][0])
        self.assertEqual(abs(count_legacy - count_ng).max(), 0, "count is the same")
        # same for normalisation ... in this case
        count_ng = self.results["histoBBox2d_ng"][4]["norm"]
        self.assertEqual(abs(count_legacy - count_ng).max(), 0, "norm is old-count")

        # Weighted signal:
        weighted_legacy = self.results["histoBBox2d_legacy"][3]
        signal = self.results["histoBBox2d_ng"][4]["signal"]
        self.assertEqual(abs(signal - weighted_legacy).max(), 0, "Weighted is the same")

        # resulting intensity validation
        int_legacy = self.results["histoBBox2d_legacy"][0]
        int_ng = self.results["histoBBox2d_ng"][0]
        self.assertEqual(abs(int_legacy - int_ng).max(), 0, "intensity is the same")

    def test_split_pixel_2d(self):
        # radial position:
        tth_legacy = self.results["fullSplit2D_legacy"][1]
        tth_ng = self.results["fullSplit2D_ng"][2]
        self.assertEqual(abs(tth_legacy - tth_ng).max(), 0, "radial position is the same")

        # azimuthal position:
        chi_legacy = self.results["fullSplit2D_legacy"][2]
        chi_ng = self.results["fullSplit2D_ng"][3]
        self.assertEqual(abs(chi_legacy - chi_ng).max(), 0, "azimuthal position is the same")

        # pixel count:
        count_legacy = self.results["fullSplit2D_legacy"][4]
        count_ng = self.results["fullSplit2D_ng"][4]["count"]
        self.assertLess(abs(count_legacy - count_ng).mean(), 1, "count is almost the same")
        # same for normalisation ... in this case
        count_ng = self.results["fullSplit2D_ng"][4]["norm"]
        self.assertLess(abs(count_legacy - count_ng).mean(), 1, "norm is almost old-count")

#         # Weighted signal:
#         weighted_legacy = self.results["fullSplit2D_legacy"][3]
#         signal = self.results["fullSplit2D_ng"][4]["signal"]
#         print(weighted_legacy)
#         print(signal)
#         print(abs(signal / weighted_legacy).nanmean())
#         self.assertEqual(abs(signal - weighted_legacy).max(), 0, "Weighted is the same")

        # resulting intensity validation
#         int_legacy = self.results["fullSplit2D_legacy"][0]
#         int_ng = self.results["histoBBox2d_ng"][0]
#         print(int_legacy)
#         print(int_ng)
#         self.assertEqual(abs(int_legacy - int_ng).max(), 0, "intensity is the same")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestSplitPixel))
    testsuite.addTest(loader(TestSplitBBoxNg))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
    UtilsTest.clean_up()
