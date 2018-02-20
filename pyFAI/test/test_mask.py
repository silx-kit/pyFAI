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

"""Test suite for masked arrays"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "20/02/2018"


import unittest
import numpy
import logging
import fabio
from .utilstest import UtilsTest
logger = logging.getLogger(__name__)

if logger.getEffectiveLevel() <= logging.INFO:
    import pylab

from ..third_party import six
from .. import load, detectors
from ..azimuthalIntegrator import AzimuthalIntegrator


class TestMask(unittest.TestCase):
    dataFile = "testMask.edf"
    poniFile = "Pilatus1M.poni"

    def setUp(self):
        """Download files"""
        self.dataFile = UtilsTest.getimage(self.__class__.dataFile)
        self.poniFile = UtilsTest.getimage(self.__class__.poniFile)
        self.ai = load(self.poniFile)
        self.data = fabio.open(self.dataFile).data
        self.mask = self.data < 0

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.dataFile = self.data = self.ai = self.mask = self.poniFile = None

    def test_mask_hist(self):
        """
        The masked image has a masked ring around 1.5deg with value -10
        without mask the pixels should be at -10 ; with mask they are at 0
        """
        meth = "cython"
        x1 = self.ai.integrate1d(self.data, 1000, unit="2th_deg", method=meth)
        x2 = self.ai.integrate1d(self.data, 1000, mask=self.mask, unit="2th_deg", method=meth)
        x3 = self.ai.integrate1d(self.data, 1000, dummy=-20.0, delta_dummy=19.5, unit="2th_deg", method=meth)
        res1 = numpy.interp(1.5, *x1)
        res2 = numpy.interp(1.5, *x2)
        res3 = numpy.interp(1.5, *x3)
        if logger.getEffectiveLevel() == logging.DEBUG:
            pylab.plot(*x1, label="no mask")
            pylab.plot(*x2, label="with mask")
            pylab.plot(*x3, label="with dummy")
            pylab.title("test_mask_splitBBox")
            pylab.legend()
            pylab.show()
            six.moves.input()

        self.assertAlmostEqual(res1, -10., 1, msg="Without mask the bad pixels are around -10 (got %.4f)" % res1)
        self.assertAlmostEqual(res2, 0, 1, msg="With mask the bad pixels are actually Nan (got %.4f)" % res2)
        self.assertAlmostEqual(res3, -20., 4, msg="Without mask but dummy=-20 the dummy pixels are actually at -20 (got % .4f)" % res3)

    def test_mask_splitBBox(self):
        """
        The masked image has a masked ring around 1.5deg with value -10
        without mask the pixels should be at -10 ; with mask they are at 0
        """
        meth = "splitbbox"
        x1 = self.ai.integrate1d(self.data, 1000, unit="2th_deg", method=meth)
        x2 = self.ai.integrate1d(self.data, 1000, mask=self.mask, unit="2th_deg", method=meth)
        x3 = self.ai.integrate1d(self.data, 1000, dummy=-20.0, delta_dummy=19.5, unit="2th_deg", method=meth)
        res1 = numpy.interp(1.5, *x1)
        res2 = numpy.interp(1.5, *x2)
        res3 = numpy.interp(1.5, *x3)
        if logger.getEffectiveLevel() == logging.DEBUG:
            pylab.plot(*x1, label="no mask")
            pylab.plot(*x2, label="with mask")
            pylab.plot(*x3, label="with dummy")
            pylab.title("test_mask_splitBBox")
            pylab.legend()
            pylab.show()
            six.moves.input()

        self.assertAlmostEqual(res1, -10., 1, msg="Without mask the bad pixels are around -10 (got %.4f)" % res1)
        self.assertAlmostEqual(res2, 0, 1, msg="With mask the bad pixels are actually Nan (got %.4f)" % res2)
        self.assertAlmostEqual(res3, -20., 4, msg="Without mask but dummy=-20 the dummy pixels are actually at -20 (got % .4f)" % res3)

    def test_mask_splitfull(self):
        """
        The masked image has a masked ring around 1.5deg with value -10
        without mask the pixels should be at -10 ; with mask they are at 0
        """
        meth = "splitpixel"
        x1 = self.ai.integrate1d(self.data, 1000, unit="2th_deg", method=meth)
        x2 = self.ai.integrate1d(self.data, 1000, mask=self.mask, unit="2th_deg", method=meth)
        x3 = self.ai.integrate1d(self.data, 1000, dummy=-20.0, delta_dummy=19.5, unit="2th_deg", method=meth)
        res1 = numpy.interp(1.5, *x1)
        res2 = numpy.interp(1.5, *x2)
        res3 = numpy.interp(1.5, *x3)
        if logger.getEffectiveLevel() == logging.DEBUG:
            pylab.plot(*x1, label="no mask")
            pylab.plot(*x2, label="with mask")
            pylab.plot(*x3, label="with dummy")
            pylab.title("test_mask_splitBBox")
            pylab.legend()
            pylab.show()
            six.moves.input()

        self.assertAlmostEqual(res1, -10., 1, msg="Without mask the bad pixels are around -10 (got %.4f)" % res1)
        self.assertAlmostEqual(res2, 0, 1, msg="With mask the bad pixels are actually Nan (got %.4f)" % res2)
        self.assertAlmostEqual(res3, -20., 4, msg="Without mask but dummy=-20 the dummy pixels are actually at -20 (got % .4f)" % res3)

    def test_mask_LUT(self):
        """
        The masked image has a masked ring around 1.5deg with value -10
        without mask the pixels should be at -10 ; with mask they are at 0
        """
        meth = "lut"
        x1 = self.ai.integrate1d(self.data, 1000, unit="2th_deg", method=meth)
        x2 = self.ai.integrate1d(self.data, 1000, mask=self.mask, unit="2th_deg", method=meth)
        x3 = self.ai.integrate1d(self.data, 1000, mask=numpy.zeros(shape=self.mask.shape, dtype="uint8"), dummy=-20.0, delta_dummy=19.5, unit="2th_deg", method=meth)
        res1 = numpy.interp(1.5, *x1)
        res2 = numpy.interp(1.5, *x2)
        res3 = numpy.interp(1.5, *x3)
        if logger.getEffectiveLevel() == logging.DEBUG:
            pylab.plot(*x1, label="nomask")
            pylab.plot(*x2, label="mask")
            pylab.plot(*x3, label="dummy")
            pylab.legend()
            pylab.show()
            six.moves.input()

        self.assertAlmostEqual(res1, -10., 1, msg="Without mask the bad pixels are around -10 (got %.4f)" % res1)
        self.assertAlmostEqual(res2, 0, 1, msg="With mask the bad pixels are actually Nan (got %.4f)" % res2)
        self.assertAlmostEqual(res3, -20., 4, msg="Without mask but dummy=-20 the dummy pixels are actually at -20 (got % .4f)" % res3)

    def test_mask_CSR(self):
        """
        The masked image has a masked ring around 1.5deg with value -10
        without mask the pixels should be at -10 ; with mask they are at 0
        """
        meth = "csr"
        x1 = self.ai.integrate1d(self.data, 1000, unit="2th_deg", method=meth)
        x2 = self.ai.integrate1d(self.data, 1000, mask=self.mask, unit="2th_deg", method=meth)
        x3 = self.ai.integrate1d(self.data, 1000, mask=numpy.zeros(shape=self.mask.shape, dtype="uint8"), dummy=-20.0, delta_dummy=19.5, unit="2th_deg", method=meth)
        res1 = numpy.interp(1.5, *x1)
        res2 = numpy.interp(1.5, *x2)
        res3 = numpy.interp(1.5, *x3)
        if logger.getEffectiveLevel() == logging.DEBUG:
            pylab.plot(*x1, label="nomask")
            pylab.plot(*x2, label="mask")
            pylab.plot(*x3, label="dummy")
            pylab.legend()
            pylab.show()
            six.moves.input()

        self.assertAlmostEqual(res1, -10., 1, msg="Without mask the bad pixels are around -10 (got %.4f)" % res1)
        self.assertAlmostEqual(res2, 0, 1, msg="With mask the bad pixels are actually Nan (got %.4f)" % res2)
        self.assertAlmostEqual(res3, -20., 4, msg="Without mask but dummy=-20 the dummy pixels are actually at -20 (got % .4f)" % res3)

    @unittest.skipIf(UtilsTest.opencl is False, "User request to skip OpenCL tests")
    def test_mask_LUT_OCL(self):
        """
        The masked image has a masked ring around 1.5deg with value -10
        without mask the pixels should be at -10 ; with mask they are at 0
        """
        meth = "lut_ocl"
        x1 = self.ai.integrate1d(self.data, 1000, unit="2th_deg", method=meth)
        x2 = self.ai.integrate1d(self.data, 1000, mask=self.mask, unit="2th_deg", method=meth)
        x3 = self.ai.integrate1d(self.data, 1000, dummy=-20.0, delta_dummy=19.5, unit="2th_deg", method=meth)
        res1 = numpy.interp(1.5, *x1)
        res2 = numpy.interp(1.5, *x2)
        res3 = numpy.interp(1.5, *x3)
        if logger.getEffectiveLevel() == logging.DEBUG:
            pylab.plot(*x1, label="nomask")
            pylab.plot(*x2, label="mask")
            pylab.plot(*x3, label="dummy")
            pylab.legend()
            pylab.show()
            six.moves.input()

        self.assertAlmostEqual(res1, -10., 1, msg="Without mask the bad pixels are around -10 (got %.4f)" % res1)
        self.assertAlmostEqual(res2, 0, 1, msg="With mask the bad pixels are actually around 0 (got %.4f)" % res2)
        self.assertAlmostEqual(res3, -20., 4, msg="Without mask but dummy=-20 the dummy pixels are actually at -20 (got % .4f)" % res3)

    @unittest.skipIf(UtilsTest.opencl is False, "User request to skip OpenCL tests")
    def test_mask_CSR_OCL(self):
        """
        The masked image has a masked ring around 1.5deg with value -10
        without mask the pixels should be at -10 ; with mask they are at 0
        """
        meth = "CSR_ocl"
        x1 = self.ai.integrate1d(self.data, 1000, unit="2th_deg", method=meth)
        x2 = self.ai.integrate1d(self.data, 1000, mask=self.mask, unit="2th_deg", method=meth)
        x3 = self.ai.integrate1d(self.data, 1000, dummy=-20.0, delta_dummy=19.5, unit="2th_deg", method=meth)
        res1 = numpy.interp(1.5, *x1)
        res2 = numpy.interp(1.5, *x2)
        res3 = numpy.interp(1.5, *x3)
        if logger.getEffectiveLevel() == logging.DEBUG:
            pylab.plot(*x1, label="nomask")
            pylab.plot(*x2, label="mask")
            pylab.plot(*x3, label="dummy")
            pylab.legend()
            pylab.show()
            six.moves.input()

        self.assertAlmostEqual(res1, -10., 1, msg="Without mask the bad pixels are around -10 (got %.4f)" % res1)
        self.assertAlmostEqual(res2, 0, 1, msg="With mask the bad pixels are actually around 0 (got %.4f)" % res2)
        self.assertAlmostEqual(res3, -20., 4, msg="Without mask but dummy=-20 the dummy pixels are actually at -20 (got % .4f)" % res3)


class TestMaskBeamstop(unittest.TestCase):
    """
    Test for https://github.com/silx-kit/pyFAI/issues/76
    """
    dataFile = "mock.tif"

    def setUp(self):
        """
        Download files
        Create a mask for tth<3.7 deg
        """
        self.dataFile = UtilsTest.getimage(self.__class__.dataFile)
        detector = detectors.Detector(pixel1=0.0001, pixel2=0.0001)
        self.ai = AzimuthalIntegrator(dist=0.1, poni1=0.03, poni2=0.03, detector=detector)
        self.data = fabio.open(self.dataFile).data
        self.tth, self.I = self.ai.integrate1d(self.data, 1000, unit="2th_deg")
        self.mask = self.ai.ttha < numpy.deg2rad(3.7)

    def test_nomask(self):
        """
        without mask, tth value should start at 0
        """
        if logger.getEffectiveLevel() == logging.DEBUG:
            pylab.plot(self.tth, self.I, label="nomask")
            pylab.legend()
            pylab.show()
            six.moves.input()

        self.assertAlmostEqual(self.tth[0], 0.0, 1, "tth without mask starts at 0")

    def test_mask_splitBBox(self):
        """
        With a mask with and without limits
        """
        tth, _ = self.ai.integrate1d(self.data, 1000, mask=self.mask, unit="2th_deg", method="splitBBox")
        self.assertAlmostEqual(tth[0], 3.7, 1, msg="tth range starts at 3.7 (got %.4f)" % tth[0])
        tth, _ = self.ai.integrate1d(self.data, 1000, mask=self.mask, unit="2th_deg", method="splitBBox", radial_range=[1, 10])
        self.assertAlmostEqual(tth[0], 1.0, 1, msg="tth range should start at 1.0 (got %.4f)" % tth[0])

    def test_mask_LUT(self):
        """
        With a mask with and without limits
        """
        tth, _ = self.ai.integrate1d(self.data, 1000, mask=self.mask, unit="2th_deg", method="LUT")
        self.assertAlmostEqual(tth[0], 3.7, 1, msg="tth range starts at 3.7 (got %.4f)" % tth[0])
        tth, _ = self.ai.integrate1d(self.data, 1000, mask=self.mask, unit="2th_deg", method="LUT", radial_range=[1, 10])
        self.assertAlmostEqual(tth[0], 1.0, 1, msg="tth range should start at 1.0 (got %.4f)" % tth[0])

    @unittest.skipIf(UtilsTest.opencl is False, "User request to skip OpenCL tests")
    def test_mask_LUT_OCL(self):
        """
        With a mask with and without limits
        """
        tth, _ = self.ai.integrate1d(self.data, 1000, mask=self.mask, unit="2th_deg", method="lut_ocl")
        self.assertTrue(tth[0] > 3.5, msg="tth range starts at 3.7 (got %.4f)" % tth[0])
        tth, _ = self.ai.integrate1d(self.data, 1000, mask=self.mask, unit="2th_deg", method="lut_ocl", radial_range=[1, 10])
        self.assertAlmostEqual(tth[0], 1.0, 1, msg="tth range should start at 1.0 (got %.4f)" % tth[0])

    def test_nomask_LUT(self):
        """
        without mask, tth value should start at 0
        """
        tth, _ = self.ai.integrate1d(self.data, 1000, unit="2th_deg", method="lut")
        self.assertAlmostEqual(tth[0], 0.0, 1, msg="tth range starts at 3.7 (got %.4f)" % tth[0])
        tth, _ = self.ai.integrate1d(self.data, 1000, unit="2th_deg", method="lut", radial_range=[1, 10])
        self.assertAlmostEqual(tth[0], 1.0, 1, msg="tth range should start at 1.0 (got %.4f)" % tth[0])

    @unittest.skipIf(UtilsTest.opencl is False, "User request to skip OpenCL tests")
    def test_nomask_LUT_OCL(self):
        """
        without mask, tth value should start at 0
        """
        tth, _ = self.ai.integrate1d(self.data, 1000, unit="2th_deg", method="lut_ocl")
        self.assertAlmostEqual(tth[0], 0.0, 1, msg="tth range starts at 3.7 (got %.4f)" % tth[0])
        tth, _ = self.ai.integrate1d(self.data, 1000, unit="2th_deg", method="lut_ocl", radial_range=[1, 10])
        self.assertAlmostEqual(tth[0], 1.0, 1, msg="tth range should start at 1.0 (got %.4f)" % tth[0])


def suite():
    testsuite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite.addTest(loader(TestMask))
    testsuite.addTest(loader(TestMaskBeamstop))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
