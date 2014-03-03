#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://forge.epn-campus.eu/projects/azimuthal
#
#    File: "$Id$"
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
"test suite for masked arrays"

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "28/06/2012"


import unittest
import numpy
import logging
import sys
import fabio
from utilstest import UtilsTest, getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]

if logger.getEffectiveLevel() <= logging.INFO:
    import pylab

class test_mask(unittest.TestCase):
    dataFile = "1894/testMask.edf"
    poniFile = "1893/Pilatus1M.poni"

    def setUp(self):
        """Download files"""
        self.dataFile = UtilsTest.getimage(self.__class__.dataFile)
        self.poniFile = UtilsTest.getimage(self.__class__.poniFile)
        self.ai = pyFAI.load(self.poniFile)
#        self.ai.mask = None
        self.data = fabio.open(self.dataFile).data
        self.mask = self.data < 0

    def test_mask_splitBBox(self):
        """
        The masked image has a masked ring around 1.5deg with value -10
        without mask the pixels should be at -10 ; with mask they are at 0
        """
        x1 = self.ai.integrate1d(self.data, 1000, unit="2th_deg")
        x2 = self.ai.integrate1d(self.data, 1000, mask=self.mask, unit="2th_deg")
        x3 = self.ai.integrate1d(self.data, 1000, dummy= -20.0, delta_dummy=19.5, unit="2th_deg")
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
            raw_input()

        self.assertAlmostEqual(res1, -10., 1, msg="Without mask the bad pixels are around -10 (got %.4f)" % res1)
        self.assertAlmostEqual(res2, 0., 4, msg="With mask the bad pixels are actually at 0 (got %.4f)" % res2)
        self.assertAlmostEqual(res3, -20., 4, msg="Without mask but dummy=-20 the dummy pixels are actually at -20 (got % .4f)" % res3)


    def test_mask_LUT(self):
        """
        The masked image has a masked ring around 1.5deg with value -10
        without mask the pixels should be at -10 ; with mask they are at 0
        """
        x1 = self.ai.xrpd_LUT(self.data, 1000)
#        print self.ai._lut_integrator.lut_checksum
        x2 = self.ai.xrpd_LUT(self.data, 1000, mask=self.mask)
#        print self.ai._lut_integrator.lut_checksum
        x3 = self.ai.xrpd_LUT(self.data, 1000, mask=numpy.zeros(shape=self.mask.shape, dtype="uint8"), dummy= -20.0, delta_dummy=19.5)
#        print self.ai._lut_integrator.lut_checksum
        res1 = numpy.interp(1.5, *x1)
        res2 = numpy.interp(1.5, *x2)
        res3 = numpy.interp(1.5, *x3)
        if logger.getEffectiveLevel() == logging.DEBUG:
            pylab.plot(*x1, label="nomask")
            pylab.plot(*x2, label="mask")
            pylab.plot(*x3, label="dummy")
            pylab.legend()
            pylab.show()
            raw_input()

        self.assertAlmostEqual(res1, -10., 1, msg="Without mask the bad pixels are around -10 (got %.4f)" % res1)
        self.assertAlmostEqual(res2, 0., 4, msg="With mask the bad pixels are actually at 0 (got %.4f)" % res2)
        self.assertAlmostEqual(res3, -20., 4, msg="Without mask but dummy=-20 the dummy pixels are actually at -20 (got % .4f)" % res3)

    def test_mask_LUT_OCL(self):
        """
        The masked image has a masked ring around 1.5deg with value -10
        without mask the pixels should be at -10 ; with mask they are at 0
        """
        x1 = self.ai.xrpd_LUT_OCL(self.data, 1000)
#        print self.ai._lut_integrator.lut_checksum
        x2 = self.ai.xrpd_LUT_OCL(self.data, 1000, mask=self.mask)
#        print self.ai._lut_integrator.lut_checksum
        x3 = self.ai.xrpd_LUT_OCL(self.data, 1000, dummy= -20.0, delta_dummy=19.5)
#        print self.ai._lut_integrator.lut_checksum
        res1 = numpy.interp(1.5, *x1)
        res2 = numpy.interp(1.5, *x2)
        res3 = numpy.interp(1.5, *x3)
        if logger.getEffectiveLevel() == logging.DEBUG:
            pylab.plot(*x1, label="nomask")
            pylab.plot(*x2, label="mask")
            pylab.plot(*x3, label="dummy")
            pylab.legend()
            pylab.show()
            raw_input()

        self.assertAlmostEqual(res1, -10., 1, msg="Without mask the bad pixels are around -10 (got %.4f)" % res1)
        self.assertAlmostEqual(res2, 0., 4, msg="With mask the bad pixels are actually at 0 (got %.4f)" % res2)
        self.assertAlmostEqual(res3, -20., 4, msg="Without mask but dummy=-20 the dummy pixels are actually at -20 (got % .4f)" % res3)

class test_mask_beamstop(unittest.TestCase):
    """
    Test for https://github.com/kif/pyFAI/issues/76
    """
    dataFile = "1788/moke.tif"

    def setUp(self):
        """
        Download files 
        Create a mask for tth<3.7 deg
        """
        self.dataFile = UtilsTest.getimage(self.__class__.dataFile)
        detector = pyFAI.detectors.Detector(pixel1=0.0001, pixel2=0.0001)
        self.ai = pyFAI.AzimuthalIntegrator(dist=0.1, poni1=0.03, poni2=0.03, detector=detector)
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
            raw_input()

        self.assertAlmostEqual(self.tth[0], 0.0, 1, "tth without mask starts at 0")

    def test_mask_splitBBox(self):
        """
        With a mask with and without limits
        """
        tth, I = self.ai.integrate1d(self.data, 1000, mask=self.mask, unit="2th_deg", method="splitBBox")
        self.assertAlmostEqual(tth[0], 3.7, 1, msg="tth range starts at 3.7 (got %.4f)" % tth[0])
        tth, I = self.ai.integrate1d(self.data, 1000, mask=self.mask, unit="2th_deg", method="splitBBox", radial_range=[1, 10])
        self.assertAlmostEqual(tth[0], 1.0, 1, msg="tth range should start at 1.0 (got %.4f)" % tth[0])

    def test_mask_LUT(self):
        """
        With a mask with and without limits
        """
        tth, I = self.ai.integrate1d(self.data, 1000, mask=self.mask, unit="2th_deg", method="LUT")
        self.assertAlmostEqual(tth[0], 3.7, 1, msg="tth range starts at 3.7 (got %.4f)" % tth[0])
        tth, I = self.ai.integrate1d(self.data, 1000, mask=self.mask, unit="2th_deg", method="LUT", radial_range=[1, 10])
        self.assertAlmostEqual(tth[0], 1.0, 1, msg="tth range should start at 1.0 (got %.4f)" % tth[0])

    def test_mask_LUT_OCL(self):
        """
        With a mask with and without limits
        """
        tth, I = self.ai.integrate1d(self.data, 1000, mask=self.mask, unit="2th_deg", method="lut_ocl")
        self.assert_(tth[0] > 3.5, msg="tth range starts at 3.7 (got %.4f)" % tth[0])
        tth, I = self.ai.integrate1d(self.data, 1000, mask=self.mask, unit="2th_deg", method="lut_ocl", radial_range=[1, 10])
        self.assertAlmostEqual(tth[0], 1.0, 1, msg="tth range should start at 1.0 (got %.4f)" % tth[0])

    def test_nomask_LUT(self):
        """
        without mask, tth value should start at 0
        """
        tth, I = self.ai.integrate1d(self.data, 1000, unit="2th_deg", method="lut")
        self.assertAlmostEqual(tth[0], 0.0, 1, msg="tth range starts at 3.7 (got %.4f)" % tth[0])
        tth, I = self.ai.integrate1d(self.data, 1000, unit="2th_deg", method="lut", radial_range=[1, 10])
        self.assertAlmostEqual(tth[0], 1.0, 1, msg="tth range should start at 1.0 (got %.4f)" % tth[0])


    def test_nomask_LUT_OCL(self):
        """
        without mask, tth value should start at 0
        """
        tth, I = self.ai.integrate1d(self.data, 1000, unit="2th_deg", method="lut_ocl")
        self.assertAlmostEqual(tth[0], 0.0, 1, msg="tth range starts at 3.7 (got %.4f)" % tth[0])
        tth, I = self.ai.integrate1d(self.data, 1000, unit="2th_deg", method="lut_ocl", radial_range=[1, 10])
        self.assertAlmostEqual(tth[0], 1.0, 1, msg="tth range should start at 1.0 (got %.4f)" % tth[0])


def test_suite_all_Mask():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_mask("test_mask_splitBBox"))
    testSuite.addTest(test_mask("test_mask_LUT"))
    testSuite.addTest(test_mask("test_mask_LUT_OCL"))
    testSuite.addTest(test_mask_beamstop("test_nomask"))
    testSuite.addTest(test_mask_beamstop("test_mask_splitBBox"))
    testSuite.addTest(test_mask_beamstop("test_mask_LUT"))
    testSuite.addTest(test_mask_beamstop("test_mask_LUT_OCL"))
    testSuite.addTest(test_mask_beamstop("test_nomask_LUT"))
    testSuite.addTest(test_mask_beamstop("test_nomask_LUT_OCL"))
    return testSuite

if __name__ == '__main__':

    mysuite = test_suite_all_Mask()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
