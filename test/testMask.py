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
import os
import numpy
import logging, time
import sys
import fabio
from utilstest import UtilsTest, Rwp, getLogger
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
        self.data = fabio.open(self.dataFile).data
        self.mask = self.data < 0

    def test_mask_splitBBox(self):
        """
        The masked image has a masked ring around 1.5deg with value -10
        without mask the pixels should be at -10 ; with mask they are at 0 
        """
        x1 = self.ai.xrpd_splitBBox(self.data, 1000)
        x2 = self.ai.xrpd_splitBBox(self.data, 1000, mask=self.mask)
        x3 = self.ai.xrpd_splitBBox(self.data, 1000, dummy= -20.0, delta_dummy=19.5)
        res1 = numpy.interp(1.5, *x1)
        res2 = numpy.interp(1.5, *x2)
        res3 = numpy.interp(1.5, *x3)
        if logger.getEffectiveLevel() == logging.DEBUG:
            pylab.plot(*x1)
            pylab.plot(*x2)
            pylab.plot(*x3)
            pylab.show()
            raw_input()

        self.assertAlmostEqual(res1, -10., 2, msg="Without mask the bad pixels are actually at -10 (got %.4f)" % res1)
        self.assertAlmostEqual(res2, 0., 4, msg="With mask the bad pixels are actually at 0 (got %.4f)" % res2)
        self.assertAlmostEqual(res3, -20., 4, msg="With mask the dummy pixels are actually at 0 (got % .4f)" % res3)



def test_suite_all_Mask():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_mask("test_mask_splitBBox"))
#    testSuite.addTest(test_mask("test_mask_OpenCL"))
#    testSuite.addTest(test_mask("test_mask_splitBBox"))
#    testSuite.addTest(test_mask("test_mask_splitBBox"))
#    testSuite.addTest(test_mask("test_mask_splitBBox"))
#    testSuite.addTest(test_mask("test_mask_splitBBox"))
    return testSuite

if __name__ == '__main__':

    mysuite = test_suite_all_Mask()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
