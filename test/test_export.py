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

def testExport(direct=100, centerX=900, centerY=1000, tilt=0, tpr=0, pixelX=50, pixelY=60):

    a1 = pyFAI.AzimuthalIntegrator()
    a2 = pyFAI.AzimuthalIntegrator()
    a3 = pyFAI.AzimuthalIntegrator()
    a1.setFit2D(direct, centerX, centerY, tilt, tpr, pixelX, pixelY)
#    print a1
    a2.setPyFAI(**a1.getPyFAI())
    a3.setFit2D(**a2.getFit2D())
    res = ""
    for e, o in [(a1, a2), (a1, a3), (a2, a3)]:
        for key in ["dist", "poni1", "poni2", "rot1", "rot2", "rot3", "pixel1", "pixel2", "splineFile"]:
            refv = e.__getattribute__(key)
            obtv = o.__getattribute__(key)
            try:
                if round(abs(float(refv) - float(obtv))) != 0:
                    res += "%s: %s != %s" % (key, refv, obtv)
            except TypeError as error:
                if refv != obtv:
                    res += "%s: %s != %s" % (key, refv, obtv)
    return res

class test_fit2d(unittest.TestCase):
    poniFile = "1893/Pilatus1M.poni"

    def setUp(self):
        """Download files"""
        self.poniFile = UtilsTest.getimage(self.__class__.poniFile)

    def test_simple(self):
        ref = pyFAI.load(self.poniFile)
        obt = pyFAI.AzimuthalIntegrator()
        obt.setFit2D(**ref.getFit2D())
        for key in ["dist", "poni1", "poni2", "rot1", "rot2", "rot3", "pixel1", "pixel2", "splineFile"]:
            refv = ref.__getattribute__(key)
            obtv = obt.__getattribute__(key)
            if refv is  None:
                self.assertEqual(refv, obtv , "%s: %s != %s" % (key, refv, obtv))
            else:
                self.assertAlmostEqual(refv, obtv , 4, "%s: %s != %s" % (key, refv, obtv))

    def test_export(self):
        res = testExport()
        self.assertFalse(res, res)
        res = testExport(tilt=20)
        self.assertFalse(res, res)
        res = testExport(tilt=20, tpr=80)
        self.assertFalse(res, res)
        res = testExport(tilt=20, tpr=580)
        self.assertFalse(res, res)

def test_suite_all_Export():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_fit2d("test_simple"))
    testSuite.addTest(test_fit2d("test_export"))
    return testSuite

if __name__ == '__main__':

    mysuite = test_suite_all_Export()
    runner = unittest.TextTestRunner()
    if not runner.run(mysuite).wasSuccessful():
        sys.exit(1)

