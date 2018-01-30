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
__date__ = "10/01/2018"


import unittest
import logging
from .utilstest import UtilsTest
from ..azimuthalIntegrator import AzimuthalIntegrator

logger = logging.getLogger(__name__)


def testExport(direct=100, centerX=900, centerY=1000, tilt=0, tpr=0, pixelX=50, pixelY=60):

    a1 = AzimuthalIntegrator()
    a2 = AzimuthalIntegrator()
    a3 = AzimuthalIntegrator()
    a1.setFit2D(direct, centerX, centerY, tilt, tpr, pixelX, pixelY)
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
            except TypeError:
                if refv != obtv:
                    res += "%s: %s != %s" % (key, refv, obtv)
    return res


class TestFIT2D(unittest.TestCase):
    poniFile = "Pilatus1M.poni"

    def setUp(self):
        """Download files"""
        self.poniFile = UtilsTest.getimage(self.__class__.poniFile)

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.poniFile = None

    def test_simple(self):
        ref = AzimuthalIntegrator.sload(self.poniFile)
        obt = AzimuthalIntegrator()
        obt.setFit2D(**ref.getFit2D())
        for key in ["dist", "poni1", "poni2", "rot1", "rot2", "rot3", "pixel1", "pixel2", "splineFile"]:
            refv = ref.__getattribute__(key)
            obtv = obt.__getattribute__(key)
            if refv is None:
                self.assertEqual(refv, obtv, "%s: %s != %s" % (key, refv, obtv))
            else:
                self.assertAlmostEqual(refv, obtv, 4, "%s: %s != %s" % (key, refv, obtv))

    def test_export(self):
        res = testExport()
        self.assertFalse(res, res)
        res = testExport(tilt=20)
        self.assertFalse(res, res)
        res = testExport(tilt=20, tpr=80)
        self.assertFalse(res, res)
        res = testExport(tilt=20, tpr=580)
        self.assertFalse(res, res)


class TestSPD(unittest.TestCase):
    poniFile = "Pilatus1M.poni"

    def setUp(self):
        """Download files"""
        self.poniFile = UtilsTest.getimage(self.__class__.poniFile)

    def test_simple(self):
        ref = AzimuthalIntegrator.sload(self.poniFile)
        obt = AzimuthalIntegrator()
        obt.setSPD(**ref.getSPD())
        for key in ["dist", "poni1", "poni2", "rot3", "pixel1", "pixel2", "splineFile"]:
            refv = ref.__getattribute__(key)
            obtv = obt.__getattribute__(key)
            if refv is None:
                self.assertEqual(refv, obtv, "%s: %s != %s" % (key, refv, obtv))
            else:
                self.assertAlmostEqual(refv, obtv, 4, "%s: %s != %s" % (key, refv, obtv))


def suite():
    testsuite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite.addTest(loader(TestFIT2D))
    testsuite.addTest(loader(TestSPD))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
