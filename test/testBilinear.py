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
"test suite for Azimuthal integrator class"

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "31/01/2012"


import unittest
# import os
import numpy
# import logging  # , time
import sys
# import fabio

from utilstest import getLogger  # UtilsTest, Rwp, getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]
from pyFAI import bilinear
# from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
# if logger.getEffectiveLevel() <= logging.INFO:
#    import pylab
#from pyFAI import bilinear
#bilinear = sys.modules["pyFAI.bilinear"]

class test_bilinear(unittest.TestCase):
    """basic maximum search test"""
    a = numpy.arange(100) - 40.
    b = numpy.arange(100) - 60.
    ga = numpy.exp(-a * a / 4000)
    gb = numpy.exp(-b * b / 6000)
    gg = numpy.outer(ga, gb)
    b = bilinear.Bilinear(gg)
    N = 10000
    def test_max_search(self):
        """test maximum search using random points"""
        ok = 0
        for s in range(self.N):
            i, j = numpy.random.randint(100), numpy.random.randint(100)
            k, l = self.b.local_maxi((i, j), 1)
            if abs(k - 40) > 1e-4 or abs(l - 60) > 1e-4:
                logger.warning("Wrong guess maximum (%i,%i) -> (%.1f,%.1f)" % (i, j, k, l))
            else:
                logger.debug("Good guess maximum (%i,%i) -> (%.1f,%.1f)" % (i, j, k, l))
                ok += 1
        logger.info("Success rate: %.1f" % (100.*ok / self.N))
        self.assertEqual(ok, self.N, "Maximum is always found")


def test_suite_all_bilinear():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_bilinear("test_max_search"))
#    testSuite.addTest(test_azim_halfFrelon("test_numpy_vs_fit2d"))
#    testSuite.addTest(test_azim_halfFrelon("test_cythonSP_vs_fit2d"))
#    testSuite.addTest(test_azim_halfFrelon("test_cython_vs_numpy"))
#    testSuite.addTest(test_flatimage("test_splitPixel"))
#    testSuite.addTest(test_flatimage("test_splitBBox"))
# This test is known to be broken ...
#    testSuite.addTest(test_saxs("test_mask"))

    return testSuite

if __name__ == '__main__':

    mysuite = test_suite_all_bilinear()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
