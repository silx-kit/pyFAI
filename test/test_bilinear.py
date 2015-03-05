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

"test suite for bilinear interpolator class"

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "05/03/2015"


import unittest
import os
import numpy
# import logging  # , time
import sys
# import fabio
if __name__ == '__main__':
    import pkgutil
    __path__ = pkgutil.extend_path([os.path.dirname(__file__)], "pyFAI.test")
from .utilstest import getLogger, UtilsTest
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]
from pyFAI import bilinear
# from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
# if logger.getEffectiveLevel() <= logging.INFO:
#    import pylab
# from pyFAI import bilinear
# bilinear = sys.modules["pyFAI.bilinear"]


class TestBilinear(unittest.TestCase):
    """basic maximum search test"""
    N = 10000

    def test_max_search_round(self):
        """test maximum search using random points: maximum is at the pixel center"""
        a = numpy.arange(100) - 40.
        b = numpy.arange(100) - 60.
        ga = numpy.exp(-a * a / 4000)
        gb = numpy.exp(-b * b / 6000)
        gg = numpy.outer(ga, gb)
        b = bilinear.Bilinear(gg)
        ok = 0
        for s in range(self.N):
            i, j = numpy.random.randint(100), numpy.random.randint(100)
            k, l = b.local_maxi((i, j), 1)
            if abs(k - 40) > 1e-4 or abs(l - 60) > 1e-4:
                logger.warning("Wrong guess maximum (%i,%i) -> (%.1f,%.1f)" % (i, j, k, l))
            else:
                logger.debug("Good guess maximum (%i,%i) -> (%.1f,%.1f)" % (i, j, k, l))
                ok += 1
        logger.info("Success rate: %.1f" % (100.*ok / self.N))
        self.assertEqual(ok, self.N, "Maximum is always found")

    def test_max_search_half(self):
        """test maximum search using random points: maximum is at a pixel edge"""
        a = numpy.arange(100) - 40.5
        b = numpy.arange(100) - 60.5
        ga = numpy.exp(-a * a / 4000)
        gb = numpy.exp(-b * b / 6000)
        gg = numpy.outer(ga, gb)
        b = bilinear.Bilinear(gg)
        ok = 0
        for s in range(self.N):
            i, j = numpy.random.randint(100), numpy.random.randint(100)
            k, l = b.local_maxi((i, j), 1)
            if abs(k - 40.5) > 0.5 or abs(l - 60.5) > 0.5:
                logger.warning("Wrong guess maximum (%i,%i) -> (%.1f,%.1f)" % (i, j, k, l))
            else:
                logger.debug("Good guess maximum (%i,%i) -> (%.1f,%.1f)" % (i, j, k, l))
                ok += 1
        logger.info("Success rate: %.1f" % (100. * ok / self.N))
        self.assertEqual(ok, self.N, "Maximum is always found")


class TestConversion(unittest.TestCase):
    """basic 2d -> 4d transformation and vice-versa"""
    def test4d(self):
        Nx = 1000
        Ny = 1024
        y, x = numpy.mgrid[:Ny + 1, :Nx + 1]
        y = y.astype(float)
        x = x.astype(float)
        pos = bilinear.convert_corner_2D_to_4D(3, y, x)
        y1, x1 = bilinear.calc_cartesian_positions(y.ravel(), x.ravel(), pos)
        self.assert_(numpy.allclose(y, y1), "Maximum error on y is %s" % (abs(y - y1).max()))
        self.assert_(numpy.allclose(x, x1), "Maximum error on x is %s" % (abs(x - x1).max()))
        x = x[:-1, :-1] + 0.5
        y = y[:-1, :-1] + 0.5
        y1, x1 = bilinear.calc_cartesian_positions((y).ravel(), (x).ravel(), pos)

        self.assert_(numpy.allclose(y, y1), "Maximum error on y_center is %s" % (abs(y - y1).max()))
        self.assert_(numpy.allclose(x, x1), "Maximum error on x_center is %s" % (abs(x - x1).max()))


def test_suite_all_bilinear():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestBilinear("test_max_search_round"))
    testSuite.addTest(TestBilinear("test_max_search_half"))
    testSuite.addTest(TestConversion("test4d"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_all_bilinear()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
    UtilsTest.clean_up()
