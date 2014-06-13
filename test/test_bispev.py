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
import time
from utilstest import UtilsTest, getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]

if logger.getEffectiveLevel() <= logging.INFO:
    import matplotlib;matplotlib.use('Qt4Agg');import pylab
from pyFAI import spline, _bispev

try:
    from scipy.interpolate import fitpack
except:
    fitpack = None


class TestBispev(unittest.TestCase):
    spinefile = "1461/halfccd.spline"

    def setUp(self):
        """Download files"""
        self.splineFile = UtilsTest.getimage(self.__class__.spinefile)
        self.spline = spline.Spline(self.splineFile)
        self.spline.spline2array(timing=True)
    def test_bispev(self):
        print
        x_1d_array = numpy.arange(self.spline.xmin, self.spline.xmax + 1)
        y_1d_array = numpy.arange(self.spline.ymin, self.spline.ymax + 1)
        t0 = time.time()
        dx_ref = fitpack.bisplev(
                x_1d_array, y_1d_array, [self.spline.xSplineKnotsX,
                                         self.spline.xSplineKnotsY,
                                         self.spline.xSplineCoeff,
                                         self.spline.splineOrder,
                                         self.spline.splineOrder],
                dx=0, dy=0)
        t1 = time.time()
        print self.spline.xSplineKnotsX.dtype
        print self.spline.xSplineKnotsY.dtype
        print self.spline.xSplineCoeff.dtype
        dx_loc = _bispev.bisplev(
                x_1d_array, y_1d_array, [self.spline.xSplineKnotsX,
                                         self.spline.xSplineKnotsY,
                                         self.spline.xSplineCoeff,
                                         self.spline.splineOrder,
                                         self.spline.splineOrder],
                )
        t2 = time.time()
        print("Scipy timings: %.3fs\t cython timings: %.3fs" % (t1 - t0, t2 - t1))
        print(dx_ref.shape, dx_loc.shape)
        print(dx_ref)
        print(dx_loc)
        print(abs(dx_loc - dx_ref).max())
        if logger.getEffectiveLevel() == logging.DEBUG:
            fig = pylab.figure()
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax1.imshow(dx_ref)
            ax2.imshow(dx_loc)
            fig.show()
            raw_input()
        self.assert_(abs(dx_loc - dx_ref).max() < 2e-5, "Result are similar")


def test_suite_all_bispev():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestBispev("test_bispev"))
    return testSuite

if __name__ == '__main__':

    mysuite = test_suite_all_bispev()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
