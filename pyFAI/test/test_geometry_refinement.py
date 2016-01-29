#!/usr/bin/python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) 2015 European Synchrotron Radiation Facility, Grenoble, France
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

__doc__ = "test suite for Geometric Refinement class"
__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "29/01/2016"

import unittest
import os
import numpy
import sys
import random

from .utilstest import UtilsTest, getLogger

logger = getLogger(__file__)
from .. import geometryRefinement
from .. import calibrant
GeometryRefinement = geometryRefinement.GeometryRefinement


class TestGeometryRefinement(unittest.TestCase):
    """ tests geometric refinements with or without spline"""

    def test_noSpline(self):
        """tests geometric refinements without spline"""

        pixelSize = [1.5e-5, 1.5e-5]
        data = [
[1585.9999996029055, 2893.9999991192408, 0.53005649383067788],
[1853.9999932086102, 2873.0000001637909, 0.53005649383067788],
[2163.9999987531855, 2854.9999987738884, 0.53005649383067788],
[2699.9999977914931, 2893.9999985831755, 0.53005649383067788],
[3186.9999966428777, 3028.9999985930604, 0.53005649383067788],
[3595.0000039534661, 3167.0000022967461, 0.53005649383067788],
[3835.0000007197755, 3300.0000002536408, 0.53005649383067788],
[1252.0000026881371, 2984.0000056421914, 0.53005649383067788],
[576.99992486352289, 3220.0000014469815, 0.53005649383067788],
[52.999989546760531, 3531.9999975314959, 0.53005649383067788],
[520.99999862452842, 2424.0000005943775, 0.65327673902147754],
[1108.0000045189499, 2239.9999793751085, 0.65327673902147754],
[2022.0000098770186, 2136.9999921020726, 0.65327673902147754],
[2436.000002384907, 2137.0000034435734, 0.65327673902147754],
[2797.9999973906524, 2169.9999849019205, 0.65327673902147754],
[3516.0000041508365, 2354.0000059814265, 0.65327673902147754],
[3870.9999995625412, 2464.9999964079757, 0.65327673902147754],
[3735.9999952703465, 2417.9999888223151, 0.65327673902147754],
[3374.0001428680412, 2289.9999885080188, 0.65327673902147754],
[1709.99999872134, 2165.0000006693272, 0.65327673902147754],
[2004.0000081015958, 1471.0000012076148, 0.7592182246175333],
[2213.0000015244159, 1464.0000243454842, 0.7592182246175333],
[2115.9999952456633, 1475.0000015176133, 0.7592182246175333],
[2242.0000023736206, 1477.0000046142911, 0.7592182246175333],
[2463.9999967564663, 1464.0000011704756, 0.7592182246175333],
[2986.000011249705, 1540.9999994523619, 0.7592182246175333],
[2760.0000031761901, 1514.0000002442944, 0.7592182246175333],
[3372.0000025298395, 1617.9999995345927, 0.7592182246175333],
[3187.0000005152106, 1564.9999952212884, 0.7592182246175333],
[3952.0000062252166, 1765.0000234029771, 0.7592182246175333],
[200.99999875941003, 1190.0000046393075, 0.85451320177642376],
[463.00000674257342, 1121.9999956648539, 0.85451320177642376],
[1455.0000001416358, 936.99999830341949, 0.85451320177642376],
[1673.9999958962637, 927.99999934328309, 0.85451320177642376],
[2492.0000021823594, 922.00000383122256, 0.85451320177642376],
[2639.9999948599761, 936.00000247819059, 0.85451320177642376],
[3476.9999490636446, 1027.9999838362451, 0.85451320177642376],
[3638.9999965727247, 1088.0000258143732, 0.85451320177642376],
[4002.0000051610787, 1149.9999925115812, 0.85451320177642376],
[2296.9999822277705, 908.00000939182382, 0.85451320177642376],
[266.00000015817864, 576.00000049157074, 0.94195419730133967],
[364.00001493127616, 564.00000136247968, 0.94195419730133967],
[752.99999958240187, 496.9999948653093, 0.94195419730133967],
[845.99999758606646, 479.00000730401808, 0.94195419730133967],
[1152.0000082161678, 421.9999937722655, 0.94195419730133967],
[1215.0000019951258, 431.00019867504369, 0.94195419730133967],
[1728.0000096657914, 368.00000247754218, 0.94195419730133967],
[2095.9999932673395, 365.99999862304219, 0.94195419730133967],
[2194.0000006543587, 356.99999967534075, 0.94195419730133967],
[2598.0000021676074, 386.99999979901884, 0.94195419730133967],
[2959.9998766657627, 410.00000323183838, 0.94195419730133967],
]
        data = numpy.array(data, dtype=numpy.float64)
#        tth = data[:,2]
        ring = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3,
                3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
                5, 5, 5, 5, 5]
        ds = [ 4.15695   , 2.93940753, 2.4000162 , 2.078475  , 1.85904456,
        1.69706773, 1.46970377, 1.38565   , 1.31454301, 1.25336758,
        1.2000081 , 1.15293049, 1.11099162, 1.0392375 , 1.00820847,
        0.97980251, 0.95366973, 0.92952228, 0.90712086, 0.88626472,
        0.84853387, 0.83139   , 0.81524497, 0.8000054 , 0.77192624,
        0.75895176, 0.73485188, 0.72363211, 0.71291104, 0.7026528 ,
        0.692825  , 0.68339837, 0.67434634, 0.65727151, 0.64920652,
        0.64143131, 0.63392893, 0.62668379, 0.61968152, 0.61290884,
        0.60000405, 0.59385   , 0.58788151, 0.58208943, 0.57646525,
        0.571001  , 0.56568924, 0.55549581, 0.55060148, 0.54583428,
        0.54118879, 0.53224291, 0.52793318, 0.52372647, 0.51961875,
        0.51560619, 0.51168517, 0.50785227, 0.50410423, 0.50043797,
        0.49685056]  # LaB6
        wavelength = 1.54e-10
        mycalibrant = calibrant.Calibrant(dSpacing=ds, wavelength=wavelength)
        data[:, 2] = ring

        r = GeometryRefinement(data, pixel1=pixelSize[0], pixel2=pixelSize[1],
                               wavelength=wavelength, calibrant=mycalibrant)
        r.refine2(10000000)

#        ref = numpy.array([0.089652, 0.030970, 0.027668, -0.699407, 0.010067, 0.000001])
        ref = numpy.array([0.089750, 0.030897, 0.027172, -0.704730, 0.010649, 3.51e-06])
        self.assertAlmostEqual(abs(numpy.array(r.param) - ref).max(), 0.0, 3, "ref=%s obt=%s delta=%s" % (list(ref), r.param, abs(numpy.array(r.param) - ref)))

    def test_Spline(self):
        """tests geometric refinements with spline"""
        splineFine = UtilsTest.getimage("1900/frelon.spline")
        data = [[795, 288, 0.3490658503988659],
                [890, 260, 0.3490658503988659],
                [948, 249, 0.3490658503988659],
                [710, 325, 0.3490658503988659],
                [601, 392, 0.3490658503988659],
                [1167, 248, 0.3490658503988659],
                [1200, 340, 0.3490658503988659],
                [1319, 285, 0.3490658503988659],
                [1362, 302, 0.3490658503988659],
                [1436, 338, 0.3490658503988659],
                [1526, 397, 0.3490658503988659],
                [1560, 424, 0.3490658503988659],
                [1615, 476, 0.3490658503988659],
                [1662, 529, 0.3490658503988659],
                [1742, 650, 0.3490658503988659],
                [1778, 727, 0.3490658503988659],
                [1824, 891, 0.3490658503988659],
                [1831, 947, 0.3490658503988659],
                [1832, 1063, 0.3490658503988659],
                [1828, 1106, 0.3490658503988659],
                [1828, 1106, 0.3490658503988659],
                [1810, 1202, 0.3490658503988659],
                [1775, 1307, 0.3490658503988659],
                [1724, 1407, 0.3490658503988659],
                [1655, 1502, 0.3490658503988659],
                [1489, 1649, 0.3490658503988659],
                [1397, 1700, 0.3490658503988659],
                [1251, 1752, 0.3490658503988659],
                [1126, 1772, 0.3490658503988659],
                [984, 1770, 0.3490658503988659],
                [907, 1758, 0.3490658503988659],
                [801, 1728, 0.3490658503988659],
                [696, 1681, 0.3490658503988659],
                [634, 1644, 0.3490658503988659],
                [568, 1596, 0.3490658503988659],
                [520, 1553, 0.3490658503988659],
                [453, 1479, 0.3490658503988659],
                [403, 1408, 0.3490658503988659],
                [403, 1408, 0.3490658503988659],
                [363, 1337, 0.3490658503988659],
                [320, 1228, 0.3490658503988659],
                [303, 1161, 0.3490658503988659],
                [287, 1023, 0.3490658503988659],
                [287, 993, 0.3490658503988659],
                [304, 846, 0.3490658503988659],
                [329, 758, 0.3490658503988659],
                [341, 726, 0.3490658503988659],
                [402, 606, 0.3490658503988659],
                [437, 555, 0.3490658503988659],
                [513, 467, 0.3490658503988659]
                ]
#        data = numpy.array(data)
        random.shuffle(data)
        tth = data[0][2]
        # data[:, 2] = ring
        wl = 2e-10 * numpy.sin(tth / 2.0)
        ds = [1.0]
        mycalibrant = calibrant.Calibrant(dSpacing=ds, wavelength=wl)
        r2 = GeometryRefinement(data, dist=0.1, splineFile=splineFine, wavelength=wl, calibrant=mycalibrant)
#        r2.poni1 = 5e-2
#        r2.poni2 = 5e-2
        r2.rot1_max = 0.0001
        r2.rot1_min = -0.0001
        r2.rot2_max = 0.0001
        r2.rot2_min = -0.0001
        r2.rot3_max = 0.0001
        r2.rot3_min = -0.0001
        r2.refine2(10000000)
        ref2 = numpy.array([0.1, 4.917310e-02, 4.722438e-02, 0, 0., 0.00000])
        for i, key in enumerate(("dist", "poni1", "poni2", "rot1", "rot2", "rot3")):
            self.assertAlmostEqual(ref2[i], r2.__getattribute__(key), 3,
                                   "%s is %s, I expected %s%s%s" % (key, r2.__getattribute__(key), ref2[i], os.linesep, r2))
#        assert abs(numpy.array(r2.param) - ref2).max() < 1e-3


def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(TestGeometryRefinement("test_noSpline"))
    testsuite.addTest(TestGeometryRefinement("test_Spline"))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

