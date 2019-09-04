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


from __future__ import division, print_function, absolute_import

"""Test suite for math utilities library"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "17/05/2019"

import unittest
import numpy
from pyFAI.utils import ellipse as ellipse_mdl


def modulo(value, div=numpy.pi):
    """hack to calculate the value%div but returns the smallest 
    absolute value, possibly negative"""
    q = value / div
    i = round(q)
    return (i - q) * div


class TestEllipse(unittest.TestCase):

    def test_ellipse(self):
        angles = numpy.arange(0, numpy.pi * 2, 0.2)
        pty = numpy.sin(angles) * 20 + 50
        ptx = numpy.cos(angles) * 10 + 100
        ellipse = ellipse_mdl.fit_ellipse(pty, ptx)
        self.assertAlmostEqual(ellipse.center_1, 50)
        self.assertAlmostEqual(ellipse.center_2, 100)
        self.assertAlmostEqual(ellipse.half_long_axis, 20)
        self.assertAlmostEqual(ellipse.half_short_axis, 10)
        self.assertAlmostEqual(modulo(ellipse.angle), 0)

    def test_ellipse2(self):
        angles = numpy.arange(0, numpy.pi * 2, 0.2)
        pty = numpy.sin(angles) * 10 + 50
        ptx = numpy.cos(angles) * 20 + 100
        ellipse = ellipse_mdl.fit_ellipse(pty, ptx)
        self.assertAlmostEqual(ellipse.center_1, 50)
        self.assertAlmostEqual(ellipse.center_2, 100)
        self.assertAlmostEqual(ellipse.half_long_axis, 20)
        self.assertAlmostEqual(ellipse.half_short_axis, 10)
        self.assertAlmostEqual(modulo(ellipse.angle), 0)

    def test_half_circle(self):
        angles = numpy.linspace(0, numpy.pi, 10)
        pty = numpy.sin(angles) * 20 + 10
        ptx = numpy.cos(angles) * 20 + 10
        ellipse = ellipse_mdl.fit_ellipse(pty, ptx)
        self.assertAlmostEqual(ellipse.center_1, 10)
        self.assertAlmostEqual(ellipse.center_2, 10)
        self.assertAlmostEqual(ellipse.half_long_axis, 20)
        self.assertAlmostEqual(ellipse.half_short_axis, 20)

    def test_quarter_circle(self):
        angles = numpy.linspace(0, numpy.pi / 2, 10)
        pty = numpy.sin(angles) * 20 + 10
        ptx = numpy.cos(angles) * 20 + 10
        ellipse = ellipse_mdl.fit_ellipse(pty, ptx)
        self.assertAlmostEqual(ellipse.center_1, 10)
        self.assertAlmostEqual(ellipse.center_2, 10)
        self.assertAlmostEqual(ellipse.half_long_axis, 20)
        self.assertAlmostEqual(ellipse.half_short_axis, 20)

    def test_halfquater_circle_5ptx(self):
        angles = numpy.linspace(0, numpy.pi / 4, 5)
        pty = numpy.sin(angles) * 20 + 10
        ptx = numpy.cos(angles) * 20 + 10
        ellipse = ellipse_mdl.fit_ellipse(pty, ptx)
        self.assertAlmostEqual(ellipse.center_1, 10, places=5)
        self.assertAlmostEqual(ellipse.center_2, 10, places=5)
        self.assertAlmostEqual(ellipse.half_long_axis, 20, places=5)
        self.assertAlmostEqual(ellipse.half_short_axis, 20, places=5)

    def test_centered_to_zero(self):
        angles = numpy.linspace(0, numpy.pi, 10)
        pty = numpy.sin(angles) * 20
        ptx = numpy.cos(angles) * 20
        ellipse = ellipse_mdl.fit_ellipse(pty, ptx)
        self.assertAlmostEqual(ellipse.center_1, 0, places=5)
        self.assertAlmostEqual(ellipse.center_2, 0, places=5)
        self.assertAlmostEqual(ellipse.half_long_axis, 20, places=5)
        self.assertAlmostEqual(ellipse.half_short_axis, 20, places=5)

    def test_line(self):
        pty = numpy.arange(10)
        ptx = numpy.arange(10)
        with self.assertRaises(ValueError):
            ellipse_mdl.fit_ellipse(pty, ptx)

    def test_real_data(self):
        # From real peaking
        pty = numpy.array([0.06599215, 0.06105629, 0.06963708, 0.06900191, 0.06496001, 0.06352082, 0.05923421, 0.07080027, 0.07276284, 0.07170048])
        ptx = numpy.array([0.05836343, 0.05866434, 0.05883284, 0.05872581, 0.05823667, 0.05839846, 0.0591999, 0.05907079, 0.05945377, 0.05909428])
        _ = ellipse_mdl.fit_ellipse(pty, ptx)


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestEllipse))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
