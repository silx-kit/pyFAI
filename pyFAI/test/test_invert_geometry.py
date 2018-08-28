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
"test suite for masked arrays"

from __future__ import absolute_import, division, print_function

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "27/08/2018"

import unittest
import numpy
from ..azimuthalIntegrator import AzimuthalIntegrator
from ..ext.invert_geometry import InvertGeometry


class TestInvertGeometry(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        detector = "pilatus 100k"
        ai = AzimuthalIntegrator(1, detector=detector, wavelength=1e-10)
        self.r = ai.array_from_unit(typ="center", unit="r_mm", scale=True)
        self.chi = ai.chiArray()

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.r = self.chi = None

    def test_invert_gerometry(self):
        ig = InvertGeometry(self.r, self.chi)
        t0 = 101
        t1 = 99
        self.assertEqual(ig(self.r[t0, t1], self.chi[t0, t1], False), (t0, t1), "without precision")
        self.assertEqual(ig(self.r[t0, t1], self.chi[t0, t1], True), (t0, t1), "with precision")

        eps = 0.1
        r = (1 - eps) * (self.r[t0, t1]) + eps * (self.r[t0 + 1, t1])
        chi = (1 - eps) * (self.chi[t0, t1]) + eps * (self.chi[t0 + 1, t1])

        self.assertEqual(ig(r, chi, False), (t0, t1), "without precision")
        self.assertLess(abs(numpy.array(ig(r, chi, True)) - [t0 + eps, t1]).max(),
                        1e-3, "Approximate with precision")

        eps = 0.2
        r = (1 - eps) * (self.r[t0, t1]) + eps * (self.r[t0, t1 + 1])
        chi = (1 - eps) * (self.chi[t0, t1]) + eps * (self.chi[t0, t1 + 1])

        self.assertEqual(ig(r, chi, False), (t0, t1), "without precision")
        self.assertLess(abs(numpy.array(ig(r, chi, True)) - [t0, t1 + eps]).max(),
                        1e-3, "Approximate with precision")


def suite():
    testsuite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite.addTest(loader(TestInvertGeometry))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
