#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
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

"test suite for preprocessing corrections"

from __future__ import absolute_import, division, print_function

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "28/11/2016"


import unittest
import numpy
import logging

logger = logging.getLogger(__file__)

from ..ext import preproc


class TestPreproc(unittest.TestCase):
    def test(self):
        """
        The final pattern should look like a 4x4 square with 1 and -1 elsewhere.
        """
        shape = 8, 8
        size = shape[0] * shape[1]
        target = numpy.ones(shape)
        target[:2, :] = -1
        target[-2:, :] = -1
        target[:, -2:] = -1
        target[:, :2] = -1
        mask = numpy.zeros(shape, "int8")
        mask[:2, :] = 1
        dark = numpy.random.poisson(10, size).reshape(shape)
        flat = 1.0 + numpy.random.random(shape)
        scale = 10
        raw = scale * flat + dark
        raw[-2:, :] = numpy.NaN
        dummy = -1
        raw[:, :2] = dummy
        flat[:, -2:] = dummy

        # add some tests with various levels of conditionning
        res = preproc.preproc(raw)
        # then Nan on last lines -> 0
        self.assertEqual(abs(res[-2:, 2:]).max(), 0, "Nan filtering")

        res = preproc.preproc(raw, empty=-1)
        # then Nan on last lines -> -1
        self.assertEqual(abs(res[-2:, :] + 1).max(), 0, "Nan filtering")

        res = preproc.preproc(raw, dummy=-1, delta_dummy=0.5)
        # test dummy
        self.assertEqual(abs(res[-2:, :] + 1).max(), 0, "dummy")

        # test polarization, solidangle and sensor thickness  with dummy.
        res = preproc.preproc(raw, dark, polarization=flat, dummy=dummy, mask=mask, normalization_factor=scale)
        self.assertEqual(abs(numpy.round(res[2:-2, 2:-2]) - 1).max(), 0, "mask is properly applied")
        self.assertGreater(abs(numpy.round(res) - target).max(), 0, "flat != polarization")

        res = preproc.preproc(raw, dark, solidangle=flat, dummy=dummy, mask=mask, normalization_factor=scale)
        self.assertEqual(abs(numpy.round(res[2:-2, 2:-2]) - 1).max(), 0, "mask is properly applied")
        self.assertGreater(abs(numpy.round(res) - target).max(), 0, "flat != solidangle")

        res = preproc.preproc(raw, dark, absorption=flat, dummy=dummy, mask=mask, normalization_factor=scale)
        self.assertEqual(abs(numpy.round(res[2:-2, 2:-2]) - 1).max(), 0, "mask is properly applied")
        self.assertGreater(abs(numpy.round(res) - target).max(), 0, "flat != absorption")

        # Test all features together
        res = preproc.preproc(raw, dark, flat, dummy=dummy, mask=mask, normalization_factor=scale)
        self.assertEqual(abs(numpy.round(res) - target).max(), 0, "mask is properly applied")


def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(TestPreproc("test"))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
