#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2020-2020 European Synchrotron Radiation Facility, Grenoble, France
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

"""Test suite for the decomposition of masks in a set of rectangles"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "10/11/2020"

import numpy
from ..detectors import detector_factory
from ..ext import dynamic_rectangle
import unittest
import logging
logger = logging.getLogger(__name__)


class TestRectangle(unittest.TestCase):

    def test_simple(self):
        mask = numpy.zeros((64, 64), dtype=numpy.int8)
        mask[11:17, 29:37] = 1
        ref_sum = mask.sum()
        rect = dynamic_rectangle.get_largest_rectangle(mask)
        self.assertEqual(rect.row, 11, "row is OK")
        self.assertEqual(rect.col, 29, "col is OK")
        self.assertEqual(rect.height, 17 - 11, "heigth is OK")
        self.assertEqual(rect.width, 37 - 29, "width is OK")
        self.assertEqual(rect.area, (37 - 29) * (17 - 11), "area is OK")
        self.assertEqual(mask.sum(), rect.area, "Mask is unchanged")

    def test_decomposition(self):
        mask = detector_factory("PilatusCdTe300k").mask
        ref = mask.sum()
        lst = dynamic_rectangle.decompose_mask(mask, False)
        self.assertEqual(mask.sum(), ref, "Mask is unchanged")
        self.assertEqual(len(lst), 5, "Decomposes in 5 rectangles")

        lst = dynamic_rectangle.decompose_mask(mask, True)
        self.assertEqual(mask.sum(), ref, "Mask is unchanged")
        self.assertEqual(len(lst), 3, "Decomposes in 3 overlapping bands")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestRectangle))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
