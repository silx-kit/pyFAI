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

"""Test suite for the image segmentation "massif" """

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "22/03/2019"


import unittest
import numpy
import logging

logger = logging.getLogger(__name__)

from ..massif import Massif


class TestMassif(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestMassif, cls).setUpClass()
        cls.shape = (100, 100)
        cls.image = numpy.random.poisson(10, cls.shape)
        cls.mask = numpy.zeros(cls.shape, dtype=numpy.int8)
        cls.mask[48:52, :] = 1
        cls.mask[:, 48:52] = 1

    @classmethod
    def tearDownClass(cls):
        super(TestMassif, cls).tearDownClass()
        cls.shape = cls.image = cls.mask = None

    def test_nomask(self):
        massif = Massif(self.image)
        massif.get_labeled_massif()
        print(massif._number_massif)

    def test_mask_noreconstruct(self):
        massif = Massif(self.image, self.mask)
        massif.get_labeled_massif(reconstruct=False)
        print(massif._number_massif)

    def test_mask_reconstruct(self):
        massif = Massif(self.image, self.mask)
        massif.get_labeled_massif(reconstruct=True)
        print(massif._number_massif)

    def test_himask(self):
        image = self.image[...]
        image[self.mask != 0] = 65000
        massif = Massif(image, self.mask)
        massif.get_labeled_massif(reconstruct=False)

    def test_lomask(self):
        image = self.image[...]
        image[self.mask != 0] = -100
        massif = Massif(image, self.mask)
        massif.get_labeled_massif(reconstruct=False)
        # let's increase slightly the test coverage
        mask = numpy.zeros_like(self.mask)
        mask[5:45, 5:45] = 1
        peaks = massif.peaks_from_area(mask)
        if peaks:
            massif.find_peaks([int(i) for i in peaks[0]], stdout=None)
        massif.get_median_data()


def suite():
    testsuite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite.addTest(loader(TestMassif))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
