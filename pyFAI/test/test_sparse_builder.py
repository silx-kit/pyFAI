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

"""Test suites for sparse builder module"""

from __future__ import absolute_import, division, print_function

__author__ = "Valentin Valls"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "11/07/2018"


import unittest
import numpy
import logging
logger = logging.getLogger(__name__)
from ..ext import sparse_builder


class TestSparseBuilder(unittest.TestCase):
    """Test for sparse builder
    """

    def test_insert(self):
        builder = sparse_builder.SparseBuilder(10, block_size=512)
        builder.insert(0, 0, 1.0)
        builder.insert(0, 1, 0.6)
        builder.insert(1, 1, 0.4)
        self.assertEqual(builder.size(), 3)

    def test_bin_size(self):
        builder = sparse_builder.SparseBuilder(10, block_size=512)
        builder.insert(0, 0, 1.0)
        builder.insert(0, 1, 0.6)
        builder.insert(1, 1, 0.4)
        self.assertEqual(builder.get_bin_size(0), 2)
        self.assertEqual(builder.get_bin_size(1), 1)
        self.assertEqual(builder.get_bin_size(2), 0)

    def test_bin_coefs(self):
        builder = sparse_builder.SparseBuilder(10, block_size=512)
        builder.insert(0, 0, 1.0)
        builder.insert(0, 1, 0.6)
        builder.insert(1, 1, 0.4)
        builder.insert(0, 2, 1.0)
        self.assertTrue(numpy.allclose(builder.get_bin_coefs(0), numpy.array([1.0, 0.6, 1.0])))
        self.assertTrue(numpy.allclose(builder.get_bin_coefs(1), numpy.array([0.4])))
        self.assertTrue(numpy.allclose(builder.get_bin_coefs(2), numpy.array([])))

    def test_bin_indexes(self):
        builder = sparse_builder.SparseBuilder(10, block_size=512)
        builder.insert(0, 0, 1.0)
        builder.insert(0, 1, 0.6)
        builder.insert(1, 1, 0.4)
        builder.insert(0, 2, 1.0)
        self.assertTrue(numpy.allclose(builder.get_bin_indexes(0), numpy.array([0, 1, 2])))
        self.assertTrue(numpy.allclose(builder.get_bin_indexes(1), numpy.array([1])))
        self.assertTrue(numpy.allclose(builder.get_bin_indexes(2), numpy.array([])))

    def test_block_overflow(self):
        builder = sparse_builder.SparseBuilder(10, block_size=16)
        for i in range(50):
            builder.insert(0, i, i * 0.1)
        self.assertEqual(builder.get_bin_size(0), 50)
        x = numpy.array(range(50))
        self.assertTrue(numpy.allclose(builder.get_bin_indexes(0), x))
        self.assertTrue(numpy.allclose(builder.get_bin_coefs(2), x * 0.1))

    def test_csr(self):
        builder = sparse_builder.SparseBuilder(10, block_size=512)
        builder.insert(0, 0, 1.0)
        builder.insert(0, 1, 0.6)
        builder.insert(1, 1, 0.4)
        a, b, c = builder.to_csr()
        self.assertTrue(numpy.allclose(a, numpy.array([1.0, 0.6, 0.4])))
        self.assertTrue(numpy.allclose(b, numpy.array([0, 1, 1])))
        self.assertTrue(numpy.allclose(c, numpy.array([0, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3])))

    def test_small_tth(self):
        shape = (512, 512)
        npt = 200
        y, x = numpy.ogrid[:shape[0], :shape[1]]
        tth = numpy.sqrt(x * x + y * y)
        maxI = 1000
        mod = 0.5 + 0.5 * numpy.cos(tth / 12) + 0.25 * numpy.cos(tth / 6) + 0.1 * numpy.cos(tth / 4)
        data = (numpy.ones(shape) * maxI * mod).astype("uint16")

        builder = sparse_builder.SparseBuilder(npt, block_size=512)
        sparse_builder.feed_histogram(builder, tth, data, npt)
        self.assertEqual(builder.size(), shape[0] * shape[1])


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestSparseBuilder))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
