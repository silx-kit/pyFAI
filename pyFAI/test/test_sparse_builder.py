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
__date__ = "16/07/2018"


import unittest
import numpy
import logging
logger = logging.getLogger(__name__)
from ..ext import sparse_builder
from . import utilstest
import collections


class TestSparseBuilder(utilstest.ParametricTestCase):
    """Test for sparse builder
    """

    def each_builders(self, nbin):
        builders = collections.OrderedDict()

        builders["block"] = sparse_builder.SparseBuilder(nbin, mode="block", block_size=512)
        builders["block_small"] = sparse_builder.SparseBuilder(nbin, mode="block", block_size=2)
        builders["block_heap"] = sparse_builder.SparseBuilder(nbin, mode="block", block_size=2, heap_size=128)
        builders["block_small_heap"] = sparse_builder.SparseBuilder(nbin, mode="block", block_size=1, heap_size=3)
        builders["heaplist"] = sparse_builder.SparseBuilder(nbin, mode="heaplist", heap_size=512)
        builders["heaplist_small_heap"] = sparse_builder.SparseBuilder(nbin, mode="heaplist", heap_size=3)
        builders["stdlist"] = sparse_builder.SparseBuilder(nbin, mode="stdlist")
        builders["pack"] = sparse_builder.SparseBuilder(nbin, mode="pack", heap_size=512)
        builders["pack_small_heap"] = sparse_builder.SparseBuilder(nbin, mode="pack", heap_size=3)

        for builder_name, builder in builders.items():
            yield builder_name, builder

    def subtest_each_builders(self, nbin):
        for builder_name, builder in self.each_builders(nbin):
            with self.subTest(builder=builder_name):
                yield builder

    def test_insert(self):
        for builder in self.subtest_each_builders(10):
            builder.insert(0, 0, 1.0)
            builder.insert(0, 1, 0.6)
            builder.insert(1, 1, 0.4)
            self.assertEqual(builder.size(), 3)

    def test_bin_sizes(self):
        for builder in self.subtest_each_builders(5):
            builder.insert(0, 0, 1.0)
            builder.insert(0, 1, 0.6)
            builder.insert(1, 1, 0.4)
            builder.insert(4, 2, 1.0)
            builder.insert(4, 3, 1.0)
            builder.insert(4, 4, 1.0)
            builder.insert(4, 5, 1.0)
            builder.insert(4, 6, 1.0)
            self.assertTrue(numpy.allclose(builder.get_bin_sizes(), numpy.array([2, 1, 0, 0, 5])))

    def test_bin_size(self):
        for builder in self.subtest_each_builders(10):
            builder.insert(0, 0, 1.0)
            builder.insert(0, 1, 0.6)
            builder.insert(1, 1, 0.4)
            self.assertEqual(builder.get_bin_size(0), 2)
            self.assertEqual(builder.get_bin_size(1), 1)
            self.assertEqual(builder.get_bin_size(2), 0)

    def test_bin_coefs(self):
        for builder in self.subtest_each_builders(10):
            if builder.mode() == "pack":
                continue
            builder.insert(0, 0, 1.0)
            builder.insert(0, 1, 0.6)
            builder.insert(1, 1, 0.4)
            builder.insert(0, 2, 1.0)
            self.assertTrue(numpy.allclose(builder.get_bin_coefs(0), numpy.array([1.0, 0.6, 1.0])))
            self.assertTrue(numpy.allclose(builder.get_bin_coefs(1), numpy.array([0.4])))
            self.assertTrue(numpy.allclose(builder.get_bin_coefs(2), numpy.array([])))

    def test_bin_indexes(self):
        for builder in self.subtest_each_builders(10):
            if builder.mode() == "pack":
                continue
            builder.insert(0, 0, 1.0)
            builder.insert(0, 1, 0.6)
            builder.insert(1, 1, 0.4)
            builder.insert(0, 2, 1.0)
            self.assertTrue(numpy.allclose(builder.get_bin_indexes(0), numpy.array([0, 1, 2])))
            self.assertTrue(numpy.allclose(builder.get_bin_indexes(1), numpy.array([1])))
            self.assertTrue(numpy.allclose(builder.get_bin_indexes(2), numpy.array([])))

    def test_insert_overflow(self):
        for builder in self.subtest_each_builders(10):
            for i in range(50):
                builder.insert(0, i, i * 0.1)
            self.assertEqual(builder.get_bin_size(0), 50)
            x = numpy.array(range(50))
            coefs, indexes, _bin_indexes = builder.to_csr()
            self.assertTrue(numpy.allclose(indexes, x))
            self.assertTrue(numpy.allclose(coefs, x * 0.1))

    def test_csr(self):
        for builder in self.subtest_each_builders(10):
            builder.insert(0, 0, 1.0)
            builder.insert(0, 1, 0.6)
            builder.insert(1, 1, 0.4)
            a, b, c = builder.to_csr()
            self.assertTrue(numpy.allclose(a, numpy.array([1.0, 0.6, 0.4])))
            self.assertTrue(numpy.allclose(b, numpy.array([0, 1, 1])))
            self.assertTrue(numpy.allclose(c, numpy.array([0, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3])))

    def test_lut(self):
        expected_idx = numpy.array([[0, 1],
                                    [1, 0],
                                    [0, 0],
                                    [2, 3],
                                    [0, 0]])
        expected_coef = numpy.array([[1.0, 0.6],
                                     [0.4, 0.0],
                                     [0.0, 0.0],
                                     [0.8, 0.2],
                                     [0.0, 0.0]])

        for builder in self.subtest_each_builders(5):
            builder.insert(0, 0, 1.0)
            builder.insert(0, 1, 0.6)
            builder.insert(1, 1, 0.4)
            builder.insert(3, 2, 0.8)
            builder.insert(3, 3, 0.2)
            lut = builder.to_lut()
            self.assertTrue(numpy.allclose(lut["idx"], expected_idx))
            self.assertTrue(numpy.allclose(lut["coef"], expected_coef))

    def test_small_tth(self):
        shape = (256, 256)
        npt = 200
        y, x = numpy.ogrid[:shape[0], :shape[1]]
        tth = numpy.sqrt(x * x + y * y)
        maxI = 1000
        mod = 0.5 + 0.5 * numpy.cos(tth / 12) + 0.25 * numpy.cos(tth / 6) + 0.1 * numpy.cos(tth / 4)
        data = (numpy.ones(shape) * maxI * mod).astype("uint16")
        previous_coefs, previous_indexes, previous_bin_indexes = None, None, None

        for builder in self.subtest_each_builders(npt):
            sparse_builder.feed_histogram(builder, tth, data, npt)
            self.assertEqual(builder.size(), shape[0] * shape[1])

            # Check consistancy of the results
            coefs, indexes, bin_indexes = builder.to_csr()
            if previous_coefs is not None:
                self.assertTrue(numpy.allclose(coefs, previous_coefs))
            if previous_indexes is not None:
                self.assertTrue(numpy.allclose(indexes, previous_indexes))
            if previous_bin_indexes is not None:
                self.assertTrue(numpy.allclose(bin_indexes, previous_bin_indexes))
            previous_coefs, previous_indexes, previous_bin_indexes = coefs, indexes, bin_indexes


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestSparseBuilder))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
