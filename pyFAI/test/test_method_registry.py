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

"""Test suite for masked arrays"""

__author__ = "Valentin Valls"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "09/05/2019"

import unittest
from silx.utils.testutils import ParametricTestCase
from ..method_registry import Method


class TestMethod(ParametricTestCase):

    def test_parsed(self):
        samples = [
            ("numpy", Method(dim=None, split='*', algo='*', impl='python', target=None)),
            ("cython", Method(dim=None, split='*', algo='*', impl='cython', target=None)),
            ("bbox", Method(dim=None, split='bbox', algo='*', impl='*', target=None)),
            # ("splitpixel", Method(dim=None, split='bbox', algo='*', impl='*', target=None))),
            ("lut", Method(dim=None, split='*', algo='lut', impl='*', target=None)),
            ("csr", Method(dim=None, split='*', algo='csr', impl='*', target=None)),
            ("nosplit_csr", Method(dim=None, split='no', algo='csr', impl='*', target=None)),
            ("full_csr", Method(dim=None, split='full', algo='csr', impl='*', target=None)),
            ("lut_ocl", Method(dim=None, split='*', algo='lut', impl='opencl', target=None)),
            ("csr_ocl", Method(dim=None, split='*', algo='csr', impl='opencl', target=None)),
            ("csr_ocl_1,5", Method(dim=None, split='*', algo='csr', impl='opencl', target=(1, 5))),
            ("ocl_2,3", Method(dim=None, split='*', algo='*', impl='opencl', target=(2, 3))),
        ]
        for string, expected in samples:
            with self.subTest(string=string):
                method = Method.parsed(string)
                self.assertEqual(method, expected)

    def test_fixed(self):
        value = Method(dim=None, split='*', algo='*', impl='python', target=None)
        expected = Method(dim=None, split=None, algo='*', impl='foo', target=None)
        result = value.fixed(split=None, impl="foo")
        self.assertEqual(result, expected)


def suite():
    testsuite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite.addTest(loader(TestMethod))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
