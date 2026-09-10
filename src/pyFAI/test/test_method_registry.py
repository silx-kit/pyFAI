#!/usr/bin/env python
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

"""Test suite for masked arrays"""

__author__ = "Valentin Valls"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "30/10/2024"

import unittest

from silx.utils.testutils import ParametricTestCase

from ..method_registry import IntegrationMethod, Method


class TestMethod(ParametricTestCase):

    def test_parsed(self):
        samples = [
            ("numpy", Method(dim=None, split=None, algo=None, impl='python', target=None)),
            ("cython", Method(dim=None, split=None, algo=None, impl='cython', target=None)),
            ("bbox", Method(dim=None, split='bbox', algo=None, impl=None, target=None)),
            # ("splitpixel", Method(dim=None, split='bbox', algo=None, impl=None, target=None))),
            ("lut", Method(dim=None, split=None, algo='lut', impl=None, target=None)),
            ("csr", Method(dim=None, split=None, algo='csr', impl=None, target=None)),
            ("nosplit_csr", Method(dim=None, split='no', algo='csr', impl=None, target=None)),
            ("full_csr", Method(dim=None, split='full', algo='csr', impl=None, target=None)),
            ("lut_ocl", Method(dim=None, split=None, algo='lut', impl='opencl', target=None)),
            ("csr_ocl", Method(dim=None, split=None, algo='csr', impl='opencl', target=None)),
            ("csr_ocl_1,5", Method(dim=None, split=None, algo='csr', impl='opencl', target=(1, 5))),
            ("ocl_2,3", Method(dim=None, split=None, algo=None, impl='opencl', target=(2, 3))),
        ]
        for string, expected in samples:
            with self.subTest(string=string):
                method = Method.parsed(string)
                self.assertEqual(method, expected)

    def test_fixed(self):
        value = Method(dim=None, split='*', algo='*', impl='python', target=None)
        expected = Method(dim=None, split=None, algo=None, impl='cython', target=None)
        result = value.fixed(split=None, impl="cython")
        self.assertEqual(result, expected)

    def test_wildcards(self):
        "All the legacy spellings of the wildcard are normalized to None"
        reference = Method()
        self.assertEqual(reference, Method(None, None, None, None, None))
        for wildcard in ("*", "any", "all", "", None):
            with self.subTest(wildcard=wildcard):
                self.assertEqual(Method(wildcard, wildcard, wildcard, wildcard, wildcard),
                                 reference)
        # 0 used to be the wildcard for the dimensionality
        self.assertEqual(Method(0), reference)

    def test_normalization(self):
        "Case, aliases and mutable targets are normalized at construction time"
        method = Method(1, "FULL", "Histo", "OpenCL", [0, 1])
        self.assertEqual(method, Method(1, "full", "histogram", "opencl", (0, 1)))
        self.assertIsInstance(method.target, tuple, "target is hashable")
        self.assertEqual(hash(method), hash(Method(1, "full", "histogram", "opencl", (0, 1))))
        self.assertEqual(Method(2, "nosplit", "csr", "numpy"),
                         Method(2, "no", "csr", "python"))
        self.assertEqual(method, (1, "full", "histogram", "opencl", (0, 1)),
                         "still compares equal to a plain tuple")

    def test_validation(self):
        "Unsupported values are rejected instead of silently matching nothing"
        for kwargs in ({"dim": 3},
                       {"split": "bidon"},
                       {"algo": "bidon"},
                       {"impl": "bidon"}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    Method(**kwargs)

    def test_normalization_is_not_bypassed(self):
        "_replace() and _make() must not build invalid instances (see #2757)"
        method = Method(1, "full", "csr", "opencl", (0, 0))
        self.assertEqual(method._replace(algo="LUT", target=[1, 2]),
                         Method(1, "full", "lut", "opencl", (1, 2)))
        self.assertEqual(Method._make([1, "FULL", "CSR", "CYTHON", None]),
                         Method(1, "full", "csr", "cython", None))
        self.assertIsInstance(method._replace(target=[1, 2]).target, tuple)

    def test_with_accessors(self):
        method = Method(1, "full", "csr", "opencl", (0, 0))
        self.assertEqual(method.with_dim(2), Method(2, "full", "csr", "opencl", (0, 0)))
        self.assertEqual(method.with_split("no"), Method(1, "no", "csr", "opencl", (0, 0)))
        self.assertEqual(method.with_algo("lut"), Method(1, "full", "lut", "opencl", (0, 0)))
        self.assertEqual(method.with_impl("cython"), Method(1, "full", "csr", "cython", (0, 0)))
        self.assertEqual(method.with_target(None), Method(1, "full", "csr", "opencl", None))
        self.assertEqual(method.with_dim(None).with_dim(1), method, "wildcards round-trip")

    def test_is_concrete(self):
        self.assertTrue(Method(1, "no", "csr", "cython").is_concrete)
        self.assertFalse(Method(1, "no", "csr").is_concrete)
        self.assertFalse(Method().is_concrete)

    def test_degraded(self):
        "The degradation path ends on a method which degrades to itself"
        expected = [Method(2, "full", "csr", "cython", None),
                    Method(2, "full", "histogram", "cython", None),
                    Method(2, "pseudo", "histogram", "cython", None),
                    Method(2, "bbox", "histogram", "cython", None),
                    Method(2, "no", "histogram", "cython", None),
                    Method(2, "no", "histogram", "python", None)]
        method = Method(2, "full", "csr", "opencl", (0, 0))
        for step in expected:
            former = method
            method = method.degraded()
            self.assertEqual(method, step, f"{former}.degraded() => {method} != {step}")
        self.assertEqual(method.degraded(), method, "fail-safe method is a fixed point")

    def test_parse_any(self):
        samples = [(None, Method()),
                   ("csr_ocl", Method(None, None, "csr", "opencl")),
                   (("full", "csr", "cython"), Method(None, "full", "csr", "cython")),
                   ((1, "full", "csr", "cython"), Method(1, "full", "csr", "cython")),
                   ((1, "full", "csr", "opencl", (0, 0)), Method(1, "full", "csr", "opencl", (0, 0))),
                   ({"split": "no", "algorithm": "lut", "implementation": "cython"},
                    Method(None, "no", "lut", "cython")),
                   (Method(1, "no", "csr", "cython"), Method(1, "no", "csr", "cython")),
                   ]
        for value, expected in samples:
            with self.subTest(value=value):
                self.assertEqual(Method.parse_any(value), expected)
        self.assertEqual(Method.parse_any(["full", "csr", "cython"], dim=2),
                         Method(2, "full", "csr", "cython"))
        self.assertEqual(Method.parse_any("csr_ocl", target=(1, 2)),
                         Method(None, None, "csr", "opencl", (1, 2)))
        with self.assertRaises(TypeError):
            Method.parse_any((1, 2))

    def test_2292(self):
        self.assertNotEqual(IntegrationMethod.select_one_available(("No","Histo","Python"), 2), None, "properly change case and histo->histogram")


def suite():
    testsuite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite.addTest(loader(TestMethod))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
