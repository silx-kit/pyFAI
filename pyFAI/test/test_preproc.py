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

"test suite for preprocessing corrections"

from __future__ import absolute_import, division, print_function

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "25/02/2019"


import os
import unittest
import numpy
import logging

logger = logging.getLogger(__name__)


from ..engines import preproc as python_preproc
from ..ext import preproc as cython_preproc
from .utilstest import UtilsTest


class TestPreproc(unittest.TestCase):
    def one_test(self, preproc):
        """
        The final pattern should look like a 4x4 square with 1 and -1 elsewhere.

        :param preproc: the preproc module to use
        """
        logger.debug("using preproc from: %s", preproc.__name__)
        shape = 8, 8
        size = shape[0] * shape[1]
        target = numpy.ones(shape)
        dummy = -1.0
        target[:2, :] = dummy
        target[-2:, :] = dummy
        target[:, -2:] = dummy
        target[:, :2] = dummy
        mask = numpy.zeros(shape, "int8")
        mask[:2, :] = 1
        dark = numpy.random.poisson(10, size).reshape(shape)
        flat = 1.0 + numpy.random.random(shape)
        scale = 10.0
        raw = scale * flat + dark
        raw[-2:, :] = numpy.NaN

        raw[:, :2] = dummy
        flat[:, -2:] = dummy

        epsilon = 1e-3

        # add some tests with various levels of conditioning
        res = preproc.preproc(raw)
        # then Nan on last lines -> 0
        self.assertEqual(abs(res[-2:, 2:]).max(), 0, "Nan filtering")
        self.assertGreater(abs(res[:-2, 2:]).max(), scale, "untouched other")

        res = preproc.preproc(raw, empty=-1)
        # then Nan on last lines -> -1
        self.assertEqual(abs(res[-2:, :] + 1).max(), 0, "Nan filtering with empty filling")

        # test dummy
        res = preproc.preproc(raw, dummy=dummy, delta_dummy=0.5)
        self.assertEqual(abs(res[-2:, :] + 1).max(), 0, "dummy filtering")

        # test polarization, solidangle and sensor thickness  with dummy.
        res = preproc.preproc(raw, dark, polarization=flat, dummy=dummy, mask=mask, normalization_factor=scale)

        self.assertEqual(abs(numpy.round(res[2:-2, 2:-2]) - 1).max(), 0, "mask is properly applied")

        # search for numerical instability:
        # delta = abs(numpy.round(res, 3) - target).max()
        delta = abs(res - target).max()
        if delta <= epsilon:
            ll = ["flat != polarization",
                  str(preproc),
                  "raw:", str(raw),
                  "dark", str(dark),
                  "flat:", str(flat),
                  "delta", str(delta)]
            logger.warning(os.linesep.join(ll))

        delta = abs(res - target).max()
        self.assertGreater(delta, epsilon, "flat != polarization")

        res = preproc.preproc(raw, dark, solidangle=flat, dummy=dummy, mask=mask, normalization_factor=scale)
        self.assertEqual(abs(numpy.round(res[2:-2, 2:-2]) - 1).max(), 0, "mask is properly applied")
        delta = abs(res - target).max()
        if delta <= epsilon:
            ll = ["flat != solidangle",
                  str(preproc),
                  "raw:", str(raw),
                  "dark", str(dark),
                  "flat:", str(flat),
                  "delta", str(delta)]
            logger.warning(os.linesep.join(ll))
        self.assertGreater(delta, epsilon, "flat != solidangle")

        res = preproc.preproc(raw, dark, absorption=flat, dummy=dummy, mask=mask, normalization_factor=scale)
        self.assertEqual(abs(numpy.round(res[2:-2, 2:-2]) - 1).max(), 0, "mask is properly applied")
        delta = abs(res - target).max()
        if delta <= epsilon:
            ll = ["flat != absorption",
                  str(preproc),
                  "raw:", str(raw),
                  "dark", str(dark),
                  "flat:", str(flat),
                  "delta", str(delta)]
            logger.warning(os.linesep.join(ll))
        self.assertGreater(delta, epsilon, "flat != absorption")

        # Test all features together
        res = preproc.preproc(raw, dark=dark, flat=flat, dummy=dummy, mask=mask, normalization_factor=scale)
        self.assertLessEqual(abs(res - target).max(), 1e-6, "test all features ")

    def test_python(self):
        self.one_test(python_preproc)

    def test_cython(self):
        self.one_test(cython_preproc)

    @unittest.skipIf(UtilsTest.opencl is False, "User request to skip OpenCL tests")
    def test_opencl(self):
        from ..opencl import ocl
        if ocl is None:
            self.skipTest("OpenCL not available")
        from ..opencl import preproc as ocl_preproc
        self.one_test(ocl_preproc)


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestPreproc))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
