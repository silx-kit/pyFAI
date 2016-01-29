# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/pyFAI/pyFAI
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

from __future__ import absolute_import, print_function, division
__doc__ = """Test for OpenCL sorting on GPU"""
__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__date__ = "29/01/2016"
__copyright__ = "2015, ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import sys
import os
import unittest
import numpy
import logging
from .utilstest import UtilsTest, getLogger

logger = getLogger(__file__)


try:
    import pyopencl
except ImportError as error:
    logger.warning("OpenCL module (pyopencl) is not present, skip tests. %s." % error)
    skip = True
else:
    skip = False
    from .. import ocl_sort


class TestOclSort(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.shape = (128, 256)
        self.ary = numpy.random.random(self.shape).astype(numpy.float32)
        self.sorted_vert = numpy.sort(self.ary.copy(), axis=0)
        self.sorted_hor = numpy.sort(self.ary.copy(), axis=1)
        self.vector_vert = self.sorted_vert[self.shape[0] // 2]
        self.vector_hor = self.sorted_hor[:, self.shape[1] // 2]
        if logger.level < logging.INFO:
            self.PROFILE = True
        else:
            self.PROFILE = False

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.shape = self.ary = self.sorted_vert = self.sorted_hor = self.vector_vert = self.sorted_hor = None

    def test_sort_vert(self):
        s = ocl_sort.Separator(self.shape[0], self.shape[1], profile=self.PROFILE)
        res = s.sort_vertical(self.ary).get()
        self.assert_(numpy.allclose(self.sorted_vert, res), "vertical sort is OK")
        if self.PROFILE:
            s.log_profile()
            s.reset_timer()

    def test_filter_vert(self):
        s = ocl_sort.Separator(self.shape[0], self.shape[1], profile=self.PROFILE)
        res = s.filter_vertical(self.ary).get()
#         import pylab
#         pylab.plot(self.vector, label="ref")
#         pylab.plot(res, label="obt")
#         pylab.legend()
#         pylab.show()
#         raw_input()
        self.assert_(numpy.allclose(self.vector_vert, res), "vertical filter is OK")
        if self.PROFILE:
            s.log_profile()
            s.reset_timer()

    def test_sort_hor(self):
        s = ocl_sort.Separator(self.shape[0], self.shape[1], profile=self.PROFILE)
        res = s.sort_horizontal(self.ary).get()
        self.assert_(numpy.allclose(self.sorted_hor, res), "horizontal sort is OK")
        if self.PROFILE:
            s.log_profile()
            s.reset_timer()

    def test_filter_hor(self):
        s = ocl_sort.Separator(self.shape[0], self.shape[1], profile=self.PROFILE)
        res = s.filter_horizontal(self.ary).get()
#         import pylab
#         pylab.plot(self.vector_hor, label="ref")
#         pylab.plot(res, label="obt")
#         pylab.legend()
#         pylab.show()
#         raw_input()
        self.assert_(numpy.allclose(self.vector_hor, res), "horizontal filter is OK")
        if self.PROFILE:
            s.log_profile()
            s.reset_timer()


def suite():
    testsuite = unittest.TestSuite()
    if skip:
        logger.warning("OpenCL module (pyopencl) is not present or no device available: skip test_ocl_sort")
    else:
        testsuite.addTest(TestOclSort("test_sort_hor"))
        testsuite.addTest(TestOclSort("test_sort_vert"))
        testsuite.addTest(TestOclSort("test_filter_hor"))
        testsuite.addTest(TestOclSort("test_filter_vert"))
    return testsuite

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
    UtilsTest.clean_up()
