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

"""
Test for OpenCL sorting on GPU
"""

from __future__ import absolute_import, print_function, division
__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__date__ = "22/09/2015"
__copyright__ = "2015, ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import sys
import os
import unittest
import numpy
import logging
is_main = (__name__ == '__main__')
if is_main:
    import pkgutil
    __path__ = pkgutil.extend_path([os.path.dirname(__file__)], "pyFAI.test")
from .utilstest import UtilsTest, getLogger

logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]


try:
    import pyopencl
except ImportError as error:
    logger.warning("OpenCL module (pyopencl) is not present, skip tests. %s." % error)
    skip = True
else:
    skip = False


from pyFAI import ocl_sort


class TestOclSort(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.shape = (256, 500)
        self.ary = numpy.random.random(self.shape).astype(numpy.float32)
        self.sorted = numpy.sort(self.ary.copy(), axis=0)
        if logger.level < logging.INFO:
            self.PROFILE = True
        else:
            self.PROFILE = False

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.shape = self.ary = self.sorted = None

    def test_sort(self):
        s = ocl_sort.Separator(self.shape[0], self.shape[1], profile=self.PROFILE)
        res = s.sort_vertical(self.ary).get()
        self.assert_(numpy.allclose(self.sorted, res), "vertical sort is OK")
        if self.PROFILE:
            s.log_profile()


def test_suite_all_ocl_sort():
    testSuite = unittest.TestSuite()
    if skip:
        logger.warning("OpenCL module (pyopencl) is not present or no device available: skip test_ocl_sort")
    else:
        testSuite.addTest(TestOclSort("test_sort"))
    return testSuite

if is_main:
    mysuite = test_suite_all_ocl_sort()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
    UtilsTest.clean_up()
