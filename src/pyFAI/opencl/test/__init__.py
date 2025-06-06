# coding: utf-8
#
#    Project: pyFAI
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2012-2020  European Synchrotron Radiation Facility, Grenoble, France
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

__authors__ = ["J. Kieffer"]
__license__ = "MIT"
__date__ = "30/10/2024"

import unittest
from ...test.utilstest import UtilsTest


def suite():
    testSuite = unittest.TestSuite()

    if UtilsTest.opencl:
        from . import test_addition
        from . import test_preproc
        from . import test_ocl_histo
        from . import test_ocl_azim_csr
        from . import test_ocl_azim_lut
        from . import test_peak_finder
        from . import test_ocl_sort
        from . import test_openCL
        from . import test_collective
        testSuite.addTests(test_addition.suite())
        testSuite.addTests(test_preproc.suite())
        testSuite.addTests(test_openCL.suite())
        testSuite.addTests(test_ocl_histo.suite())
        testSuite.addTests(test_ocl_azim_csr.suite())
        testSuite.addTests(test_ocl_azim_lut.suite())
        testSuite.addTests(test_peak_finder.suite())
        testSuite.addTests(test_ocl_sort.suite())
        testSuite.addTests(test_collective.suite())
    return testSuite
