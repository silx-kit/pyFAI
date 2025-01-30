#!/usr/bin/env python3
# coding: utf-8
#
#    Project: pyFAI
#             https://github.com/silx-kit/pyFAI
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

"""
Test of preprocessing
"""

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "19/11/2024"

import logging
import numpy
import unittest
from .. import ocl, get_opencl_code
if ocl:
    import pyopencl.array
from ...test.utilstest import UtilsTest

logger = logging.getLogger(__name__)


@unittest.skipIf(UtilsTest.opencl is False, "User request to skip OpenCL tests")
@unittest.skipUnless(ocl, "PyOpenCl is missing")
class TestPreproc(unittest.TestCase):

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_preproc(self):
        """
        tests the preproc kernel
        """
        from ..preproc import preproc
        ary = numpy.arange(12).reshape(4,3)
        for dtype in (numpy.uint8, numpy.int8, numpy.int16, numpy.uint16, numpy.uint32, numpy.int32, numpy.uint64, numpy.int64, numpy.float32):
            import sys; sys.stderr.write(f"test {dtype}\n")
            self.assertEqual(abs(preproc(ary.astype(dtype),split_result=4)[..., 0]-ary).max(), 0, "Result OK for dtype {dtype}")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testSuite = unittest.TestSuite()
    testSuite.addTest(loader(TestPreproc))
    return testSuite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
