#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Median filter of images + OpenCL
#             https://github.com/silx-kit/silx
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
Simple test of the median filter
"""

from __future__ import division, print_function

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "2013-2017 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "09/02/2017"

import time
import logging
import numpy
import unittest
from collections import namedtuple
from ..common import ocl
if ocl:
    import pyopencl
    import pyopencl.array
    from .. import medfilt

logger = logging.getLogger(__name__)

Result = namedtuple("Result", ["size", "error", "sp_time", "oc_time"])

try:
    from scipy.misc import ascent
except:
    def ascent():
        """Dummy image from random data"""
        return numpy.random.random((512, 512))

from scipy.ndimage import filters

@unittest.skipUnless(ocl, "PyOpenCl is missing")
class TestMedianFilter(unittest.TestCase):

#     @classmethod
#     def setUpClass(cls):
#         super(TestAddition, cls).setUpClass()
#         if ocl:
#             cls.ctx = ocl.create_context()
#             if logger.getEffectiveLevel() <= logging.INFO:
#                 cls.PROFILE = True
#                 cls.queue = pyopencl.CommandQueue(
#                                 cls.ctx,
#                                 properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
#             else:
#                 cls.PROFILE = False
#                 cls.queue = pyopencl.CommandQueue(cls.ctx)
#             cls.max_valid_wg = 0

#     @classmethod
#     def tearDownClass(cls):
#         super(TestAddition, cls).tearDownClass()
#         print("Maximum valid workgroup size %s on device %s" % (cls.max_valid_wg, cls.ctx.devices[0]))
#         cls.ctx = None
#         cls.queue = None

    def setUp(self):
        if ocl is None:
            return
        self.data = ascent().astype(numpy.float32)
        self.medianfilter = medfilt.MedianFilter2D(self.data.shape)

    def tearDown(self):
        self.data = None
        self.medianfilter = None

    def measure(self, size):
        "Common measurement of accuracy and timimgs"
        t0 = time.time()
        ref = filters.median_filter(self.data, size, mode="nearest"),
        t1 = time.time()
        got = self.medianfilter.medfilt2d(self.data, size)
        t2 = time.time()
        delta = abs(got - ref).max()
        return Result(size, delta, t1 - t0, t2 - t1)

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_medfilt(self):
        """
        tests the median filter kernel
        """
        r = self.measure(size=9)
        logger.info("test_medfilt: size: %s error %s, t_ref: %.3fs, t_obt: %.3fs" % r)
        self.assert_(r.error == 0, 'Results are correct')

    def benchmark(self):
        from pylab import *
        f = figure()
        sp = f.add_subplot(1, 1, 1)
        f.set_title("Median filter of an image 512x512")
        sp.set_xlabel("Window width/height")
        sp.set_tlabel("Executiton time")
        for s in range(3, 31, 2):
            r = self.measure(s)
            print(r)


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestMedianFilter("test_medfilt"))
    return testSuite


def benchmark():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestMedianFilter("test_medfilt"))
    return testSuite



if __name__ == '__main__':
    unittest.main(defaultTest="suite")
