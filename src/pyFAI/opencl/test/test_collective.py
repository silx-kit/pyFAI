#!/usr/bin/env python3
# coding: utf-8
#
#    Project: Basic OpenCL test
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
Simple test for collective functions
"""

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "2013 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "30/10/2024"

import logging
import numpy
import platform
import unittest
from .. import ocl
if ocl:
    import pyopencl.array
from ...test.utilstest import UtilsTest
from silx.opencl.common import _measure_workgroup_size
from silx.opencl.utils import get_opencl_code

logger = logging.getLogger(__name__)


@unittest.skipIf(UtilsTest.opencl is False, "User request to skip OpenCL tests")
@unittest.skipUnless(ocl, "PyOpenCl is missing")
class TestReduction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestReduction, cls).setUpClass()

        if ocl:
            cls.ctx = ocl.create_context()
            if logger.getEffectiveLevel() <= logging.INFO:
                cls.PROFILE = True
                properties = pyopencl.command_queue_properties.PROFILING_ENABLE
                cls.queue = pyopencl.CommandQueue(cls.ctx, properties=properties)
            else:
                cls.PROFILE = False
                cls.queue = pyopencl.CommandQueue(cls.ctx)
            cls.max_valid_wg = cls.ctx.devices[0].max_work_group_size
            if (platform.machine().startswith("ppc") and
                cls.ctx.devices[0].platform.name.startswith("Portable")
                and cls.ctx.devices[0].type == pyopencl.device_type.GPU):
                raise unittest.SkipTest("Skip test on Power9 GPU with PoCL driver")

    @classmethod
    def tearDownClass(cls):
        super(TestReduction, cls).tearDownClass()
        print("Maximum valid workgroup size %s on device %s" % (cls.max_valid_wg, cls.ctx.devices[0]))
        cls.ctx = None
        cls.queue = None

    def setUp(self):
        if ocl is None:
            return
        self.shape = 4096
        rng = UtilsTest.get_rng()
        self.data = rng.poisson(10, size=self.shape).astype(numpy.int32)
        self.data_d = pyopencl.array.to_device(self.queue, self.data)
        self.sum_d = pyopencl.array.zeros_like(self.data_d)
        self.program = pyopencl.Program(self.ctx, get_opencl_code("pyfai:openCL/collective/reduction.cl")+
                                        get_opencl_code("pyfai:openCL/collective/scan.cl")
                                        ).build()

    def tearDown(self):
        self.img = self.data = None
        self.data_d = self.sum_d = self.program = None

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_reduction(self):
        """
        tests the sum_int_reduction function
        """
        # rec_workgroup = self.program.test_sum_int_reduction.get_work_group_info(pyopencl.kernel_work_group_info.WORK_GROUP_SIZE, self.ctx.devices[0])
        maxi = int(round(numpy.log2(min(self.shape,self.max_valid_wg))))+1
        for i in range(maxi):
            wg = 1 << i
            try:
                evt = self.program.test_sum_int_reduction(self.queue, (self.shape,), (wg,),
                                                          self.data_d.data,
                                                          self.sum_d.data,
                                                          pyopencl.LocalMemory(4*wg))
                evt.wait()
            except Exception as error:
                logger.error("Error %s on WG=%s: test_reduction", error, wg)
                break
            else:
                res = self.sum_d.get()
                ref = numpy.outer(self.data.reshape((-1, wg)).sum(axis=-1),numpy.ones(wg)).ravel()
                good = numpy.allclose(res, ref)
                logger.info("Wg: %s result: reduction OK %s", wg, good)
                self.assertTrue(good, "calculation is correct for WG=%s" % wg)

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_atomic(self):
        """
        tests the sum_int_atomic function
        """

        maxi = int(round(numpy.log2(min(self.shape, self.max_valid_wg))))+1
        for i in range(maxi):
            wg = 1 << i
            try:
                evt = self.program.test_sum_int_atomic(self.queue, (self.shape,), (wg,),
                                                          self.data_d.data,
                                                          self.sum_d.data,
                                                          pyopencl.LocalMemory(4*wg))
                evt.wait()
            except Exception as error:
                logger.error("Error %s on WG=%s: test_atomic", error, wg)
                break
            else:
                res = self.sum_d.get()
                ref = numpy.outer(self.data.reshape((-1, wg)).sum(axis=-1),numpy.ones(wg)).ravel()
                good = numpy.allclose(res, ref)
                logger.info("Wg: %s result: atomic good: %s", wg, good)
                self.assertTrue(good, "calculation is correct for WG=%s" % wg)

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_Hillis_Steele(self):
        """
        tests the Hillis_Steele scan function
        """
        data_d = pyopencl.array.to_device(self.queue, self.data.astype("float32"))
        scan_d = pyopencl.array.empty_like(data_d)
        maxi = int(round(numpy.log2(min(self.shape, self.max_valid_wg))))+1
        for i in range(0, maxi):
            wg = 1 << i
            try:
                evt = self.program.test_cumsum(self.queue, (self.shape,), (wg,),
                                                          data_d.data,
                                                          scan_d.data,
                                                          pyopencl.LocalMemory(2*4*wg))
                evt.wait()
            except Exception as error:
                logger.error("Error %s on WG=%s: Hillis_Steele", error, wg)
                break
            else:
                res = scan_d.get().reshape((-1, wg))
                ref = numpy.array([numpy.cumsum(i) for i in self.data.reshape((-1, wg))])
                good = numpy.allclose(res, ref)
                logger.info("Wg: %s result: cumsum good: %s", wg, good)
                self.assertTrue(good, "Cumsum calculation is correct for WG=%s" % wg)

    @unittest.skipUnless(ocl, "pyopencl is missing")
    @unittest.skip("Fix me")
    def test_Blelloch(self):
        """
        tests the Blelloch scan function
        """
        data_d = pyopencl.array.to_device(self.queue, self.data.astype("float32"))
        scan_d = pyopencl.array.empty_like(data_d)
        maxi = int(round(numpy.log2(min(self.shape, self.max_valid_wg))))+1
        for i in range(maxi):
            wg = 1 << i
            try:
                evt = self.program.test_blelloch_scan(self.queue, (self.shape,), (wg,),
                                                          data_d.data,
                                                          scan_d.data,
                                                          pyopencl.LocalMemory(2*4*wg))
                evt.wait()
            except Exception as error:
                logger.error("Error %s on WG=%s: Hillis_Steele", error, wg)
                break
            else:
                res = self.sum_d.get()
                ref = numpy.array([numpy.cumsum(i) for i in self.data.reshape((-1, wg))])
                good = numpy.allclose(res, ref)
                logger.info("Wg: %s result: cumsum good: %s", wg, good)
                self.assertTrue(good, "calculation is correct for WG=%s" % wg)


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testSuite = unittest.TestSuite()
    testSuite.addTest(loader(TestReduction))
    return testSuite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")