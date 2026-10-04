#!/usr/bin/env python3
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
__date__ = "03/10/2026"

import logging
import platform
import unittest

import numpy

from .. import ocl

if ocl:
    import pyopencl.array
from silx.opencl.utils import get_opencl_code

from ...test.utilstest import UtilsTest

logger = logging.getLogger(__name__)


@unittest.skipIf(UtilsTest.opencl is False, "User request to skip OpenCL tests")
@unittest.skipUnless(ocl, "PyOpenCl is missing")
class TestGroupFunction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

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
        super().tearDownClass()
        # print("Maximum valid workgroup size %s on device %s" % (cls.max_valid_wg, cls.ctx.devices[0]))
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
                                        get_opencl_code("pyfai:openCL/collective/scan.cl")+
                                        get_opencl_code("pyfai:openCL/collective/comb_sort.cl")).build()

    def tearDown(self):
        self.img = self.data = None
        self.data_d = self.sum_d = self.program = None

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_reduction(self):
        """
        tests the sum_int_reduction function
        """
        # rec_workgroup = self.program.test_sum_int_reduction.get_work_group_info(pyopencl.kernel_work_group_info.WORK_GROUP_SIZE, self.ctx.devices[0])
        maxi = round(numpy.log2(min(self.shape,self.max_valid_wg)))+1
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
                self.assertTrue(good, f"calculation is correct for WG={wg}")

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_atomic(self):
        """
        tests the sum_int_atomic function
        """

        maxi = round(numpy.log2(min(self.shape, self.max_valid_wg)))+1
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
                self.assertTrue(good, f"calculation is correct for WG={wg}")

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_Hillis_Steele(self):
        """
        tests the Hillis_Steele scan function
        """
        data_d = pyopencl.array.to_device(self.queue, self.data.astype("float32"))
        scan_d = pyopencl.array.empty_like(data_d)
        maxi = round(numpy.log2(min(self.shape, self.max_valid_wg)))+1
        for i in range(maxi):
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
                self.assertTrue(good, f"Cumsum calculation is correct for WG={wg}")

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_Blelloch(self):
        """
        tests the Blelloch scan function
        """
        data_d = pyopencl.array.to_device(self.queue, self.data.astype("float32"))
        scan_d = pyopencl.array.empty_like(data_d)
        maxi = round(numpy.log2(min(self.shape/2, self.max_valid_wg)))+1
        for i in range(maxi):
            wg = 1 << i
            try:
                evt = self.program.test_blelloch_scan(self.queue, (self.shape//2,), (wg,),
                                                          data_d.data,
                                                          scan_d.data,
                                                          pyopencl.LocalMemory(2*4*wg))
                evt.wait()
            except Exception as error:
                logger.error("Error %s on WG=%s: Blelloch", error, wg)
                break
            else:
                res = scan_d.get().reshape((-1, 2*wg))
                ref = numpy.array([numpy.cumsum(i) for i in self.data.reshape((-1, 2*wg))])
                good = numpy.allclose(res, ref)
                if not good:
                    print(ref)
                    print(res)
                logger.info("Wg: %s result: cumsum good: %s", wg, good)
                self.assertTrue(good, f"calculation is correct for WG={wg}")

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_Blelloch_multipass(self):
        """
        tests the Blelloch cumsum using multiple passes ...
        """
        data_d = pyopencl.array.to_device(self.queue, self.data.astype("float32"))
        scan_d = pyopencl.array.empty_like(data_d)
        maxi = round(numpy.log2(min(self.shape/2, self.max_valid_wg)))+1
        for i in range(maxi):
            wg = 1 << i
            try:
                evt = self.program.test_blelloch_multi(self.queue, (wg,), (wg,),
                                                       data_d.data,
                                                       scan_d.data,
                                                       numpy.int32(self.shape),
                                                       pyopencl.LocalMemory(2*4*wg))
                evt.wait()
            except Exception as error:
                logger.error("Error %s on WG=%s: Blelloch multi", error, wg)
                break
            else:
                res = scan_d.get()
                ref = numpy.cumsum(self.data)
                good = numpy.allclose(res, ref)
                if not good:
                    print(ref)
                    print(res)
                logger.info("Wg: %s result: cumsum good: %s", wg, good)
                self.assertTrue(good, f"calculation is correct for WG={wg}")


    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_sort(self):
        """
        tests the sort of floating points in a workgroup
        """
        data = numpy.arange(self.shape).astype(numpy.float32)
        numpy.random.shuffle(data)
        data_d = pyopencl.array.to_device(self.queue, data)

        maxi = round(numpy.log2(self.shape))+1
        for i in range(5,maxi):
            wg = 1 << i

            ref = data.reshape((-1, wg))
            positions = ((numpy.arange(ref.shape[0])+1)*wg).astype(numpy.int32)
            positions_d = pyopencl.array.to_device(self.queue, positions)
            data_d = pyopencl.array.to_device(self.queue, data)
            # print(ref.shape, (ref.shape[0],min(wg, self.max_valid_wg)), (1, min(wg, self.max_valid_wg)), positions)
            try:
                evt = self.program.test_combsort_float(self.queue, (min(wg, self.max_valid_wg), ref.shape[0]), (min(wg, self.max_valid_wg), 1),
                                                       data_d.data,
                                                       positions_d.data,
                                                       pyopencl.LocalMemory(4*min(wg, self.max_valid_wg)))
                evt.wait()
            except Exception as error:
                logger.error("Error %s on WG=%s: test_sort", error, wg)
                break
            else:
                res = data_d.get()
                ref = numpy.sort(ref)
                good = numpy.allclose(res, ref.ravel())
                logger.info("Wg: %s result: sort OK %s", wg, good)
                if not good:
                    print(res.reshape(ref.shape))
                    print(ref)
                    print(numpy.where(res.reshape(ref.shape)-ref))

                self.assertTrue(good, f"calculation is correct for WG={wg}")

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_sort4(self):
        """
        tests the sort of floating points in a workgroup
        """
        data = numpy.arange(self.shape).astype(numpy.float32)
        data = numpy.outer(data, numpy.ones(4, numpy.float32)).view(numpy.dtype([("s0","<f4"),("s1","<f4"),("s2","<f4"),("s3","<f4")]))
        numpy.random.shuffle(data)
        data_d = pyopencl.array.to_device(self.queue, data)

        maxi = round(numpy.log2(self.shape))+1
        for i in range(5,maxi):
            wg = 1 << i

            ref = data.reshape((-1, wg))
            positions = ((numpy.arange(ref.shape[0])+1)*wg).astype(numpy.int32)
            positions_d = pyopencl.array.to_device(self.queue, positions)
            data_d = pyopencl.array.to_device(self.queue, data)
            # print(ref.shape, (ref.shape[0],min(wg, self.max_valid_wg)), (1, min(wg, self.max_valid_wg)), positions)
            try:
                evt = self.program.test_combsort_float4(self.queue, (min(wg, self.max_valid_wg), ref.shape[0]), (min(wg, self.max_valid_wg),1),
                                                       data_d.data,
                                                       positions_d.data,
                                                       pyopencl.LocalMemory(4*min(wg, self.max_valid_wg)))
                evt.wait()
            except Exception as error:
                logger.error("Error %s on WG=%s: test_sort", error, wg)
                break
            else:
                res = data_d.get()
                # print(res.dtype)
                ref = numpy.sort(ref, order="s0")
                # print(ref.dtype)
                good = numpy.allclose(res.view(numpy.float32).ravel(), ref.view(numpy.float32).ravel())
                logger.info("Wg: %s result: sort OK %s", wg, good)
                if not good:
                    print(res.reshape(ref.shape))
                    print(ref)
                    print(numpy.where(res.reshape(ref.shape)-ref))

                self.assertTrue(good, f"calculation is correct for WG={wg}")

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_sort2(self):
        """
        tests the sort of (key, payload) pairs stored as float2
        """
        data = numpy.arange(self.shape).astype(numpy.float32)
        data = numpy.outer(data, numpy.ones(2, numpy.float32)).view(numpy.dtype([("s0", "<f4"), ("s1", "<f4")]))
        numpy.random.shuffle(data)

        maxi = round(numpy.log2(self.shape)) + 1
        for i in range(5, maxi):
            wg = 1 << i
            ref = data.reshape((-1, wg))
            positions = ((numpy.arange(ref.shape[0]) + 1) * wg).astype(numpy.int32)
            positions_d = pyopencl.array.to_device(self.queue, positions)
            data_d = pyopencl.array.to_device(self.queue, data)
            try:
                evt = self.program.test_combsort_float2(self.queue, (min(wg, self.max_valid_wg), ref.shape[0]), (min(wg, self.max_valid_wg), 1),
                                                        data_d.data,
                                                        positions_d.data,
                                                        pyopencl.LocalMemory(4 * min(wg, self.max_valid_wg)))
                evt.wait()
            except Exception as error:
                logger.error("Error %s on WG=%s: test_sort2", error, wg)
                break
            else:
                res = data_d.get()
                ref = numpy.sort(ref, order="s0")
                good = numpy.array_equal(res.view(numpy.float32).ravel(), ref.view(numpy.float32).ravel())
                logger.info("Wg: %s result: sort OK %s", wg, good)
                self.assertTrue(good, f"calculation is correct for WG={wg}")

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_sort_local(self):
        """
        tests the comb sort of (key, position) pairs held in local memory
        """
        data = numpy.arange(self.shape).astype(numpy.float32)
        numpy.random.shuffle(data)

        maxi = round(numpy.log2(self.shape)) + 1
        for i in range(5, maxi):
            wg = 1 << i
            ref = data.reshape((-1, wg))
            positions = ((numpy.arange(ref.shape[0]) + 1) * wg).astype(numpy.int32)
            positions_d = pyopencl.array.to_device(self.queue, positions)
            data_d = pyopencl.array.to_device(self.queue, data)
            lwg = min(wg, self.max_valid_wg)
            try:
                evt = self.program.test_combsort_local(self.queue, (lwg, ref.shape[0]), (lwg, 1),
                                                       data_d.data,
                                                       positions_d.data,
                                                       pyopencl.LocalMemory(4 * lwg),
                                                       pyopencl.LocalMemory(4 * wg),
                                                       pyopencl.LocalMemory(4 * wg))
                evt.wait()
            except Exception as error:
                logger.error("Error %s on WG=%s: test_sort_local", error, wg)
                break
            else:
                res = data_d.get()
                good = numpy.array_equal(res, numpy.sort(ref).ravel())
                logger.info("Wg: %s result: sort OK %s", wg, good)
                self.assertTrue(good, f"calculation is correct for WG={wg}")

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_sort_empty_bin(self):
        """Empty and single-element bins used to hang first_step() forever: with
        size<2 the step decreases down to 0 and `0>=size` never becomes false.
        """
        sizes = [0, 1, 0, 2, 5, 0, 1]
        positions = numpy.cumsum(sizes).astype(numpy.int32)
        wg = min(32, self.max_valid_wg)
        shared = pyopencl.LocalMemory(4 * wg)
        rng = UtilsTest.get_rng()

        data = rng.random(positions[-1]).astype(numpy.float32)
        data_d = pyopencl.array.to_device(self.queue, data)
        positions_d = pyopencl.array.to_device(self.queue, positions)
        self.program.test_combsort_float(self.queue, (wg, len(sizes)), (wg, 1),
                                         data_d.data, positions_d.data, shared).wait()
        res = data_d.get()

        data4 = numpy.outer(data, numpy.ones(4, numpy.float32)).view(
            numpy.dtype([("s0", "<f4"), ("s1", "<f4"), ("s2", "<f4"), ("s3", "<f4")]))
        data4_d = pyopencl.array.to_device(self.queue, data4)
        self.program.test_combsort_float4(self.queue, (wg, len(sizes)), (wg, 1),
                                          data4_d.data, positions_d.data, shared).wait()
        res4 = data4_d.get()["s0"].ravel()

        data2 = numpy.outer(data, numpy.ones(2, numpy.float32)).view(
            numpy.dtype([("s0", "<f4"), ("s1", "<f4")]))
        data2_d = pyopencl.array.to_device(self.queue, data2)
        self.program.test_combsort_float2(self.queue, (wg, len(sizes)), (wg, 1),
                                          data2_d.data, positions_d.data, shared).wait()
        res2 = data2_d.get()["s0"].ravel()

        local_d = pyopencl.array.to_device(self.queue, data)
        capacity = max(max(sizes), 1)
        self.program.test_combsort_local(self.queue, (wg, len(sizes)), (wg, 1),
                                         local_d.data, positions_d.data, shared,
                                         pyopencl.LocalMemory(4 * capacity),
                                         pyopencl.LocalMemory(4 * capacity)).wait()
        res_local = local_d.get()

        start = 0
        for size in sizes:
            ref = numpy.sort(data[start:start + size])
            self.assertTrue(numpy.array_equal(res[start:start + size], ref),
                            f"float sort is correct for the bin of size {size}")
            self.assertTrue(numpy.array_equal(res2[start:start + size], ref),
                            f"float2 sort is correct for the bin of size {size}")
            self.assertTrue(numpy.array_equal(res4[start:start + size], ref),
                            f"float4 sort is correct for the bin of size {size}")
            self.assertTrue(numpy.array_equal(res_local[start:start + size], ref),
                            f"local sort is correct for the bin of size {size}")
            start += size


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testSuite = unittest.TestSuite()
    testSuite.addTest(loader(TestGroupFunction))
    return testSuite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
