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

"test suite for OpenCL code"

from __future__ import absolute_import, division, print_function

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "11/01/2019"


import unittest
import os
import time
import fabio
import gc
import numpy
import logging
import shutil
import platform

logger = logging.getLogger(__name__)
try:
    import pyopencl
except ImportError as error:
    logger.warning("OpenCL module (pyopencl) is not present, skip tests. %s.", error)
    pyopencl = None

from ..opencl import ocl
if ocl is not None:
    from ..opencl import pyopencl, read_cl_file
    import pyopencl.array
from .. import load
from . import utilstest
from .utilstest import test_options
from ..utils import mathutil
from ..utils.decorators import depreclog


class TestMask(unittest.TestCase):

    def setUp(self):
        if not test_options.opencl:
            self.skipTest("User request to skip OpenCL tests")
        if pyopencl is None or ocl is None:
            self.skipTest("OpenCL module (pyopencl) is not present or no device available")

        self.tmp_dir = os.path.join(test_options.tempdir, "opencl")
        if not os.path.isdir(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        self.N = 1000
        self.datasets = [{"img": test_options.getimage("Pilatus1M.edf"),
                          "poni": test_options.getimage("Pilatus1M.poni"),
                          "spline": None},
                         {"img": test_options.getimage("halfccd.edf"),
                          "poni": test_options.getimage("halfccd.poni"),
                          "spline": test_options.getimage("halfccd.spline")},
#                          {"img": test_options.getimage("Frelon2k.edf"),
#                           "poni": test_options.getimage("Frelon2k.poni"),
#                           "spline": test_options.getimage("frelon.spline")},
#                          {"img": test_options.getimage("Pilatus6M.cbf"),
#                           "poni": test_options.getimage("Pilatus6M.poni"),
#                           "spline": None},
                         ]
        for ds in self.datasets:
            if ds["spline"] is not None:
                with open(ds["poni"], "r") as ponifile:
                    data = ponifile.read()
                # spline = os.path.basename(ds["spline"])
                with open(ds["poni"]) as f:
                    data = []
                    for line in f:
                        if line.startswith("SplineFile:"):
                            data.append("SplineFile: " + ds["spline"])
                        else:
                            data.append(line.strip())
                ds["poni"] = os.path.join(self.tmp_dir, os.path.basename(ds["poni"]))
                with open(ds["poni"], "w") as f:
                    f.write(os.linesep.join(data))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        self.tmp_dir = self.N = self.datasets = None

    @unittest.skipIf(test_options.low_mem, "test using >200M")
    def test_OpenCL(self):
        logger.info("Testing histogram-based algorithm (forward-integration)")
        for devtype in ("GPU", "CPU"):
            ids = ocl.select_device(devtype, extensions=["cl_khr_int64_base_atomics"])
            if ids is None:
                logger.error("No suitable %s OpenCL device found", devtype)
                continue
            else:
                logger.info("I found a suitable device %s %s: %s %s ", devtype, ids, ocl.platforms[ids[0]], ocl.platforms[ids[0]].devices[ids[1]])
                if ocl.platforms[ids[0]].name == "Portable Computing Language":
                    logger.warning("POCL is known error-prone on this test")
                    continue

            for ds in self.datasets:
                ai = load(ds["poni"])
                data = fabio.open(ds["img"]).data
                with utilstest.TestLogging(logger=depreclog, warning=1):
                    # Filter deprecated warning
                    res = ai.xrpd_OpenCL(data, self.N, devicetype="all", platformid=ids[0], deviceid=ids[1], useFp64=True)
                ref = ai.integrate1d(data, self.N, method="splitBBox", unit="2th_deg")
                r = mathutil.rwp(ref, res)
                logger.info("OpenCL histogram vs histogram SplitBBox has R= %.3f for dataset %s", r, ds)
                self.assertTrue(r < 6, "Rwp=%.3f for OpenCL histogram processing of %s" % (r, ds))
                ai.reset()
                del ai, data
                gc.collect()

    @unittest.skipIf(test_options.low_mem, "test using >500M")
    def test_OpenCL_LUT(self):
        logger.info("Testing LUT-based algorithm (backward-integration)")
        for devtype in ("GPU", "CPU"):
            ids = ocl.select_device(devtype, best=True)
            if ids is None:
                logger.error("No suitable %s OpenCL device found", devtype)
                continue
            else:
                logger.info("I found a suitable device %s %s: %s %s ", devtype, ids, ocl.platforms[ids[0]], ocl.platforms[ids[0]].devices[ids[1]])

            for ds in self.datasets:
                ai = load(ds["poni"])
                data = fabio.open(ds["img"]).data
                ref = ai.integrate1d(data, self.N, method="splitBBox", unit="2th_deg")
                try:
                    res = ai.integrate1d(data, self.N, method="ocl_lut_%i,%i" % (ids[0], ids[1]), unit="2th_deg")
                except (pyopencl.MemoryError, MemoryError, pyopencl.RuntimeError, RuntimeError) as error:
                    logger.warning("Memory error on %s dataset %s: %s%s. Converted into warnining: device may not have enough memory.", devtype, os.path.basename(ds["img"]), os.linesep, error)
                    break
                else:
                    r = mathutil.rwp(ref, res)
                    logger.info("OpenCL CSR vs histogram SplitBBox has R= %.3f for dataset %s", r, ds)
                    self.assertTrue(r < 3, "Rwp=%.3f for OpenCL LUT processing of %s" % (r, ds))
                ai.reset()
                del ai, data
                gc.collect()

    @unittest.skipIf(test_options.low_mem, "test using >200M")
    def test_OpenCL_CSR(self):
        logger.info("Testing CSR-based algorithm (backward-integration)")
        for devtype in ("GPU", "CPU"):
            ids = ocl.select_device(devtype, best=True)
            if ids is None:
                logger.error("No suitable %s OpenCL device found", devtype)
                continue
            else:
                logger.info("I found a suitable device %s %s: %s %s", devtype, ids, ocl.platforms[ids[0]], ocl.platforms[ids[0]].devices[ids[1]])

            for ds in self.datasets:
                ai = load(ds["poni"])
                data = fabio.open(ds["img"]).data
                ref = ai.integrate1d(data, self.N, method="splitBBox", unit="2th_deg")
                try:
                    res = ai.integrate1d(data, self.N, method="ocl_csr_%i,%i" % (ids[0], ids[1]), unit="2th_deg")
                except (pyopencl.MemoryError, MemoryError, pyopencl.RuntimeError, RuntimeError) as error:
                    logger.warning("Memory error on %s dataset %s: %s%s. Converted into Warning: device may not have enough memory.", devtype, os.path.basename(ds["img"]), os.linesep, error)
                    break
                else:
                    r = mathutil.rwp(ref, res)
                    logger.info("OpenCL CSR vs histogram SplitBBox has R= %.3f for dataset %s", r, ds)
                    self.assertTrue(r < 3, "Rwp=%.3f for OpenCL CSR processing of %s" % (r, ds))
                ai.reset()
                del ai, data
                gc.collect()


class TestSort(unittest.TestCase):
    """
    Test the kernels for vector and image sorting
    """
    N = 1024
    ws = N // 8

    def setUp(self):
        if not test_options.opencl:
            self.skipTest("User request to skip OpenCL tests")
        if pyopencl is None or ocl is None:
            self.skipTest("OpenCL module (pyopencl) is not present or no device available")

        self.h_data = numpy.random.random(self.N).astype("float32")
        self.h2_data = numpy.random.random((self.N, self.N)).astype("float32").reshape((self.N, self.N))

        self.ctx = ocl.create_context(devicetype="GPU")
        device = self.ctx.devices[0]
        try:
            devtype = pyopencl.device_type.to_string(device.type).upper()
        except ValueError:
            # pocl does not describe itself as a CPU !
            devtype = "CPU"
        workgroup = device.max_work_group_size
        if (devtype == "CPU") and (device.platform.vendor == "Apple"):
            logger.info("For Apple's OpenCL on CPU: enforce max_work_goup_size=1")
            workgroup = 1

        self.ws = min(workgroup, self.ws)
        self.queue = pyopencl.CommandQueue(self.ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
        self.local_mem = pyopencl.LocalMemory(self.ws * 32)  # 2float4 = 2*4*4 bytes per workgroup size
        src = read_cl_file("pyfai:openCL/bitonic.cl")
        self.prg = pyopencl.Program(self.ctx, src).build()

    def tearDown(self):
        self.h_data = None
        self.queue = None
        self.ctx = None
        self.local_mem = None
        self.h2_data = None

    def test_reference_book(self):
        d_data = pyopencl.array.to_device(self.queue, self.h_data)
        t0 = time.time()
        hs_data = numpy.sort(self.h_data)
        t1 = time.time()
        time_sort = 1e3 * (t1 - t0)

        evt = self.prg.bsort_book(self.queue, (self.ws,), (self.ws,), d_data.data, self.local_mem)
        evt.wait()
        err = abs(hs_data - d_data.get()).max()
        logger.info("test_reference_book")
        logger.info("Numpy sort on %s element took %s ms", self.N, time_sort)
        logger.info("Reference sort time: %s ms, err=%s ", 1e-6 * (evt.profile.end - evt.profile.start), err)
        # this test works under linux:
        if platform.system() == "Linux":
            self.assertTrue(err == 0.0)
        else:
            logger.warning("Measured error on %s is %s", platform.system(), err)

    def test_reference_file(self):
        d_data = pyopencl.array.to_device(self.queue, self.h_data)
        t0 = time.time()
        hs_data = numpy.sort(self.h_data)
        t1 = time.time()
        time_sort = 1e3 * (t1 - t0)

        evt = self.prg.bsort_file(self.queue, (self.ws,), (self.ws,), d_data.data, self.local_mem)
        evt.wait()
        err = abs(hs_data - d_data.get()).max()
        logger.info("test_reference_file")
        logger.info("Numpy sort on %s element took %s ms", self.N, time_sort)
        logger.info("Reference sort time: %s ms, err=%s", 1e-6 * (evt.profile.end - evt.profile.start), err)
        # this test works anywhere !
        self.assertTrue(err == 0.0)

    def test_sort_all(self):
        d_data = pyopencl.array.to_device(self.queue, self.h_data)
        t0 = time.time()
        hs_data = numpy.sort(self.h_data)
        t1 = time.time()
        time_sort = 1e3 * (t1 - t0)

        evt = self.prg.bsort_all(self.queue, (self.ws,), (self.ws,), d_data.data, self.local_mem)
        evt.wait()
        err = abs(hs_data - d_data.get()).max()
        logger.info("test_sort_all")
        logger.info("Numpy sort on %s element took %s ms", self.N, time_sort)
        logger.info("modified function execution time: %s ms, err=%s", 1e-6 * (evt.profile.end - evt.profile.start), err)
        self.assertTrue(err == 0.0)

    def test_sort_horizontal(self):
        d2_data = pyopencl.array.to_device(self.queue, self.h2_data)
        t0 = time.time()
        h2s_data = numpy.sort(self.h2_data, axis=-1)
        t1 = time.time()
        time_sort = 1e3 * (t1 - t0)
        evt = self.prg.bsort_horizontal(self.queue, (self.N, self.ws), (1, self.ws), d2_data.data, self.local_mem)
        evt.wait()
        err = abs(h2s_data - d2_data.get()).max()
        logger.info("Numpy horizontal sort on %sx%s elements took %s ms", self.N, self.N, time_sort)
        logger.info("Horizontal execution time: %s ms, err=%s", 1e-6 * (evt.profile.end - evt.profile.start), err)
        self.assertTrue(err == 0.0)

    def test_sort_vertical(self):
        d2_data = pyopencl.array.to_device(self.queue, self.h2_data)
        t0 = time.time()
        h2s_data = numpy.sort(self.h2_data, axis=0)
        t1 = time.time()
        time_sort = 1e3 * (t1 - t0)
        evt = self.prg.bsort_vertical(self.queue, (self.ws, self.N), (self.ws, 1), d2_data.data, self.local_mem)
        evt.wait()
        err = abs(h2s_data - d2_data.get()).max()
        logger.info("Numpy vertical sort on %sx%s elements took %s ms", self.N, self.N, time_sort)
        logger.info("Vertical execution time: %s ms, err=%s ", 1e-6 * (evt.profile.end - evt.profile.start), err)
        self.assertTrue(err == 0.0)


class TestKahan(unittest.TestCase):
    """
    Test the kernels for compensated math in OpenCL
    """

    def setUp(self):
        if not test_options.opencl:
            self.skipTest("User request to skip OpenCL tests")
        if pyopencl is None or ocl is None:
            self.skipTest("OpenCL module (pyopencl) is not present or no device available")

        self.ctx = ocl.create_context(devicetype="GPU")
        self.queue = pyopencl.CommandQueue(self.ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)

        # this is running 32 bits OpenCL with POCL
        if (platform.machine() in ("i386", "i686", "x86_64") and (tuple.__itemsize__ == 4) and
                self.ctx.devices[0].platform.name == 'Portable Computing Language'):
            self.args = "-DX87_VOLATILE=volatile"
        else:
            self.args = ""

    def tearDown(self):
        self.queue = None
        self.ctx = None

    @staticmethod
    def dummy_sum(ary, dtype=None):
        "perform the actual sum in a dummy way "
        if dtype is None:
            dtype = ary.dtype.type
        sum_ = dtype(0)
        for i in ary:
            sum_ += i
        return sum_

    def test_kahan(self):
        # simple test
        N = 26
        data = (1 << (N - 1 - numpy.arange(N))).astype(numpy.float32)

        ref64 = numpy.sum(data, dtype=numpy.float64)
        ref32 = self.dummy_sum(data)
        if (ref64 == ref32):
            logger.warning("Kahan: invalid tests as float32 provides the same result as float64")
        # Dummy kernel to evaluate
        src = """
        kernel void summation(global float* data,
                                           int size,
                                    global float* result)
        {
            float2 acc = (float2)(0.0f, 0.0f);
            for (int i=0; i<size; i++)
            {
                acc = kahan_sum(acc, data[i]);
            }
            result[0] = acc.s0;
            result[1] = acc.s1;
        }
        """
        prg = pyopencl.Program(self.ctx, read_cl_file("pyfai:openCL/kahan.cl") + src).build(self.args)
        ones_d = pyopencl.array.to_device(self.queue, data)
        res_d = pyopencl.array.zeros(self.queue, 2, numpy.float32)
        evt = prg.summation(self.queue, (1,), (1,), ones_d.data, numpy.int32(N), res_d.data)
        evt.wait()
        res = res_d.get().sum(dtype=numpy.float64)
        self.assertEqual(ref64, res, "test_kahan")

    def test_dot16(self):
        # simple test
        N = 16
        data = (1 << (N - 1 - numpy.arange(N))).astype(numpy.float32)

        ref64 = numpy.dot(data.astype(numpy.float64), data.astype(numpy.float64))
        ref32 = numpy.dot(data, data)
        if (ref64 == ref32):
            logger.warning("dot16: invalid tests as float32 provides the same result as float64")
        # Dummy kernel to evaluate
        src = """
        kernel void test_dot16(global float* data,
                                           int size,
                               global float* result)
        {
            float2 acc = (float2)(0.0f, 0.0f);
            float16 data16 = (float16) (data[0],data[1],data[2],data[3],data[4],
                                        data[5],data[6],data[7],data[8],data[9],
                         data[10],data[11],data[12],data[13],data[14],data[15]);
            acc = comp_dot16(data16, data16);
            result[0] = acc.s0;
            result[1] = acc.s1;
        }

        kernel void test_dot8(global float* data,
                                           int size,
                               global float* result)
        {
            float2 acc = (float2)(0.0f, 0.0f);
            float8 data0 = (float8) (data[0],data[2],data[4],data[6],data[8],data[10],data[12],data[14]);
            float8 data1 = (float8) (data[1],data[3],data[5],data[7],data[9],data[11],data[13],data[15]);
            acc = comp_dot8(data0, data1);
            result[0] = acc.s0;
            result[1] = acc.s1;
        }

        kernel void test_dot4(global float* data,
                                           int size,
                               global float* result)
        {
            float2 acc = (float2)(0.0f, 0.0f);
            float4 data0 = (float4) (data[0],data[4],data[8],data[12]);
            float4 data1 = (float4) (data[3],data[7],data[11],data[15]);
            acc = comp_dot4(data0, data1);
            result[0] = acc.s0;
            result[1] = acc.s1;
        }

        kernel void test_dot3(global float* data,
                                           int size,
                               global float* result)
        {
            float2 acc = (float2)(0.0f, 0.0f);
            float3 data0 = (float3) (data[0],data[4],data[12]);
            float3 data1 = (float3) (data[3],data[11],data[15]);
            acc = comp_dot3(data0, data1);
            result[0] = acc.s0;
            result[1] = acc.s1;
        }

        kernel void test_dot2(global float* data,
                                           int size,
                               global float* result)
        {
            float2 acc = (float2)(0.0f, 0.0f);
            float2 data0 = (float2) (data[0],data[14]);
            float2 data1 = (float2) (data[1],data[15]);
            acc = comp_dot2(data0, data1);
            result[0] = acc.s0;
            result[1] = acc.s1;
        }

        """

        prg = pyopencl.Program(self.ctx, read_cl_file("pyfai:openCL/kahan.cl") + src).build(self.args)
        ones_d = pyopencl.array.to_device(self.queue, data)
        res_d = pyopencl.array.zeros(self.queue, 2, numpy.float32)
        evt = prg.test_dot16(self.queue, (1,), (1,), ones_d.data, numpy.int32(N), res_d.data)
        evt.wait()
        res = res_d.get().sum(dtype="float64")
        self.assertEqual(ref64, res, "test_dot16")

        res_d.fill(0)
        data0 = data[0::2]
        data1 = data[1::2]
        ref64 = numpy.dot(data0.astype(numpy.float64), data1.astype(numpy.float64))
        ref32 = numpy.dot(data0, data1)
        if (ref64 == ref32):
            logger.warning("dot8: invalid tests as float32 provides the same result as float64")
        evt = prg.test_dot8(self.queue, (1,), (1,), ones_d.data, numpy.int32(N), res_d.data)
        evt.wait()
        res = res_d.get().sum(dtype="float64")
        self.assertEqual(ref64, res, "test_dot8")

        res_d.fill(0)
        data0 = data[0::4]
        data1 = data[3::4]
        ref64 = numpy.dot(data0.astype(numpy.float64), data1.astype(numpy.float64))
        ref32 = numpy.dot(data0, data1)
        if (ref64 == ref32):
            logger.warning("dot4: invalid tests as float32 provides the same result as float64")
        evt = prg.test_dot4(self.queue, (1,), (1,), ones_d.data, numpy.int32(N), res_d.data)
        evt.wait()
        res = res_d.get().sum(dtype="float64")
        self.assertEqual(ref64, res, "test_dot4")

        res_d.fill(0)
        data0 = numpy.array([data[0], data[4], data[12]])
        data1 = numpy.array([data[3], data[11], data[15]])
        ref64 = numpy.dot(data0.astype(numpy.float64), data1.astype(numpy.float64))
        ref32 = numpy.dot(data0, data1)
        if (ref64 == ref32):
            logger.warning("dot3: invalid tests as float32 provides the same result as float64")
        evt = prg.test_dot3(self.queue, (1,), (1,), ones_d.data, numpy.int32(N), res_d.data)
        evt.wait()
        res = res_d.get().sum(dtype="float64")
        self.assertEqual(ref64, res, "test_dot3")

        res_d.fill(0)
        data0 = numpy.array([data[0], data[14]])
        data1 = numpy.array([data[1], data[15]])
        ref64 = numpy.dot(data0.astype(numpy.float64), data1.astype(numpy.float64))
        ref32 = numpy.dot(data0, data1)
        if (ref64 == ref32):
            logger.warning("dot2: invalid tests as float32 provides the same result as float64")
        evt = prg.test_dot2(self.queue, (1,), (1,), ones_d.data, numpy.int32(N), res_d.data)
        evt.wait()
        res = res_d.get().sum(dtype="float64")
        self.assertEqual(ref64, res, "test_dot2")


def suite():
    testsuite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite.addTest(loader(TestMask))
    testsuite.addTest(loader(TestSort))
    testsuite.addTest(loader(TestKahan))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
