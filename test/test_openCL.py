#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal Integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"test suite for OpenCL code"

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "02/04/2015"


import unittest
import os
import time
import sys
import fabio
import gc
import tempfile
import numpy
import platform
if __name__ == '__main__':
    import pkgutil
    __path__ = pkgutil.extend_path([os.path.dirname(__file__)], "pyFAI.test")
from .utilstest import UtilsTest, Rwp, getLogger, recursive_delete
logger = getLogger(__file__)
try:
    import pyopencl
except ImportError as error:
    logger.warning("OpenCL module (pyopencl) is not present, skip tests. %s." % error)
    skip = True
else:
    skip = False

pyFAI = sys.modules["pyFAI"]
from pyFAI.opencl import ocl
if ocl is None:
    skip = True
else:
    pyopencl = pyFAI.opencl.pyopencl
    import pyopencl.array


class TestMask(unittest.TestCase):
    tmp_dir = tempfile.mkdtemp(prefix="pyFAI_test_OpenCL_")
    N = 1000

    def setUp(self):
        self.datasets = [{"img": UtilsTest.getimage("1883/Pilatus1M.edf"),
                          "poni": UtilsTest.getimage("1893/Pilatus1M.poni"),
                          "spline": None},
                         {"img": UtilsTest.getimage("1882/halfccd.edf"),
                          "poni": UtilsTest.getimage("1895/halfccd.poni"),
                          "spline": UtilsTest.getimage("1461/halfccd.spline")},
                         {"img": UtilsTest.getimage("1881/Frelon2k.edf"),
                          "poni": UtilsTest.getimage("1896/Frelon2k.poni"),
                          "spline": UtilsTest.getimage("1900/frelon.spline")},
                         {"img": UtilsTest.getimage("1884/Pilatus6M.cbf"),
                          "poni": UtilsTest.getimage("1897/Pilatus6M.poni"),
                          "spline": None},
            ]
        if not os.path.isdir(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        for ds in self.datasets:
            if ds["spline"] is not None:
                data = open(ds["poni"], "r").read()
#                spline = os.path.basename(ds["spline"])
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
        recursive_delete(self.tmp_dir)

    def test_OpenCL(self):
        logger.info("Testing histogram-based algorithm (forward-integration)")
        for devtype in ("GPU", "CPU"):
            ids = ocl.select_device(devtype, extensions=["cl_khr_int64_base_atomics"])
            if ids is None:
                logger.error("No suitable %s OpenCL device found" % devtype)
                continue
            else:
                logger.info("I found a suitable device %s %s: %s %s " % (devtype, ids, ocl.platforms[ids[0]], ocl.platforms[ids[0]].devices[ids[1]]))

            for ds in self.datasets:
                ai = pyFAI.load(ds["poni"])
                data = fabio.open(ds["img"]).data
                res = ai.xrpd_OpenCL(data, self.N, devicetype="all", platformid=ids[0], deviceid=ids[1], useFp64=True)
                ref = ai.integrate1d(data, self.N, method="splitBBox", unit="2th_deg")
                r = Rwp(ref, res)
                logger.info("OpenCL histogram vs histogram SplitBBox has R= %.3f for dataset %s" % (r, ds))
                self.assertTrue(r < 6, "Rwp=%.3f for OpenCL histogram processing of %s" % (r, ds))
                del ai, data
                gc.collect()

    def test_OpenCL_LUT(self):
        logger.info("Testing LUT-based algorithm (backward-integration)")
        for devtype in ("GPU", "CPU"):
            ids = ocl.select_device(devtype, best=True)
            if ids is None:
                logger.error("No suitable %s OpenCL device found" % devtype)
                continue
            else:
                logger.info("I found a suitable device %s %s: %s %s " % (devtype, ids, ocl.platforms[ids[0]], ocl.platforms[ids[0]].devices[ids[1]]))

            for ds in self.datasets:
                ai = pyFAI.load(ds["poni"])
                data = fabio.open(ds["img"]).data
                ref = ai.integrate1d(data, self.N, method="splitBBox", unit="2th_deg")
                try:
                    res = ai.integrate1d(data, self.N, method="ocl_lut_%i,%i" % (ids[0], ids[1]), unit="2th_deg")
                except (pyFAI.opencl.pyopencl.MemoryError, MemoryError, pyFAI.opencl.pyopencl.RuntimeError, RuntimeError) as error:
                    logger.warning("Memory error on %s dataset %s: %s%s. Converted into warnining: device may not have enough memory." % (devtype, os.path.basename(ds["img"]), os.linesep, error))
                    break
                else:
                    ref = ai.xrpd(data, self.N)
                    r = Rwp(ref, res)
                    logger.info("OpenCL CSR vs histogram SplitBBox has R= %.3f for dataset %s" % (r, ds))
                    self.assertTrue(r < 3, "Rwp=%.3f for OpenCL LUT processing of %s" % (r, ds))
                del ai, data
                gc.collect()

    def test_OpenCL_CSR(self):
        logger.info("Testing CSR-based algorithm (backward-integration)")
        for devtype in ("GPU", "CPU"):
            ids = ocl.select_device(devtype, best=True)
            if ids is None:
                logger.error("No suitable %s OpenCL device found" % devtype)
                continue
            else:
                logger.info("I found a suitable device %s %s: %s %s " % (devtype, ids, ocl.platforms[ids[0]], ocl.platforms[ids[0]].devices[ids[1]]))

            for ds in self.datasets:
                ai = pyFAI.load(ds["poni"])
                data = fabio.open(ds["img"]).data
                ref = ai.integrate1d(data, self.N, method="splitBBox", unit="2th_deg")
                try:
                    res = ai.integrate1d(data, self.N, method="ocl_csr_%i,%i" % (ids[0], ids[1]), unit="2th_deg")
                except (pyFAI.opencl.pyopencl.MemoryError, MemoryError, pyFAI.opencl.pyopencl.RuntimeError, RuntimeError) as error:
                    logger.warning("Memory error on %s dataset %s: %s%s. Converted into Warning: device may not have enough memory." % (devtype, os.path.basename(ds["img"]), os.linesep, error))
                    break
                else:
                    r = Rwp(ref, res)
                    logger.info("OpenCL CSR vs histogram SplitBBox has R= %.3f for dataset %s" % (r, ds))
                    self.assertTrue(r < 3, "Rwp=%.3f for OpenCL CSR processing of %s" % (r, ds))
                del ai, data
                gc.collect()


class TestSort(unittest.TestCase):
    """
    Test the kernels for vector and image sorting
    """
    N = 1024
    ws = N // 8

    def setUp(self):
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
        src = pyFAI.utils.read_cl_file("bitonic.cl")
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
        logger.info("Numpy sort on %s element took %s ms" % (self.N, time_sort))
        logger.info("Reference sort time: %s ms, err=%s " % (1e-6 * (evt.profile.end - evt.profile.start), err))
        # this test works under linux:
        if platform.system() == "Linux":
            self.assert_(err == 0.0)
        else:
            logger.warning("Measured error on %s is %s" % (platform.system(), err))

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
        logger.info("Numpy sort on %s element took %s ms" % (self.N, time_sort))
        logger.info("Reference sort time: %s ms, err=%s " % (1e-6 * (evt.profile.end - evt.profile.start), err))
        # this test works anywhere !
        self.assert_(err == 0.0)

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
        logger.info("Numpy sort on %s element took %s ms" % (self.N, time_sort))
        logger.info("modified function execution time: %s ms, err=%s " % (1e-6 * (evt.profile.end - evt.profile.start), err))
        self.assert_(err == 0.0)

    def test_sort_horizontal(self):
        d2_data = pyopencl.array.to_device(self.queue, self.h2_data)
        t0 = time.time()
        h2s_data = numpy.sort(self.h2_data, axis=-1)
        t1 = time.time()
        time_sort = 1e3 * (t1 - t0)
        evt = self.prg.bsort_horizontal(self.queue, (self.N, self.ws), (1, self.ws), d2_data.data, self.local_mem)
        evt.wait()
        err = abs(h2s_data - d2_data.get()).max()
        logger.info("Numpy horizontal sort on %sx%s elements took %s ms" % (self.N, self.N, time_sort))
        logger.info("Horizontal execution time: %s ms, err=%s " % (1e-6 * (evt.profile.end - evt.profile.start), err))
        self.assert_(err == 0.0)

    def test_sort_vertical(self):
        d2_data = pyopencl.array.to_device(self.queue, self.h2_data)
        t0 = time.time()
        h2s_data = numpy.sort(self.h2_data, axis=0)
        t1 = time.time()
        time_sort = 1e3 * (t1 - t0)
        evt = self.prg.bsort_vertical(self.queue, (self.ws, self.N), (self.ws, 1), d2_data.data, self.local_mem)
        evt.wait()
        err = abs(h2s_data - d2_data.get()).max()
        logger.info("Numpy vertical sort on %sx%s elements took %s ms" % (self.N, self.N, time_sort))
        logger.info("Vertical execution time: %s ms, err=%s " % (1e-6 * (evt.profile.end - evt.profile.start), err))
        self.assert_(err == 0.0)


def test_suite_all_OpenCL():
    testSuite = unittest.TestSuite()
    if skip:
        logger.warning("OpenCL module (pyopencl) is not present or no device available: skip tests")
    else:
        testSuite.addTest(TestMask("test_OpenCL"))
        testSuite.addTest(TestMask("test_OpenCL_LUT"))
        testSuite.addTest(TestMask("test_OpenCL_CSR"))
        testSuite.addTest(TestSort("test_reference_book"))
        testSuite.addTest(TestSort("test_reference_file"))
        testSuite.addTest(TestSort("test_sort_all"))
        testSuite.addTest(TestSort("test_sort_horizontal"))
        testSuite.addTest(TestSort("test_sort_vertical"))
    return testSuite

if __name__ == '__main__':

    mysuite = test_suite_all_OpenCL()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
