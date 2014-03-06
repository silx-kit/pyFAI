#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://forge.epn-campus.eu/projects/azimuthal
#
#    File: "$Id$"
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
__date__ = "20/11/2012"


import unittest
import os
import time
import sys
import fabio
from utilstest import UtilsTest, Rwp, getLogger
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
# = sys.modules["pyFAI.opencl"].ocl

class test_mask(unittest.TestCase):
    tmp_dir = os.environ.get("PYFAI_TEMPDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp"))

    def setUp(self):

        self.datasets = [{"img":UtilsTest.getimage("1883/Pilatus1M.edf"), "poni":UtilsTest.getimage("1893/Pilatus1M.poni"), "spline": None},
            {"img":UtilsTest.getimage("1882/halfccd.edf"), "poni":UtilsTest.getimage("1895/halfccd.poni"), "spline": UtilsTest.getimage("1461/halfccd.spline")},
            {"img":UtilsTest.getimage("1881/Frelon2k.edf"), "poni":UtilsTest.getimage("1896/Frelon2k.poni"), "spline": UtilsTest.getimage("1900/frelon.spline")},
            {"img":UtilsTest.getimage("1884/Pilatus6M.cbf"), "poni":UtilsTest.getimage("1897/Pilatus6M.poni"), "spline": None},
#            {"img":UtilsTest.getimage("1880/Fairchild.edf"), "poni":UtilsTest.getimage("1898/Fairchild.poni"), "spline": None},
            ]
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
        unittest.TestCase.tearDown(self)
        for ds in self.datasets:
            if ds["spline"] is not None:
                if os.path.isfile(ds["poni"]):
                    os.unlink(ds["poni"])

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
                res = ai.xrpd_OpenCL(data, 1000, devicetype="all", platformid=ids[0], deviceid=ids[1], useFp64=True)
                t0 = time.time()
                ref = ai.xrpd(data, 1000)
                t1 = time.time()
                res = ai.xrpd_OpenCL(data, 1000, safe=False)
                t2 = time.time()
                logger.info("For image %15s;\tspeed up is %.3fx;\trate is %.3f Hz" % (os.path.basename(ds["img"]), ((t1 - t0) / (t2 - t1)), 1. / (t2 - t1)))
                r = Rwp(ref, res)
                self.assertTrue(r < 10, "Rwp=%.3f for OpenCL processing of %s" % (r, ds))

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
                try:
                    res = ai.xrpd_LUT_OCL(data, 1000, devicetype="all", platformid=ids[0], deviceid=ids[1])
                except MemoryError as error:
                    logger.warning("Memory Error on %s dataset %s: %s%s. Converted into warnining: device may not have enough memory." % (devtype, os.path.basename(ds["img"]), os.linesep, error))
                else:
                    t0 = time.time()
                    ref = ai.xrpd(data, 1000)
                    t1 = time.time()
                    res = ai.xrpd_LUT_OCL(data, 1000, safe=False)
                    t2 = time.time()
                    logger.info("For image %15s;\tspeed up is %.3fx;\trate is %.3f Hz" % (os.path.basename(ds["img"]), ((t1 - t0) / (t2 - t1)), 1. / (t2 - t1)))
                    r = Rwp(ref, res)
                    self.assertTrue(r < 10, "Rwp=%.3f for OpenCL processing of %s" % (r, ds))


def test_suite_all_OpenCL():
    testSuite = unittest.TestSuite()
    if skip:
        logger.warning("OpenCL module (pyopencl) is not present, skip tests")
    else:
        testSuite.addTest(test_mask("test_OpenCL"))
        testSuite.addTest(test_mask("test_OpenCL_LUT"))

    return testSuite

if __name__ == '__main__':

    mysuite = test_suite_all_OpenCL()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
