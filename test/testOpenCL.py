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
__date__ = "28/06/2012"


import unittest
import os
import time
import sys
import tempfile
import fabio
from utilstest import UtilsTest, Rwp, getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]
ocl = None
try:
    from pyFAI.ocl_azim import Integrator1d
except ImportError as error:
    logger.warning("OpenCL module (pyFAI.ocl_azim) is not present, skip tests. %s." % error)
    ocl = None
else:
    ocl = pyFAI.ocl_azim.OpenCL()

class test_mask(unittest.TestCase):

    def setUp(self):
        self.datasets = [{"img":UtilsTest.getimage("1883/Pilatus1M.edf"), "poni":UtilsTest.getimage("1893/Pilatus1M.poni"), "spline": None},
            {"img":UtilsTest.getimage("1882/halfccd.edf"), "poni":UtilsTest.getimage("1895/halfccd.poni"), "spline": UtilsTest.getimage("1461/halfccd.spline")},
            {"img":UtilsTest.getimage("1881/Frelon2k.edf"), "poni":UtilsTest.getimage("1896/Frelon2k.poni"), "spline": UtilsTest.getimage("1900/frelon.spline")},
            {"img":UtilsTest.getimage("1884/Pilatus6M.cbf"), "poni":UtilsTest.getimage("1897/Pilatus6M.poni"), "spline": None},
            {"img":UtilsTest.getimage("1880/Fairchild.edf"), "poni":UtilsTest.getimage("1898/Fairchild.poni"), "spline": None},
            ]
        for ds in self.datasets:
            if ds["spline"] is not None:
                data = open(ds["poni"], "r").read()
                spline = os.path.basename(ds["spline"])
                open(ds["poni"], "w").write(data.replace(" " + spline, " " + ds["spline"]))

    def test_OpenCL(self):
        ocl = pyFAI.ocl_azim.OpenCL()
        ids = ocl.select_device(extensions=["cl_khr_int64_base_atomics"])
        if ids is None:
            logger.error("No suitable OpenCL device found")
            return
        else:
            logger.info("I found a suitable device %s: %s %s " % (ids, ocl.platforms[ids[0]], ocl.platforms[ids[0]].devices[ids[1]]))

        for ds in self.datasets:
            ai = pyFAI.load(ds["poni"])
            data = fabio.open(ds["img"]).data
            ocl = ai.xrpd_OpenCL(data, 1000, devicetype="all", platformid=ids[0], deviceid=ids[1], useFp64=True)
            t0 = time.time()
            ref = ai.xrpd(data, 1000)
            t1 = time.time()
            ocl = ai.xrpd_OpenCL(data, 1000, safe=False)
            t2 = time.time()
            logger.info("For image %15s;\tspeed up is %.3fx;\trate is %.3f Hz" % (os.path.basename(ds["img"]), ((t1 - t0) / (t2 - t1)), 1. / (t2 - t1)))
            r = Rwp(ref, ocl)
            self.assertTrue(r < 6, "Rwp=%.3f for OpenCL processing of %s" % (r, ds))

def test_suite_all_OpenCL():
    testSuite = unittest.TestSuite()
    if ocl:
        testSuite.addTest(test_mask("test_OpenCL"))
    else:
        logger.warning("OpenCL module (pyFAI.ocl_azim) is not present, skip tests")
    return testSuite

if __name__ == '__main__':

    mysuite = test_suite_all_OpenCL()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
