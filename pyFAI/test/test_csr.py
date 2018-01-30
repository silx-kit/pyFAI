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

"""tests for Jon's geometry changes
FIXME : make some tests that the functions do what is expected
"""

from __future__ import absolute_import, division, print_function


import unittest
import numpy
import logging
from . import utilstest
logger = logging.getLogger(__name__)
from .. import opencl
from ..ext import splitBBox
from ..ext import splitBBoxCSR
from ..azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.utils.decorators import depreclog
if opencl.ocl:
    from ..opencl import azim_csr as ocl_azim_csr

import fabio


class TestOpenClCSR(utilstest.ParametricTestCase):

    @classmethod
    def setUpClass(cls):
        cls.N = 1000
        cls.ai = AzimuthalIntegrator.sload(utilstest.UtilsTest.getimage("Pilatus1M.poni"))
        cls.data = fabio.open(utilstest.UtilsTest.getimage("Pilatus1M.edf")).data
        with utilstest.TestLogging(logger=depreclog, warning=1):
            # Filter deprecated warning
            cls.ai.xrpd_LUT(cls.data, cls.N)

    @classmethod
    def tearDownClass(cls):
        cls.N = None
        cls.ai = None
        cls.data = None

    def setUp(self):
        if not utilstest.UtilsTest.opencl:
            self.skipTest("User request to skip OpenCL tests")
        if not opencl.ocl:
            self.skipTest("OpenCL not available or skiped")

    def test_csr(self):
        testcases = [8 * 2 ** i for i in range(6)]  # [8, 16, 32, 64, 128, 256]
        for workgroup_size in testcases:
            with self.subTest(workgroup_size=workgroup_size):
                out_ref = splitBBox.histoBBox1d(self.data, self.ai.ttha, self.ai._cached_array["2th_delta"], bins=self.N)
                csr = splitBBoxCSR.HistoBBox1d(self.ai.ttha, self.ai._cached_array["2th_delta"], bins=self.N, unit="2th_deg")
                if not opencl.ocl:
                    skip = True
                else:
                    try:
                        ocl_csr = ocl_azim_csr.OCL_CSR_Integrator(csr.lut, self.data.size, "ALL", profile=True, block_size=workgroup_size)
                        out_ocl_csr = ocl_csr.integrate(self.data)
                    except (opencl.pyopencl.MemoryError, MemoryError):
                        logger.warning("Skipping test due to memory error on device")
                        skip = True
                    else:
                        skip = False
                out_cyt_csr = csr.integrate(self.data)
                cmt = "Testing ocl_csr with workgroup_size= %s" % (workgroup_size)
                logger.debug(cmt)
                if skip:
                    for ref, cyth in zip(out_ref, out_cyt_csr):
                        self.assertTrue(numpy.allclose(ref, cyth), cmt + ": hist vs csr")
                else:
                    for ref, ocl, cyth in zip(out_ref[1:], out_ocl_csr, out_cyt_csr[1:]):
                        self.assertTrue(numpy.allclose(ref, ocl), cmt + ": hist vs ocl_csr")
                        self.assertTrue(numpy.allclose(ref, cyth), cmt + ": hist vs csr")
                        self.assertTrue(numpy.allclose(cyth, ocl), cmt + ": csr vs ocl_csr")
                csr = None
                ocl_csr = None
                out_ocl_csr = None
                out_ref = None


class TestCSR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.N = 1000
        cls.ai = AzimuthalIntegrator.sload(utilstest.UtilsTest.getimage("Pilatus1M.poni"))
        cls.data = fabio.open(utilstest.UtilsTest.getimage("Pilatus1M.edf")).data
        with utilstest.TestLogging(logger=depreclog, warning=1):
            # Filter deprecated warning
            cls.ai.xrpd_LUT(cls.data, cls.N)

    @classmethod
    def tearDownClass(cls):
        cls.N = None
        cls.ai = None
        cls.data = None

    def test_2d_splitbbox(self):
        self.ai.reset()
        img, tth, chi = self.ai.integrate2d(self.data, self.N, unit="2th_deg", method="splitbbox_LUT")
        img_csr, tth_csr, chi_csr = self.ai.integrate2d(self.data, self.N, unit="2th_deg", method="splitbbox_csr")
        self.assertTrue(numpy.allclose(tth, tth_csr), " 2Th are the same")
        self.assertTrue(numpy.allclose(chi, chi_csr), " Chi are the same")
        # TODO: align on splitbbox rather then splitbbox_csr
        # diff_img(img, img_csr, "splitbbox")
        self.assertTrue(numpy.allclose(img, img_csr), "img are the same")

    def test_2d_nosplit(self):
        self.ai.reset()
        result_histo = self.ai.integrate2d(self.data, self.N, unit="2th_deg", method="histogram")
        result_nosplit = self.ai.integrate2d(self.data, self.N, unit="2th_deg", method="nosplit_csr")
        # diff_crv(tth, tth_csr, "2th")
        # self.assertTrue(numpy.allclose(tth, tth_csr), " 2Th are the same")
        # self.assertTrue(numpy.allclose(chi, chi_csr), " Chi are the same")
        # diff_img(img, img_csr, "no split")
        error = ((result_histo.intensity - result_nosplit.intensity) > 1).sum()
        self.assertLess(error, 6, "img are almost the same")


def suite():
    testsuite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite.addTest(loader(TestCSR))
    testsuite.addTest(loader(TestOpenClCSR))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
