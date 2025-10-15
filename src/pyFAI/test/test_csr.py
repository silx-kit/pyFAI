#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2025 European Synchrotron Radiation Facility, Grenoble, France
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
"""
import sys
import platform
import unittest
import numpy
import logging
from . import utilstest
from .utilstest import UtilsTest
from .. import opencl
from ..containers import Integrate2dtpl
from ..ext import splitBBox
from ..ext import splitBBoxCSR
from ..engines.CSR_engine import CsrIntegrator2d, CsrIntegrator1d
from ..method_registry import IntegrationMethod
from ..containers import ErrorModel
if opencl.ocl:
    from ..opencl import azim_csr as ocl_azim_csr
logger = logging.getLogger(__name__)


class TestCSR(utilstest.ParametricTestCase):

    @classmethod
    def setUpClass(cls):
        cls.N = 800
        cls.data, cls.ai = utilstest.create_fake_data(poissonian=False)
        # Force the initialization of all caches
        cls.ai.delta_array(unit="2th_deg")

    @classmethod
    def tearDownClass(cls):
        cls.N = None
        cls.ai = None
        cls.data = None

    @unittest.skipIf((utilstest.UtilsTest.opencl is None) or (opencl.ocl is None), "Test on OpenCL disabled")
    def test_opencl_csr(self):
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
                        out_ocl_csr = ocl_csr.integrate_legacy(self.data)
                    except (opencl.pyopencl.MemoryError, MemoryError):
                        logger.warning("Skipping test due to memory error on device")
                        skip = True
                    else:
                        skip = False
                out_cyt_csr = csr.integrate_legacy(self.data)
                cmt = "Testing ocl_csr with workgroup_size= %s" % (workgroup_size)
                logger.debug(cmt)
                if skip:
                    for ref, cyth in zip(out_ref, out_cyt_csr):
                        self.assertTrue(numpy.allclose(ref, cyth), cmt + ": hist vs csr")
                else:
                    for ref, ocl, cyth in zip(out_ref[1:], out_ocl_csr, out_cyt_csr[1:]):
                        logger.debug("hist vs ocl_csr %s; hist vs csr: %s; csr vs ocl_csr: %s",
                                     abs(ref - ocl).max(),
                                     abs(ref - cyth).max(),
                                     abs(cyth - ocl).max())
                        self.assertTrue(numpy.allclose(ref, ocl), cmt + ": hist vs ocl_csr")
                        self.assertTrue(numpy.allclose(ref, cyth), cmt + ": hist vs csr")
                        self.assertTrue(numpy.allclose(cyth, ocl), cmt + ": csr vs ocl_csr")
                csr = None
                ocl_csr = None
                out_ocl_csr = None
                out_ref = None

    def test_1d_splitbbox(self):
        self.ai.reset()
        tth, img = self.ai.integrate1d_ng(self.data, self.N, unit="2th_deg", method=("bbox", "histogram", "cython"))
        res = self.ai.integrate1d_ng(self.data, self.N, unit="2th_deg", method=("bbox", "csr", "cython"))
        tth_csr, img_csr = res
        self.assertTrue(numpy.allclose(tth, tth_csr), "2Th are the same")
        error = (img - img_csr)
        logger.debug("ref: %s; obt: %s", img.shape, img_csr.shape)
        logger.debug("error mean: %s, std: %s", error.mean(), error.std())
        self.assertLess(error.mean(), 0.1, "img are almost the same")
        self.assertLess(error.std(), 3, "img are almost the same")

        # Validate the scipy integrator ....
        engine = self.ai.engines[res.method].engine
        scipy_engine = CsrIntegrator1d(self.data.size,
                                       lut=(engine.data, engine.indices, engine.indptr),
                                       empty=0.0,
                                       bin_centers=engine.bin_centers)

        res_csr = engine.integrate(self.data)
        res_scipy = scipy_engine.integrate(self.data)

        self.assertTrue(numpy.allclose(res_csr.position, res_scipy.position), "pos0 is the same")
        self.assertTrue(numpy.allclose(res_csr.count, res_scipy.count), "count is almost the same")
        self.assertTrue(numpy.allclose(res_csr.normalization, res_scipy.normalization), "count is same as normalization")
        self.assertTrue(numpy.allclose(res_csr.signal, res_scipy.signal), "sum_data is almost the same")

    def test_1d_nosplit(self):
        self.ai.reset()
        result_histo = self.ai.integrate1d_ng(self.data, self.N, unit="2th_deg", method=("no", "histogram", "cython"))
        result_nosplit = self.ai.integrate1d_ng(self.data, self.N, unit="2th_deg", method=("no", "csr", "cython"))
        self.assertTrue(numpy.allclose(result_histo.radial, result_nosplit.radial), "2Th are the same")
        error = (result_histo.intensity - result_nosplit.intensity)
        logger.debug("ref: %s; obt: %s", result_histo.intensity.shape, result_nosplit.intensity.shape)
        logger.debug("error mean: %s, std: %s", error.mean(), error.std())
        self.assertLess(error.mean(), 1e-3, "img are almost the same")
        self.assertLess(error.std(), 3, "img are almost the same")

        # Validate the scipy integrator ....
        engine = self.ai.engines[result_nosplit.method].engine
        scipy_engine = CsrIntegrator1d(self.data.size,
                                       lut=(engine.data, engine.indices, engine.indptr),
                                       empty=0.0,
                                       bin_centers=engine.bin_centers)

        res_csr = engine.integrate(self.data)
        res_scipy = scipy_engine.integrate(self.data)

        self.assertTrue(numpy.allclose(res_csr.position, res_scipy.position), "pos0 is the same")
        self.assertTrue(numpy.allclose(res_csr.count, res_scipy.count), "count is almost the same")
        self.assertTrue(numpy.allclose(res_csr.normalization, res_scipy.normalization), "count is same as normalization")
        self.assertTrue(numpy.allclose(res_csr.signal, res_scipy.signal), "sum_data is almost the same")

    def test_2d_splitbbox(self):
        self.ai.reset()
        img, tth, chi = self.ai.integrate2d(self.data, self.N, unit="2th_deg", method="splitbbox")
        img_csr, tth_csr, chi_csr = self.ai.integrate2d(self.data, self.N, unit="2th_deg", method="csr")
        self.assertTrue(numpy.allclose(tth, tth_csr), " 2Th are the same")
        self.assertTrue(numpy.allclose(chi, chi_csr), " Chi are the same")
        error = (img - img_csr)
        logger.debug("ref: %s; obt: %s", img.shape, img_csr.shape)
        logger.debug("error mean: %s, std: %s", error.mean(), error.std())
        self.assertLess(error.mean(), 0.1, "img are almost the same")
        self.assertLess(error.std(), 3, "img are almost the same")

        # Validate the scipy integrator ....
        method = IntegrationMethod.select_method(2, split="bbox", algo="CSR", impl="cython")[0]
        engine = self.ai.engines[method].engine
        scipy_engine = CsrIntegrator2d(self.data.size,
                                       lut=(engine.data, engine.indices, engine.indptr),
                                       empty=0.0,
                                       bin_centers0=engine.bin_centers0,
                                       bin_centers1=engine.bin_centers1)

        res_csr = engine.integrate_ng(self.data, error_model=ErrorModel(2))
        res_scipy = scipy_engine.integrate(self.data, error_model=ErrorModel(2))

        self.assertTrue(numpy.allclose(res_csr.radial, res_scipy.radial), "pos0 is the same")
        self.assertTrue(numpy.allclose(res_csr.azimuthal, res_scipy.azimuthal), "pos1 is the same")
        self.assertTrue(numpy.allclose(res_csr.count, res_scipy.count), "count is almost the same")
        self.assertTrue(numpy.allclose(res_csr.normalization, res_scipy.normalization), "count is same as normalization")
        self.assertTrue(numpy.allclose(res_csr.signal, res_scipy.signal), "sum_data is almost the same")
        # ensure shapes  are all consistent
        for key in Integrate2dtpl._fields:
            self.assertEqual(res_csr.__getattribute__(key).shape, res_scipy.__getattribute__(key).shape, f"shape matches for {key}")

    @unittest.skipIf(UtilsTest.TEST_IS32_BIT or (any(platform.win32_ver()) and sys.version_info[:2] == (3, 9)),
                                                 "test unreliable on 32bits processor / Windows+Python3.9")
    def test_2d_nosplit(self):
        self.ai.reset()
        result_histo = self.ai.integrate2d(self.data, self.N, unit="2th_deg", method=("no", "histogram", "cython"))
        result_nosplit = self.ai.integrate2d(self.data, self.N, unit="2th_deg", method=("no", "csr", "cython"))
        self.assertTrue(numpy.allclose(result_histo.radial, result_nosplit.radial), " 2Th are the same")
        self.assertTrue(numpy.allclose(result_histo.azimuthal, result_nosplit.azimuthal, atol=1e-5), " Chi are the same")
        # print("result_histo", result_histo.method, result_histo.method_called)
        # print("result_histo", result_histo.sum_signal.min(), result_histo.sum_signal.max(), result_histo.sum_signal.mean(), result_histo.sum_signal.std())
        # print("result_histo",result_histo.sum_normalization.min(), result_histo.sum_normalization.max(), result_histo.sum_normalization.mean(), result_histo.sum_normalization.std())
        # print("result_histo",result_histo.intensity.min(), result_histo.intensity.max(), result_histo.intensity.mean(), result_histo.intensity.std())
        # print("result_nosplit",result_nosplit.intensity.min(), result_nosplit.intensity.max(), result_nosplit.intensity.mean(), result_nosplit.intensity.std())
        # print("result_nosplit",result_nosplit.method)
        error = (result_histo.intensity - result_nosplit.intensity)
        logger.debug("ref: %s; obt: %s", result_histo.intensity.shape, result_nosplit.intensity.shape)
        logger.debug("error mean: %s, std: %s", error.mean(), error.std())

        self.assertLess(error.mean(), 1e-3, "img are almost the same")
        self.assertLess(error.std(), 3, "img are almost the same")

        # Validate the scipy integrator ....
        method = IntegrationMethod.select_method(2, split="no", algo="CSR", impl="cython")[0]

        engine = self.ai.engines[method].engine
        scipy_engine = CsrIntegrator2d(self.data.size,
                                       lut=(engine.data, engine.indices, engine.indptr),
                                       empty=0.0,
                                       bin_centers0=engine.bin_centers0,
                                       bin_centers1=engine.bin_centers1)

        res_csr = engine.integrate(self.data, error_model=ErrorModel(2))
        res_scipy = scipy_engine.integrate(self.data, error_model=ErrorModel(2))

        self.assertTrue(numpy.allclose(res_csr.radial, res_scipy.radial), "pos0 is the same")
        self.assertTrue(numpy.allclose(res_csr.azimuthal, res_scipy.azimuthal), "pos1 is the same")
        self.assertTrue(numpy.allclose(res_csr.count, res_scipy.count), "count is almost the same")
        self.assertTrue(numpy.allclose(res_csr.normalization, res_scipy.normalization), "count is same as normalization")
        self.assertTrue(numpy.allclose(res_csr.signal, res_scipy.signal), "sum_data is almost the same")
        # ensure shapes  are all consistent
        for key in Integrate2dtpl._fields:
            self.assertEqual(res_csr.__getattribute__(key).shape, res_scipy.__getattribute__(key).shape, f"shape matches for {key}")


def suite():
    testsuite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite.addTest(loader(TestCSR))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
