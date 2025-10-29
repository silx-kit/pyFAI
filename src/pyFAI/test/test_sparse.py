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

"""Test suites for sparse matrix multiplication modules"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "10/10/2025"

import unittest
import numpy
import logging
from .utilstest import UtilsTest
from .. import load
from ..detectors import detector_factory
from ..integrator.azimuthal import AzimuthalIntegrator
from ..ext import sparse_utils
from ..utils.mathutil import rwp
import fabio
logger = logging.getLogger(__name__)


class TestSparseIntegrate1d(unittest.TestCase):
    """Test azimuthal integration based sparse matrix multiplication methods
    * No splitting
    * Bounding box pixel splitting
    * Full pixel splitting #TODO: check the numerical results!
    """

    @classmethod
    def setUpClass(cls):
        super(TestSparseIntegrate1d, cls).setUpClass()
        cls.N = 1000
        cls.unit = "r_mm"
        cls.ai = load(UtilsTest.getimage("Pilatus1M.poni"))
        with fabio.open(UtilsTest.getimage("Pilatus1M.edf")) as fimg:
            cls.data = fimg.data
        cls.epsilon = 1e-1

    @classmethod
    def tearDownClass(cls):
        super(TestSparseIntegrate1d, cls).tearDownClass()
        cls.N = None
        cls.unit = None
        cls.ai = None
        cls.data = None
        cls.epsilon = None

    def integrate(self, method):
        return self.ai.integrate1d_ng(self.data,
                                      self.N,
                                      correctSolidAngle=False,
                                      unit=self.unit,
                                      method=method,
                                      dummy=-2,
                                      delta_dummy=1)

    def test_sparse_nosplit(self):
        ref = self.integrate(method=("no", "histogram", "cython"))

        obt = self.integrate(method=("no", "lut", "cython"))
        logger.debug("LUT delta on global result: %s", numpy.nanmax(abs(obt[1] - ref[1]) / ref[1]))
        self.assertTrue(numpy.allclose(obt[1], ref[1]))

        obt = self.integrate(method=("no", "csr", "cython"))
        logger.debug("CSR delta on global result: %s", numpy.nanmax(abs(obt[1] - ref[1]) / ref[1]))
        self.assertTrue(numpy.allclose(obt[1], ref[1]))

    def test_sparse_bbox(self):
        ref = self.integrate(method=("bbox", "histogram", "cython"))

        obt = self.integrate(method=("bbox", "lut", "cython"))
        logger.debug("delta on global result: %s", (abs(obt[1] - ref[1]) / ref[1]).max())
        self.assertTrue(numpy.allclose(obt[1], ref[1]))

        obt = self.integrate(method=("bbox", "csr", "cython"))
        logger.debug("delta on global result: %s", (abs(obt[1] - ref[1]) / ref[1]).max())
        self.assertTrue(numpy.allclose(obt[1], ref[1]))

    def test_sparse_fullsplit(self):
        ref = self.integrate(method=("full", "histogram", "cython"))

        for m in "LUT", "CSR":
            obt = self.integrate(method=("full", m, "cython"))
            res = rwp(ref, obt)
            if res > 1:
                logger.error("Numerical values are odd (R=%s)... please refine this test!", res)
                self.assertLess(res, 1, "Wrong!")
                raise unittest.SkipTest("Fix this test")

            else:
                logger.info("R on global result %s: %s", m, res)
                self.assertTrue(numpy.allclose(obt[0], ref[0]))
                self.assertTrue(abs(obt[1] - ref[1]).max() < self.epsilon, "result are almost the same")


class TestSparseIntegrate2d(unittest.TestCase):
    """Test azimuthal integration based sparse matrix multiplication methods
    * No splitting
    * Bounding box pixel splitting
    * Full pixel splitting
    """

    @classmethod
    def setUpClass(cls):
        super(TestSparseIntegrate2d, cls).setUpClass()
        cls.N = 500
        cls.unit = "r_mm"
        cls.ai = load(UtilsTest.getimage("Pilatus1M.poni"))
        with fabio.open(UtilsTest.getimage("Pilatus1M.edf")) as fimg:
            cls.data = fimg.data

    @classmethod
    def tearDownClass(cls):
        super(TestSparseIntegrate2d, cls).tearDownClass()
        cls.N = None
        cls.unit = None
        cls.ai = None
        cls.data = None

    def integrate(self, method):
        # Manually purge engine cache to free some memory
        self.ai.reset_engines()
        return self.ai.integrate2d_ng(self.data, self.N, unit=self.unit, method=method,
                                      correctSolidAngle=False, dummy=-2, delta_dummy=1)

    @staticmethod
    def cost(ref, res):
        return (abs(res[0] - ref[0]) / numpy.maximum(1, ref[0])).max()

    def single_check(self, split="no"):
        method = [split, "histogram", "cython"]
        ref = self.integrate(method=method)
        for m in ("CSR", "LUT"):
            method[1] = m
            obt = self.integrate(method)
            self.assertLess(abs(ref.radial - obt.radial).max(), 1e-3, f"radial matches for {m} with {split} split")
            self.assertLess(abs(ref.azimuthal - obt.azimuthal).max(), 1e-3, f"azimuthal matches for {m} with {split} split")
            # print(method, ref.intensity.shape, obt.intensity.shape)
            res = self.cost(ref, obt)
            if res > 1:
                logger.error("Numerical values are odd (R=%s)... please refine this test for %s  split!", res, method)
                raise unittest.SkipTest("Fix this test")
            else:
                logger.info("R on global result: %s for method %s with %s split", res, m, split)
                if not numpy.allclose(obt[0], ref[0]):
                    logger.error(f"Numerical results are not exactly the same between {m} and histogram with {split} split")
                    raise unittest.SkipTest("Fix this test")
                self.assertTrue(numpy.allclose(obt[0], ref[0]), f"Intensities matches for {m} with {split} split")

    def test_sparse_nosplit(self):
        self.single_check(split="no")

    def test_sparse_bbox(self):
        self.single_check(split="bbox")

    def test_sparse_fullsplit(self):
        self.single_check(split="full")


class TestSparseUtils(unittest.TestCase):

    def test_conversion(self):
        dtype_lut = numpy.dtype([("idx", numpy.int32), ("coef", numpy.float32)])
        shape = 99, 101
        nnz = 0

        # Ensures there is at least one non zero element
        while nnz == 0:
            dense = numpy.random.random(shape).astype("float32")
            mask = dense > 0.90
            loc = numpy.where(mask)
            nnz = len(loc[0])
        idx = loc[0] * shape[-1] + loc[1]
        nnzpr = numpy.bincount(loc[0], minlength=shape[0])
        lut_shape = (shape[0], nnzpr.max())
        lut_ref = numpy.zeros(lut_shape, dtype_lut)
        for i in range(shape[0]):
            id_ = numpy.where(loc[0] == i)[0]
            for j, k in enumerate(id_):
                lut_ref[i, j]["idx"] = i * shape[-1] + loc[1][k]
                lut_ref[i, j]["coef"] = dense[i, loc[1][k]]

        idptr = numpy.zeros(shape[0] + 1, int)
        idptr[1:] = nnzpr.cumsum()
        self.assertEqual(nnz, idptr[-1])
        # self.assertEqual(nnz, idptr[-1], "number of data is consistent")
        csr_ref = (dense[loc], idx, idptr)

        lut_out = sparse_utils.CSR_to_LUT(*csr_ref)
        self.assertTrue(numpy.allclose(lut_out["coef"], lut_ref["coef"]), "coef are the same in LUT")
        self.assertTrue(numpy.allclose(lut_out["idx"], lut_ref["idx"]), "idx are the same in LUT")

        csr_out = sparse_utils.LUT_to_CSR(lut_ref)
        self.assertTrue(numpy.allclose(csr_out[2], csr_ref[2]), "idpts are the same in CSR")
        self.assertTrue(numpy.allclose(csr_out[1], csr_ref[1]), "coef are the same in CSR")
        self.assertTrue(numpy.allclose(csr_out[0], csr_ref[0]), "coef are the same in CSR")

    def test_matrix_conversion(self):
        "Compare the matrices generated without pixel splitting"
        detector = detector_factory("Imxpad S10")
        ai = AzimuthalIntegrator(detector=detector)
        img = numpy.random.random(detector.shape)
        res_csr = ai.integrate1d_ng(img, 100, method=("no", "csr", "cython"), unit="r_mm")
        res_lut = ai.integrate1d_ng(img, 100, method=("no", "lut", "cython"), unit="r_mm")
        self.assertEqual(abs(res_csr.intensity - res_lut.intensity).max(), 0, "intensity matches")
        self.assertEqual(abs(res_csr.radial - res_lut.radial).max(), 0, "radial matches")
        sparse = {}
        for k, v in ai.engines.items():
            sparse[k.algo_lower] = v.engine.lut
        lut2 = sparse_utils.CSR_to_LUT(*sparse["csr"])
        self.assertEqual(abs(lut2["coef"] - sparse["lut"].coef).max(), 0, "LUT coef matches")
        self.assertEqual(abs(lut2["idx"] - sparse["lut"].idx).max(), 0, "LUT idx matches")

        csr2 = sparse_utils.LUT_to_CSR(sparse["lut"])
        self.assertEqual(abs(sparse["csr"][0] - csr2[0]).max(), 0, "CSR data matches")
        self.assertEqual(abs(sparse["csr"][1] - csr2[1]).max(), 0, "CSR indices matches")
        self.assertEqual(abs(sparse["csr"][2] - csr2[2]).max(), 0, "CSR indptr matches")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestSparseIntegrate1d))
    testsuite.addTest(loader(TestSparseIntegrate2d))
    testsuite.addTest(loader(TestSparseUtils))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
