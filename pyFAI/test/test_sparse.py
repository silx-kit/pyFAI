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

"""Test suites for sparse matrix multiplication modules"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "15/01/2021"

import unittest
import numpy
import logging
from .utilstest import UtilsTest
logger = logging.getLogger(__name__)
from .. import load
from ..detectors import detector_factory
from ..azimuthalIntegrator import AzimuthalIntegrator
from ..ext import splitBBox
from ..ext import sparse_utils
import fabio


class TestSparseBBox(unittest.TestCase):
    """Test Azimuthal integration based sparse matrix multiplication methods
    Bounding box pixel splitting
    """

    def test_sparse_bbox(self):
        N = 1000
        unit = "r_mm"
        method = ("bbox", "lut", "cython")
        ai = load(UtilsTest.getimage("Pilatus1M.poni"))
        data = fabio.open(UtilsTest.getimage("Pilatus1M.edf")).data
        ref = ai.integrate1d_ng(data, N, correctSolidAngle=False, unit=unit, method=method)

        method = ("bbox", "lut", "cython")
        obt = ai.integrate1d_ng(data, N, correctSolidAngle=False, unit=unit, method=method)
        logger.debug("delta on global result: %s", (abs(obt[1] - ref[1]) / ref[1]).max())
        self.assertTrue(numpy.allclose(obt[1], ref[1]))

        method = ("bbox", "csr", "cython")
        obt = ai.integrate1d_ng(data, N, correctSolidAngle=False, unit=unit, method=method)
        logger.debug("delta on global result: %s", (abs(obt[1] - ref[1]) / ref[1]).max())
        self.assertTrue(numpy.allclose(obt[1], ref[1]))


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
        assert nnz == idptr[-1]
        # self.assertEqual(nnz, idptr[-1], "number of data is consitent")
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
        detector = detector_factory("Pilatus100k")
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


class TestContainer(unittest.TestCase):

    def test_vector(self):
        nelem = 12
        cont = sparse_utils.Vector()
        self.assertEqual(cont.size, 0, "initialized vector is empty")
        self.assertGreaterEqual(cont.allocated, 4, "Initialized vector has some space")
        for i in range(nelem):
            cont.append(i, 0.1 * i)
        self.assertEqual(cont.size, nelem, "initialized vector is empty")
        self.assertGreaterEqual(cont.allocated, nelem, "Initialized vector has some space")
        i, f = cont.get_data()
        d = abs(numpy.arange(nelem) - i).max()
        self.assertEqual(d, 0, "result is OK")
        self.assertLess(abs(f - i / 10.0).max(), 1e-6, "result is OK")

    def test_container(self):
        nlines = 11
        ncol = 12
        nelem = nlines * ncol
        cont = sparse_utils.ArrayBuilder(nlines)
        for i in range(nelem):
            cont.append(i % nlines, i, 0.1 * i)
        s = numpy.arange(nelem).reshape((ncol, nlines)).T.ravel()
        a, b, c = cont.as_CSR()
        err = abs(a - numpy.arange(0, nelem + 1, ncol)).max()
        self.assertEqual(err, 0, "idxptr OK")
        err = abs(s - b).max()
        self.assertEqual(err, 0, "idx OK")
        err = abs((s / 10.0) - c).max()
        self.assertLessEqual(err, 1e-6, "value OK: %s" % err)
        lut = cont.as_LUT()
        s = numpy.arange(nelem).reshape((ncol, nlines)).T
        self.assertEqual(abs(lut["idx"] - s).max(), 0, "LUT idx OK")
        self.assertLessEqual(abs(lut["coef"] - s / 10.0).max(), 1e-6, "LUT coef OK")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestSparseBBox))
    testsuite.addTest(loader(TestSparseUtils))
    testsuite.addTest(loader(TestContainer))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
