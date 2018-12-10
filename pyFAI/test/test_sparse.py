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

from __future__ import absolute_import, division, print_function

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "06/12/2018"


import unittest
import numpy
import logging
from .utilstest import UtilsTest
logger = logging.getLogger(__name__)
from .. import load
from ..ext import splitBBox
from ..ext import sparse_utils
import fabio


class TestSparseBBox(unittest.TestCase):
    """Test Azimuthal integration based sparse matrix multiplication methods
    Bounding box pixel splitting
    """

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.N = 1000
        self.unit = "2th_deg"
        self.ai = load(UtilsTest.getimage("Pilatus1M.poni"))
        self.data = fabio.open(UtilsTest.getimage("Pilatus1M.edf")).data
        self.ref = self.ai.integrate1d(self.data, self.N, correctSolidAngle=False, unit=self.unit, method="splitBBox")[1]
        self.cython = splitBBox.histoBBox1d(self.data, self.ai.ttha, self.ai._cached_array["2th_delta"], bins=self.N)

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.N = self.unit = self.ai = None
        self.data = self.ref = self.cython = None

    def test_LUT(self):
        obt = self.ai.integrate1d(self.data, self.N, correctSolidAngle=False, unit=self.unit, method="LUT")[1]
        logger.debug("delta on global result: %s", (abs(obt - self.ref) / self.ref).max())
        self.assertTrue(numpy.allclose(obt, self.ref))

        cython = self.ai.engines["lut_integrator"].engine.integrate(self.data)
        for ref, obt in zip(self.cython, cython):
            logger.debug("delta on cython result: %s", (abs(obt - ref) / ref).max())
            self.assertTrue(numpy.allclose(obt, ref))

    def test_CSR(self):
        obt = self.ai.integrate1d(self.data, self.N, correctSolidAngle=False, unit=self.unit, method="bbox CSR")[1]
        logger.debug("delta on global result: %s", (abs(obt - self.ref) / self.ref).max())
        self.assertTrue(numpy.allclose(obt, self.ref))

        cython = self.ai.engines["csr_integrator"].engine.integrate(self.data)
        for ref, obt in zip(self.cython, cython):
            logger.debug("delta on cython result: %s", (abs(obt - ref) / ref).max())
            self.assertTrue(numpy.allclose(obt, ref))


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
