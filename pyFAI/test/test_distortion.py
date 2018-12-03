#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2013-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  .
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  .
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

"test suite for Distortion correction class"
from __future__ import absolute_import, division, print_function


__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "30/10/2018"


import unittest
import numpy
import fabio
import logging
from .utilstest import UtilsTest
logger = logging.getLogger(__name__)
from .. import detectors
from .. import distortion
from ..ext import _distortion
from ..ext import sparse_utils


class TestHalfCCD(unittest.TestCase):
    """basic test"""
    halfFrelon = "LaB6_0020.edf"
    splineFile = "halfccd.spline"
    fit2d_cor = "halfccd.fit2d.edf"

    @classmethod
    def setUpClass(cls):
        super(TestHalfCCD, cls).setUpClass()
        """Download files"""
        cls.fit2dFile = UtilsTest.getimage(cls.fit2d_cor)
        cls.halfFrelon = UtilsTest.getimage(cls.halfFrelon)
        cls.splineFile = UtilsTest.getimage(cls.splineFile)
        cls.det = detectors.FReLoN(cls.splineFile)
        cls.fit2d = fabio.open(cls.fit2dFile).data
        cls.ref = _distortion.Distortion(cls.det)
        cls.raw = fabio.open(cls.halfFrelon).data
        cls.dis = distortion.Distortion(cls.det, method="LUT")
        cls.larger = numpy.zeros(cls.det.shape)
        cls.larger[:-1, :] = cls.raw
        cls.preproc = numpy.zeros(cls.raw.shape + (3,))
        cls.preproc[:, :, 0] = cls.raw
        cls.preproc[:, :, 1] = cls.raw  # assume poissonian noise
        cls.preproc[:, :, 2] = 1

    @classmethod
    def tearDownClass(cls):
        super(TestHalfCCD, cls).tearDownClass()
        cls.larger = cls.fit2dFile = cls.halfFrelon = cls.splineFile = None
        cls.preproc = cls.det = cls.dis = cls.fit2d = cls.raw = cls.ref = None

    @unittest.skipIf(UtilsTest.low_mem, "skipping test using >100M")
    def test_pos_lut(self):
        """
        Compare position from _distortion.Distortion and distortion.Distortion.
        Nota the points, named ABCD have a different layout in those implementations:
        _distortion.Distortion:  B C   distortion.Distortion: D C
                                 A B                          A B
        So we compare only the position of A and C.

        """
        self.dis.reset(prepare=False)
        onp = self.dis.calc_pos(use_cython=False)[:, :, ::2, :]
        self.assertEqual(self.dis.delta1, 3)
        self.assertEqual(self.dis.delta2, 3)

        self.dis.reset(prepare=False)
        ocy = self.dis.calc_pos(use_cython=True)[:, :, ::2, :]
        ref = self.ref.calc_pos()[:, :, ::2, :]
        self.assertEqual(abs(onp - ocy).max(), 0, "Numpy and cython implementation are equivalent")
        self.assertLess(abs(ocy - ref).max(), 1e-3,
                        "equivalence of the _distortion and distortion Distortion classes at 1 per 1000 of a pixel")
        self.assertEqual(self.dis.delta1, 3)
        self.assertEqual(self.dis.delta2, 3)
        self.assertEqual(self.ref.delta0, 3)
        self.assertEqual(self.ref.delta1, 3)

        self.dis.calc_LUT(False)
        self.ref.calc_LUT()
        delta = (self.dis.lut["idx"] - self.ref.LUT["idx"])
        bad = 1.0 * self.dis.lut.size / (delta == 0).sum() - 1
        self.assertLess(bad, 1e-2,
                        "same index position < 1%% error, got %s" % bad)
        ref_pixel_size = self.ref.LUT["coef"].sum(axis=-1)
        obt_pixel_size = self.dis.lut["coef"].sum(axis=-1)
        delta = abs(ref_pixel_size - obt_pixel_size).max()
        self.assertLess(delta, 1e-3,
                        "Same pixel size at 0.1%%, got %s" % delta)

    def test_ref_vs_fit2d(self):
        """Compare reference spline correction vs fit2d's code

        precision at 1e-3 : 90% of pixels
        """
        # self.dis.reset(method="lut", prepare=False)
        try:
            self.ref.calc_LUT()
        except MemoryError as error:
            logger.warning("TestHalfCCD.test_ref_vs_fit2d failed because of MemoryError. This test tries to allocate a lot of memory and failed with %s", error)
            return
        cor = self.ref.correct(self.raw)
        delta = abs(cor - self.fit2d)
        logger.info("Delta max: %s mean: %s", delta.max(), delta.mean())
        mask = numpy.where(self.fit2d == 0)
        denom = self.fit2d.copy()
        denom[mask] = 1
        ratio = delta / denom
        ratio[mask] = 0
        good_points_ratio = 1.0 * (ratio < 1e-3).sum() / self.raw.size
        logger.info("ratio of good points (less than 1/1000 relative error): %.4f", good_points_ratio)
        self.assertTrue(good_points_ratio > 0.99, "99% of all points have a relative error below 1/1000")

    def test_lut_vs_fit2d(self):
        """Compare reference spline correction vs fit2d's code

        precision at 1e-3 : 90% of pixels
        """
        self.dis.reset(method="lut", prepare=False)
        self.dis.empty = 0.0
        try:
            self.dis.calc_LUT()
        except MemoryError as error:
            logger.warning("TestHalfCCD.test_ref_vs_fit2d failed because of MemoryError. This test tries to allocate a lot of memory and failed with %s", error)
            return
        cor = self.dis.correct(self.raw)[:-1, :]
        delta = abs(cor - self.fit2d)
        logger.info("Delta max: %s mean: %s", delta.max(), delta.mean())
        mask = numpy.where(self.fit2d == 0)
        denom = self.fit2d.copy()
        denom[mask] = 1
        ratio = delta / denom
        ratio[mask] = 0
        good_points_ratio = 1.0 * (ratio < 1e-3).sum() / self.raw.size
        logger.info("ratio of good points (less than 1/1000 relative error): %.4f", good_points_ratio)
        self.assertTrue(good_points_ratio > 0.99, "99% of all points have a relative error below 1/1000")

        a, b, c = self.dis.correct(self.preproc)
        cor = c[:-1, :, 0]
        error = b[:-1, :]
        delta = abs(cor - self.fit2d)
        logger.info("Delta max: %s mean: %s", delta.max(), delta.mean())
        mask = numpy.where(self.fit2d == 0)
        denom = self.fit2d.copy()
        denom[mask] = 1
        ratio = delta / denom
        ratio[mask] = 0
        good_points_ratio = 1.0 * (ratio < 1e-3).sum() / self.raw.size
        logger.info("ratio of good points (less than 1/1000 relative error): %.4f", good_points_ratio)
        self.assertTrue(good_points_ratio > 0.99, "99% of all points have a relative error below 1/1000")
        self.assertTrue(numpy.alltrue(a >= b), "signal is greater then error")
        self.assertTrue(numpy.alltrue(b >= 0), "error is positive")
        self.assertTrue(numpy.any(b > 0), "error is not null")
        self.assertTrue(numpy.alltrue(c >= 0), "propagated array is positive")
        self.assertTrue(numpy.any(c > 0), "propagated array is not null")

    def test_csr_vs_fit2d(self):
        """Compare reference spline correction vs fit2d's code

        precision at 1e-3 : 90% of pixels
        """
        self.dis.reset(method="csr", prepare=False)
        try:
            self.dis.calc_LUT()
        except MemoryError as error:
            logger.warning("TestHalfCCD.test_ref_vs_fit2d failed because of MemoryError. This test tries to allocate a lot of memory and failed with %s", error)
            return
        cor = self.dis.correct(self.raw)[:-1, :]
        delta = abs(cor - self.fit2d)
        logger.info("Delta max: %s mean: %s", delta.max(), delta.mean())
        mask = numpy.where(self.fit2d == 0)
        denom = self.fit2d.copy()
        denom[mask] = 1
        ratio = delta / denom
        ratio[mask] = 0
        good_points_ratio = 1.0 * (ratio < 1e-3).sum() / self.raw.size
        logger.info("ratio of good points (less than 1/1000 relative error): %.4f", good_points_ratio)
        self.assertTrue(good_points_ratio > 0.99, "99% of all points have a relative error below 1/1000")

        # Now test with error propagation
        a, b, c = self.dis.correct(self.preproc)
        cor = c[:-1, :, 0]
        error = b[:-1, :]
        delta = abs(cor - self.fit2d)
        logger.info("Delta max: %s mean: %s", delta.max(), delta.mean())
        mask = numpy.where(self.fit2d == 0)
        denom = self.fit2d.copy()
        denom[mask] = 1
        ratio = delta / denom
        ratio[mask] = 0
        good_points_ratio = 1.0 * (ratio < 1e-3).sum() / self.raw.size
        logger.info("ratio of good points (less than 1/1000 relative error): %.4f", good_points_ratio)
        self.assertTrue(good_points_ratio > 0.99, "99% of all points have a relative error below 1/1000")
        self.assertTrue(numpy.alltrue(a >= b), "signal is greater then error")
        self.assertTrue(numpy.alltrue(b >= 0), "error is positive")
        self.assertTrue(numpy.any(b > 0), "error is not null")
        self.assertTrue(numpy.alltrue(c >= 0), "propagated array is positive")
        self.assertTrue(numpy.any(c > 0), "propagated array is not null")


class TestImplementations(unittest.TestCase):
    """Ensure equivalence of implementation between numpy & Cython"""
    halfFrelon = "LaB6_0020.edf"
    splineFile = "halfccd.spline"

    def setUp(self):
        """Download files"""
        self.halfFrelon = UtilsTest.getimage(self.__class__.halfFrelon)
        self.splineFile = UtilsTest.getimage(self.__class__.splineFile)
        self.det = detectors.FReLoN(self.splineFile)
        self.det.binning = 5, 8  # larger binning makes python loops faster
        self.dis = distortion.Distortion(self.det, self.det.shape, resize=False,
                                         mask=numpy.zeros(self.det.shape, "int8"))

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.fit2dFile = self.halfFrelon = self.splineFile = self.det = self.dis = self.fit2d = self.raw = None

    def test_calc_pos(self):
        self.dis.reset(prepare=False)
        ny = self.dis.calc_pos(False)
        self.dis.reset(prepare=False)
        cy = self.dis.calc_pos(True)
        delta = abs(ny - cy).max()
        self.assertEqual(delta, 0, "calc_pos: equivalence of the cython and numpy model, max error=%s" % delta)

    def test_size(self):
        self.dis.reset(prepare=False)
        ny = self.dis.calc_size(False)
        self.dis.reset(prepare=False)
        cy = self.dis.calc_size(True)
        delta = abs(ny - cy).sum()
        self.assertEqual(delta, 0, "calc_size: equivalence of the cython and numpy model, summed error=%s" % delta)

    def test_lut(self):
        self.dis.reset(method="LUT", prepare=False)
        lut1 = self.dis.calc_LUT(False)
        csr1 = sparse_utils.LUT_to_CSR(lut1)

        self.dis.reset(method="lut", prepare=False)
        lut2 = self.dis.calc_LUT(True)
        csr2 = sparse_utils.LUT_to_CSR(lut2)

        self.dis.reset(method="csr", prepare=False)
        csr3 = self.dis.calc_LUT(True)
        self.dis.reset(method="csr", prepare=False)
        csr4 = self.dis.calc_LUT(False)
        csr4 = sparse_utils.LUT_to_CSR(sparse_utils.CSR_to_LUT(*csr4))

        self.assertEqual(csr1[2].size, csr2[2].size, "right shape 1-2")
        self.assertEqual(csr1[2].size, csr3[2].size, "right shape 1-3")
        self.assertEqual(csr1[2].size, csr4[2].size, "right shape 1-4")

        self.assertTrue(numpy.allclose(csr1[2], csr2[2]), "same indptr 1-2")
        self.assertTrue(numpy.allclose(csr1[2], csr3[2]), "same indptr 1-3")
        self.assertTrue(numpy.allclose(csr1[2], csr4[2]), "same indptr 1-4")

        self.assertTrue(numpy.allclose(csr1[1], csr2[1]), "same indices1-2")
        self.assertTrue(numpy.allclose(csr1[1], csr3[1]), "same indices1-3")
        self.assertTrue(numpy.allclose(csr1[1], csr4[1]), "same indices1-4")

        self.assertTrue(numpy.allclose(csr1[0], csr2[0], atol=2e-7), "same data 1-2")
        self.assertTrue(numpy.allclose(csr1[0], csr3[0], atol=2e-7), "same data 1-3")
        self.assertTrue(numpy.allclose(csr1[0], csr4[0], atol=2e-7), "same data 1-4")


class TestManual(unittest.TestCase):

    def test(self):
        data = numpy.empty((20, 20), dtype=numpy.float32)
        Q = distortion.Quad(data)
        Q.reinit(7.5, 6.5, 2.5, 5.5, 3.5, 1.5, 8.5, 1.5)
        Q.init_slope()
        # print(Q.calc_area_AB(Q.A0, Q.B0)
        # print(Q.calc_area_BC(Q.B0, Q.C0)
        # print(Q.calc_area_CD(Q.C0, Q.D0)
        # print(Q.calc_area_DA(Q.D0, Q.A0)
        print(Q.calc_area())
        Q.populate_box()
        print(Q)
        # print(Q.get_box().sum()
        print(data.sum())
        print("#" * 50)

        Q.reinit(8.5, 1.5, 3.5, 1.5, 2.5, 5.5, 7.5, 6.5)
        Q.init_slope()
        # print(Q.calc_area_AB(Q.A0, Q.B0)
        # print(Q.calc_area_BC(Q.B0, Q.C0)
        # print(Q.calc_area_CD(Q.C0, Q.D0)
        # print(Q.calc_area_DA(Q.D0, Q.A0)
        print(Q.calc_area())
        Q.populate_box()
        print(Q)
        # print(Q.get_box().sum()
        print(data.sum())

        Q.reinit(0.9, 0.9, 0.8, 6.9, 4.3, 3.9, 4.3, 0.9)
        # Q = distortion.Quad((-0.3, 1.9), (-0.4, 2.9), (1.3, 2.9), (1.3, 1.9))
        Q.init_slope()
        print("calc_area_vectorial", Q.calc_area())
        # print(Q.A0, Q.A1, Q.B0, Q.B1, Q.C0, Q.C1, Q.D0, Q.D1
        # print("Slope", Q.pAB, Q.pBC, Q.pCD, Q.pDA
        # print(Q.calc_area_AB(Q.A0, Q.B0), Q.calc_area_BC(Q.B0, Q.C0), Q.calc_area_CD(Q.C0, Q.D0), Q.calc_area_DA(Q.D0, Q.A0)
        print(Q.calc_area())
        Q.populate_box()
        print(data.T)
        # print(Q.get_box().sum()
        print(Q.calc_area())

        print("#" * 50)

        # workin on 256x256
        # x, y = numpy.ogrid[:256, :256]
        # grid = numpy.logical_or(x % 10 == 0, y % 10 == 0) + numpy.ones((256, 256), numpy.float32)
        # det = detectors.FReLoN("frelon_8_8.spline")

        # working with halfccd spline
        x, y = numpy.ogrid[:1024, :2048]
        grid = numpy.logical_or(x % 100 == 0, y % 100 == 0) + numpy.ones((1024, 2048), numpy.float32)
        det = detectors.FReLoN(UtilsTest.getimage("halfccd.spline"))
        # working with halfccd spline
        # x, y = numpy.ogrid[:2048, :2048]
        # grid = numpy.logical_or(x % 100 == 0, y % 100 == 0).astype(numpy.float32) + numpy.ones((2048, 2048), numpy.float32)
        # det = detectors.FReLoN("frelon.spline")

        print(det, det.max_shape)
        dis = distortion.Distortion(det)
        print(dis)
        lut = dis.self.calc_size()
        print(dis.lut_size)
        print(lut.mean())

        dis.calc_LUT()
        out = dis.correct(grid)
        fabio.edfimage.edfimage(data=out.astype("float32")).write("test_correct.edf")

        print("*" * 50)

        # x, y = numpy.ogrid[:2048, :2048]
        # grid = numpy.logical_or(x % 100 == 0, y % 100 == 0)
        # det = detectors.FReLoN("frelon.spline")
        # print(det, det.max_shape
        # dis = Distortion(det)
        # print(dis
        # lut = dis.self.calc_size()
        # print(dis.lut_size
        # print("LUT mean & max", lut.mean(), lut.max()
        # dis.calc_LUT()
        # out = dis.correct(grid)
        # fabio.edfimage.edfimage(data=out.astype("float32")).write("test2048.edf")
        from ..gui.matplotlib import pylab
        pylab.imshow(out)  # , interpolation="nearest")
        pylab.show()


def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(TestImplementations("test_calc_pos"))
    testsuite.addTest(TestImplementations("test_size"))
    testsuite.addTest(TestImplementations("test_lut"))
    testsuite.addTest(TestHalfCCD("test_pos_lut"))
    testsuite.addTest(TestHalfCCD("test_ref_vs_fit2d"))
    testsuite.addTest(TestHalfCCD("test_lut_vs_fit2d"))
    testsuite.addTest(TestHalfCCD("test_csr_vs_fit2d"))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
