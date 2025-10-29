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
"test suite for masked arrays"

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "10/10/2025"

import contextlib
import os
import unittest
import numpy.testing
import fabio
import logging
from .utilstest import UtilsTest
from ..integrator.azimuthal import AzimuthalIntegrator
from ..containers import Integrate1dResult, Integrate2dResult
from ..io import DefaultAiWriter
from ..detectors import Pilatus1M
from ..utils import mathutil
from ..method_registry import IntegrationMethod
logger = logging.getLogger(__name__)


@contextlib.contextmanager
def resulttempfile():
    fd, path = UtilsTest.tempfile(prefix="pyfai_", suffix=".out", dir=__name__)
    os.close(fd)
    os.remove(path)
    yield path
    os.remove(path)


def cleantempdir():
    tempdir = os.path.join(UtilsTest.tempdir, __name__)
    if os.path.isdir(tempdir) and not os.listdir(tempdir):
        os.rmdir(tempdir)


class TestIntegrate1D(unittest.TestCase):

    @classmethod
    def tearDownClass(self):
        self.npt = self.img = self.data = self.ai = self.Rmax = None
        cleantempdir()

    @classmethod
    def setUpClass(self):
        self.npt = 1000
        self.img = UtilsTest.getimage("Pilatus1M.edf")
        with fabio.open(self.img) as fimg:
            self.data = fimg.data
        self.ai = AzimuthalIntegrator(1.58323111834, 0.0334170169115, 0.0412277798782, 0.00648735642526, 0.00755810191106, 0.0, detector=Pilatus1M())
        self.ai.wavelength = 1e-10
        self.Rmax = 3
        self.methods = ["numpy", "cython", "BBox", "splitpixel", "lut"]
        if UtilsTest.opencl:
            self.methods.append("lut_ocl")

    def testQ(self):
        res = {}
        for m in self.methods:
            res[m] = self.ai.integrate1d_legacy(self.data, self.npt, method=m, radial_range=(0.5, 5.8))
        for a in res:
            for b in res:
                R = mathutil.rwp(res[a], res[b])
                mesg = "testQ: %s vs %s measured R=%s<%s" % (a, b, R, self.Rmax)
                if R > self.Rmax:
                    logger.error(mesg)
                else:
                    logger.info(mesg)
                self.assertTrue(R <= self.Rmax, mesg)

    def testR(self):
        res = {}
        for m in self.methods:
            res[m] = self.ai.integrate1d_legacy(self.data, self.npt, method=m, unit="r_mm", radial_range=(20, 150))
        for a in res:
            for b in res:
                R = mathutil.rwp(res[a], res[b])
                mesg = "testR: %s vs %s measured R=%s<%s" % (a, b, R, self.Rmax)
                if R > self.Rmax:
                    logger.error(mesg)
                else:
                    logger.info(mesg)
                self.assertTrue(R <= self.Rmax, mesg)

    def test2th(self):
        res = {}
        for m in self.methods:
            res[m] = self.ai.integrate1d_legacy(self.data, self.npt, method=m, unit="2th_deg", radial_range=(0.5, 5.5))
        for a in res:
            for b in res:
                R = mathutil.rwp(res[a], res[b])
                mesg = "test2th: %s vs %s measured R=%s<%s" % (a, b, R, self.Rmax)
                if R > self.Rmax:
                    logger.error(mesg)
                else:
                    logger.info(mesg)
                self.assertTrue(R <= self.Rmax, mesg)

    def test_new_generation_split(self):
        "Test the equivalent of new generation integrators"
        eps = numpy.finfo("float32").eps
        for split in ("no", "bbox" , "full"):
            res = {}
            methods = IntegrationMethod.select_method(dim=1, split=split)
            radial_range = (0.5, 7.0)
            for m in methods:
                if m.target is not None:
                    continue  # Skip OpenCL
                logger.info("Processing %s" % m)
                res[m] = self.ai.integrate1d_ng(self.data, self.npt,
                                                 variance=self.data,
                                                 method=m,
                                                 radial_range=radial_range,
                                                 error_model="poisson")
                # self.ai.reset()
            keys = list(res.keys())
            # norm = lambda a: a.sum(axis=-1, dtype="float64") if a.ndim == 2 else a
            for i, a in enumerate(keys):
                for b in keys[i:]:
                    if a == b:
                        continue
                    resa = res[a]
                    resb = res[b]
                    R = mathutil.rwp(resa[:2], resb[:2])
                    Ru = mathutil.rwp(resa[::2], resb[::2])
                    err_msg = [f"test_ng_nosplit: {a} vs {b}",
                               f"Intensity measured dev R={R:.3f}<{self.Rmax}",
                               f"Uncertain measured dev R={Ru:.3f}<{self.Rmax}"]
                    for what in ["radial", "intensity", "sigma", "count", "sum_signal", "sum_variance", "sum_normalization", "sum_normalization2", "std", "sem"]:
                        va = resa.__getattribute__(what)
                        vb = resb.__getattribute__(what)
                        if va is not None and vb is not None:
                            err_msg.append(f"{what}: {abs((va-vb)/va/eps).max()} ulp")
                    err_msg = os.linesep.join(err_msg)
                    if max(R, Ru) > self.Rmax :
                        logger.error(err_msg)
                    else:
                        logger.info(err_msg)
                    self.assertLess(R, self.Rmax)
                    self.assertLess(Ru, 50) #self.Rmax) TODO: fix this test

    def test_filename(self):
        with resulttempfile() as filename:
            self.ai.integrate1d_ng(self.data, self.npt, filename=filename)
            self.assertGreater(os.path.getsize(filename), 40)

    def test_defaultwriter(self):
        with resulttempfile() as filename:
            result = self.ai.integrate1d_ng(self.data, self.npt)
            writer = DefaultAiWriter(filename, self.ai)
            writer.write(result)
            writer.close()
            self.assertGreater(os.path.getsize(filename), 40)


class TestIntegrate2D(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.npt = 500
        cls.img = UtilsTest.getimage("Pilatus1M.edf")
        with fabio.open(cls.img) as fimg:
            cls.data = fimg.data
        class DummyLessPilatus(Pilatus1M):
            DUMMY = None
            DELTA_DUMMY = None

        cls.ai = AzimuthalIntegrator(1.58323111834, 0.0334170169115, 0.0412277798782, 0.00648735642526, 0.00755810191106, 0.0,
                                     detector=DummyLessPilatus())
        cls.ai.wavelength = 1e-10
        cls.Rmax = 30
        cls.delta_pos_azim_max = 0.28

    @classmethod
    def tearDownClass(cls):
        cls.npt = None
        cls.img = None
        cls.data = None
        cls.ai = None
        cls.Rmax = None
        cls.delta_pos_azim_max = None
        cleantempdir()

    def testQ(self):
        res = {}
        for m in ("numpy", "cython", "BBox", "splitpixel"):
            res[m] = self.ai.integrate2d(self.data, self.npt, method=m)
        mask = (res["numpy"][0] != 0)
        self.assertTrue(mask.sum() > 36 * self.npt, "10%% of the pixels are valid at least")
        for a in res:
            for b in res:
                delta_pos_rad = abs(res[a][1] - res[b][1]).max()
                delta_pos_azim = abs(res[a][2] - res[b][2]).max()
                R = abs((res[a][0][mask] - res[b][0][mask]) / numpy.maximum(1, res[a][0][mask])).mean() * 100
                mesg = "testQ 2D: %s vs %s measured delta rad=%s azim=%s R=%s<%s" % (a, b, delta_pos_rad, delta_pos_azim, R, self.Rmax)
                if R > self.Rmax:
                    logger.error(mesg)
                else:
                    logger.info(mesg)
                self.assertTrue(delta_pos_rad <= 0.01, mesg)
                self.assertTrue(delta_pos_azim <= self.delta_pos_azim_max, mesg)
                self.assertTrue(R <= self.Rmax, mesg)

    def testR(self):
        res = {}
        for m in ("numpy", "cython", "BBox", "splitpixel"):
            res[m] = self.ai.integrate2d(self.data, self.npt, method=m, unit="r_mm")
        mask = (res["numpy"][0] != 0)
        self.assertTrue(mask.sum() > 36 * self.npt, "10%% of the pixels are valid at least")
        for a in res:
            for b in res:
                delta_pos_rad = abs(res[a][1] - res[b][1]).max()
                delta_pos_azim = abs(res[a][2] - res[b][2]).max()
                R = abs((res[a][0][mask] - res[b][0][mask]) / numpy.maximum(1, res[a][0][mask])).mean() * 100
                mesg = "testR 2D: %s vs %s measured delta rad=%s azim=%s R=%s<%s" % (a, b, delta_pos_rad, delta_pos_azim, R, self.Rmax)
                if R > self.Rmax:
                    logger.error(mesg)
                else:
                    logger.info(mesg)
                self.assertTrue(delta_pos_rad <= 0.28, mesg)
                self.assertTrue(delta_pos_azim <= self.delta_pos_azim_max, mesg)
                self.assertTrue(R <= self.Rmax, mesg)

    def test2th(self):
        res = {}
        for m in ("numpy", "cython", "BBox", "splitpixel"):
            res[m] = self.ai.integrate2d(self.data, self.npt, method=m, unit="2th_deg")
        mask = (res["numpy"][0] != 0)
        self.assertTrue(mask.sum() > 36 * self.npt, "10%% of the pixels are valid at least")
        for a in res:
            for b in res:
                if a == b:
                    continue
                delta_pos_rad = abs(res[a][1] - res[b][1]).max()
                delta_pos_azim = abs(res[a][2] - res[b][2]).max()
                R = abs((res[a][0][mask] - res[b][0][mask]) / numpy.maximum(1, res[a][0][mask])).mean() * 100
                mesg = "test2th 2D: %s vs %s measured delta rad=%s azim=%s R=%s<%s" % (a, b, delta_pos_rad, delta_pos_azim, R, self.Rmax)
                if R > self.Rmax:
                    logger.error(mesg)
                else:
                    logger.info(mesg)
                self.assertTrue(delta_pos_rad <= 0.01, mesg)
                self.assertTrue(R <= self.Rmax, mesg)

    def test_filename(self):
        with resulttempfile() as filename:
            self.ai.integrate2d(self.data, self.npt, filename=filename)
            self.assertGreater(os.path.getsize(filename), 40)

    def test_defaultwriter(self):
        with resulttempfile() as filename:
            result = self.ai.integrate2d(self.data, self.npt)
            writer = DefaultAiWriter(filename, self.ai)
            writer.write(result)
            writer.close()
            self.assertGreater(os.path.getsize(filename), 40)


class TestIntegrateResult(unittest.TestCase):

    def setUp(self):
        self.intensity = numpy.array([[1, 2], [3, 4]])
        self.radial = numpy.array([[3, 2], [3, 4]])
        self.azimuthal = numpy.array([[2, 2], [3, 4]])
        self.sigma = numpy.array([[4, 2], [3, 4]])

    def tearDown(self):
        self.intensity = self.radial = self.azimuthal = self.sigma = None

    def test_result_1d(self):
        result = Integrate1dResult(self.radial, self.intensity)
        # as tuple
        radial, intensity = result
        numpy.testing.assert_equal((self.intensity, self.radial), (intensity, radial))
        # as attributes
        numpy.testing.assert_array_equal(self.intensity, result.intensity)
        numpy.testing.assert_array_equal(self.radial, result.radial)
        self.assertIsNone(result.sigma)

    def test_result_2d(self):
        result = Integrate2dResult(self.intensity, self.radial, self.azimuthal)
        # as tuple
        intensity, radial, azimuthal = result
        numpy.testing.assert_equal((self.intensity, self.radial, self.azimuthal), (intensity, radial, azimuthal))
        # as attributes
        numpy.testing.assert_array_equal(self.intensity, result.intensity)
        numpy.testing.assert_array_equal(self.radial, result.radial)
        numpy.testing.assert_array_equal(self.azimuthal, result.azimuthal)
        self.assertIsNone(result.sigma)

    def test_result_1d_with_sigma(self):
        result = Integrate1dResult(self.radial, self.intensity, self.sigma)
        # as tuple
        radial, intensity, sigma = result
        numpy.testing.assert_equal((self.radial, self.intensity, self.sigma), (radial, intensity, sigma))
        # as attributes
        numpy.testing.assert_array_equal(self.intensity, result.intensity)
        numpy.testing.assert_array_equal(self.radial, result.radial)
        numpy.testing.assert_array_equal(self.sigma, result.sigma)

    def test_result_2d_with_sigma(self):
        result = Integrate2dResult(self.intensity, self.radial, self.azimuthal, self.sigma)
        # as tuple
        intensity, radial, azimuthal, sigma = result
        numpy.testing.assert_equal((self.intensity, self.radial, self.azimuthal, self.sigma), (intensity, radial, azimuthal, sigma))
        # as attributes
        numpy.testing.assert_array_equal(self.intensity, result.intensity)
        numpy.testing.assert_array_equal(self.radial, result.radial)
        numpy.testing.assert_array_equal(self.azimuthal, result.azimuthal)
        numpy.testing.assert_array_equal(self.sigma, result.sigma)

    def test_result_1d_unit(self):
        result = Integrate1dResult(self.radial, self.intensity, self.sigma)
        result._set_unit("foobar")
        numpy.testing.assert_array_equal("foobar", result.unit)

    def test_result_1d_count(self):
        result = Integrate1dResult(self.radial, self.intensity, self.sigma)
        result._set_count(self.sigma)
        numpy.testing.assert_array_equal(self.sigma, result.count)

    def test_result_2d_sum(self):
        result = Integrate2dResult(self.intensity, self.radial, self.azimuthal, self.sigma)
        result._set_sum(self.sigma)
        numpy.testing.assert_array_equal(self.sigma, result.sum)


def suite():
    testsuite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite.addTest(loader(TestIntegrate1D))
    testsuite.addTest(loader(TestIntegrate2D))
    testsuite.addTest(loader(TestIntegrateResult))

    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
