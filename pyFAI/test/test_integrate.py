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
"test suite for masked arrays"

from __future__ import absolute_import, division, print_function

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "12/01/2018"


import tempfile
import contextlib
import os
import unittest
import numpy.testing
import fabio
import logging
from .utilstest import UtilsTest
logger = logging.getLogger(__name__)
from ..azimuthalIntegrator import AzimuthalIntegrator
from ..containers import Integrate1dResult
from ..containers import Integrate2dResult
from ..io import DefaultAiWriter
from ..detectors import Pilatus1M
from ..utils import mathutil


@contextlib.contextmanager
def resulttempfile():
    fd, path = tempfile.mkstemp(prefix="pyfai_", suffix=".out")
    os.close(fd)
    os.remove(path)
    yield path
    os.remove(path)


class TestIntegrate1D(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.npt = 1000
        self.img = UtilsTest.getimage("Pilatus1M.edf")
        self.data = fabio.open(self.img).data
        self.ai = AzimuthalIntegrator(1.58323111834, 0.0334170169115, 0.0412277798782, 0.00648735642526, 0.00755810191106, 0.0, detector=Pilatus1M())
        self.ai.wavelength = 1e-10
        self.Rmax = 3
        self.methods = ["numpy", "cython", "BBox", "splitpixel", "lut"]
        if UtilsTest.opencl:
            self.methods.append("lut_ocl")

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.npt = self.img = self.data = self.ai = self.Rmax = None

    def testQ(self):
        res = {}
        for m in self.methods:
            res[m] = self.ai.integrate1d(self.data, self.npt, method=m, radial_range=(0.5, 5.8))
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
            res[m] = self.ai.integrate1d(self.data, self.npt, method=m, unit="r_mm", radial_range=(20, 150))
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
            res[m] = self.ai.integrate1d(self.data, self.npt, method=m, unit="2th_deg", radial_range=(0.5, 5.5))
        for a in res:
            for b in res:
                R = mathutil.rwp(res[a], res[b])
                mesg = "test2th: %s vs %s measured R=%s<%s" % (a, b, R, self.Rmax)
                if R > self.Rmax:
                    logger.error(mesg)
                else:
                    logger.info(mesg)
                self.assertTrue(R <= self.Rmax, mesg)

    def test_filename(self):
        with resulttempfile() as filename:
            self.ai.integrate1d(self.data, self.npt, filename=filename)
            self.assertGreater(os.path.getsize(filename), 40)

    def test_defaultwriter(self):
        with resulttempfile() as filename:
            result = self.ai.integrate1d(self.data, self.npt)
            writer = DefaultAiWriter(filename, self.ai)
            writer.write(result)
            writer.close()
            self.assertGreater(os.path.getsize(filename), 40)


class TestIntegrate2D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.npt = 500
        cls.img = UtilsTest.getimage("Pilatus1M.edf")
        cls.data = fabio.open(cls.img).data
        cls.ai = AzimuthalIntegrator(1.58323111834, 0.0334170169115, 0.0412277798782, 0.00648735642526, 0.00755810191106, 0.0, detector=Pilatus1M())
        cls.ai.wavelength = 1e-10
        cls.Rmax = 20
        cls.delta_pos_azim_max = 0.28

    @classmethod
    def tearDownClass(cls):
        cls.npt = None
        cls.img = None
        cls.data = None
        cls.ai = None
        cls.Rmax = None
        cls.delta_pos_azim_max = None

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
        self.I = numpy.array([[1, 2], [3, 4]])
        self.radial = numpy.array([[3, 2], [3, 4]])
        self.azimuthal = numpy.array([[2, 2], [3, 4]])
        self.sigma = numpy.array([[4, 2], [3, 4]])

    def tearDown(self):
        self.I = self.radial = self.azimuthal = self.sigma = None

    def test_result_1d(self):
        result = Integrate1dResult(self.radial, self.I)
        # as tuple
        radial, I = result
        numpy.testing.assert_equal((self.I, self.radial), (I, radial))
        # as attributes
        numpy.testing.assert_array_equal(self.I, result.intensity)
        numpy.testing.assert_array_equal(self.radial, result.radial)
        self.assertIsNone(result.sigma)

    def test_result_2d(self):
        result = Integrate2dResult(self.I, self.radial, self.azimuthal)
        # as tuple
        I, radial, azimuthal = result
        numpy.testing.assert_equal((self.I, self.radial, self.azimuthal), (I, radial, azimuthal))
        # as attributes
        numpy.testing.assert_array_equal(self.I, result.intensity)
        numpy.testing.assert_array_equal(self.radial, result.radial)
        numpy.testing.assert_array_equal(self.azimuthal, result.azimuthal)
        self.assertIsNone(result.sigma)

    def test_result_1d_with_sigma(self):
        result = Integrate1dResult(self.radial, self.I, self.sigma)
        # as tuple
        radial, I, sigma = result
        numpy.testing.assert_equal((self.radial, self.I, self.sigma), (radial, I, sigma))
        # as attributes
        numpy.testing.assert_array_equal(self.I, result.intensity)
        numpy.testing.assert_array_equal(self.radial, result.radial)
        numpy.testing.assert_array_equal(self.sigma, result.sigma)

    def test_result_2d_with_sigma(self):
        result = Integrate2dResult(self.I, self.radial, self.azimuthal, self.sigma)
        # as tuple
        I, radial, azimuthal, sigma = result
        numpy.testing.assert_equal((self.I, self.radial, self.azimuthal, self.sigma), (I, radial, azimuthal, sigma))
        # as attributes
        numpy.testing.assert_array_equal(self.I, result.intensity)
        numpy.testing.assert_array_equal(self.radial, result.radial)
        numpy.testing.assert_array_equal(self.azimuthal, result.azimuthal)
        numpy.testing.assert_array_equal(self.sigma, result.sigma)

    def test_result_1d_unit(self):
        result = Integrate1dResult(self.radial, self.I, self.sigma)
        result._set_unit("foobar")
        numpy.testing.assert_array_equal("foobar", result.unit)

    def test_result_1d_count(self):
        result = Integrate1dResult(self.radial, self.I, self.sigma)
        result._set_count(self.sigma)
        numpy.testing.assert_array_equal(self.sigma, result.count)

    def test_result_2d_sum(self):
        result = Integrate2dResult(self.I, self.radial, self.azimuthal, self.sigma)
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
