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

"""Test suites for median filtering engines"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "06/12/2024"

import unittest
import numpy
import logging
logger = logging.getLogger(__name__)
from .utilstest import UtilsTest
import fabio
from .. import load
from ..opencl import ocl

class TestMedfilt(unittest.TestCase):
    """Test Azimuthal median filtering results
    """

    @classmethod
    def setUpClass(cls)->None:
        super(TestMedfilt, cls).setUpClass()
        cls.method = ("full", "csr", "python")
        cls.img = fabio.open(UtilsTest.getimage("mock.tif")).data
        cls.ai = load({ "dist": 0.1,
                        "poni1":0.03,
                        "poni2":0.03,
                        "detector": "Detector",
                        "detector_config": {"pixel1": 1e-4,
                                            "pixel2": 1e-4,
                                            "max_shape": [500, 600],
                                            "orientation": 3}})
        cls.npt = 100

    @classmethod
    def tearDownClass(cls)->None:
        super(TestMedfilt, cls).tearDownClass()
        cls.method = cls.img =cls.ai =cls.npt =None

    def test_python(self):
        method = list(self.method)
        method[-1] = "python"
        method = tuple(self.method)
        ref = self.ai.integrate1d(self.img, self.npt, unit="2th_rad", method=method, error_model="poisson")
        # print(ref.method)
        engine = self.ai.engines[ref.method].engine
        obt = engine.medfilt(self.img,
                             solidangle=self.ai.solidAngleArray(),
                             quant_min=0,quant_max=1,  # taking all Like this it works like a normal mean
                             error_model="poisson")

        self.assertTrue(numpy.allclose(ref.radial, obt.position), "radial matches")
        self.assertTrue(numpy.allclose(ref.sum_signal, obt.signal), "signal matches")
        self.assertTrue(numpy.allclose(ref.sum_variance, obt.variance), "variance matches")
        self.assertTrue(numpy.allclose(ref.sum_normalization, obt.normalization), "normalization matches")
        self.assertTrue(numpy.allclose(ref.sum_normalization2, obt.norm_sq), "norm_sq matches")

        self.assertTrue(numpy.allclose(ref.intensity, obt.intensity), "intensity matches")
        self.assertTrue(numpy.allclose(ref.sigma, obt.sigma), "sigma matches")
        self.assertTrue(numpy.allclose(ref.std, obt.std), "std matches")
        self.assertTrue(numpy.allclose(ref.sem, obt.sem), "sem matches")

    def test_cython(self):
        # print(self.ai)

        method = list(self.method)
        # method[0] = "no"
        method[-1] = "cython"
        method = tuple(method)
        ref = self.ai.integrate1d(self.img, self.npt, unit="2th_rad", method=method, error_model="poisson")
        print(ref.method)
        engine = self.ai.engines[ref.method].engine
        print(engine)
        obt = engine.medfilt(self.img,
                             solidangle=self.ai.solidAngleArray(),
                             quant_min=0,quant_max=1,  # taking all Like this it works like a normal mean
                             error_model="poisson")

        # print(ref.count-obt.count)
        # print()
        self.assertTrue(numpy.allclose(ref.radial, obt.position), "radial matches")
        self.assertTrue(numpy.allclose(ref.sum_signal, obt.signal), "signal matches")
        self.assertTrue(numpy.allclose(ref.sum_variance, obt.variance), "variance matches")
        self.assertTrue(numpy.allclose(ref.sum_normalization, obt.normalization), "normalization matches")
        self.assertTrue(numpy.allclose(ref.sum_normalization2, obt.norm_sq), "norm_sq matches")
        # self.assertTrue(numpy.allclose(ref.count, obt.count), "count matches") # not valid with pixel splitting
        self.assertTrue(numpy.allclose(ref.intensity, obt.intensity), "intensity matches")
        self.assertTrue(numpy.allclose(ref.sigma, obt.sigma), "sigma matches")
        self.assertTrue(numpy.allclose(ref.std, obt.std), "std matches")
        self.assertTrue(numpy.allclose(ref.sem, obt.sem), "sem matches")

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_opencl(self):
        method = list(self.method)
        method[-1] = "opencl"
        method = tuple(method)
        ref = self.ai.integrate1d(self.img, self.npt, unit="2th_rad", method=method, error_model="poisson")
        engine = self.ai.engines[ref.method].engine
        obt = engine.medfilt(self.img,
                             solidangle=self.ai.solidAngleArray(),
                             quant_min=0,quant_max=1,  # taking all like this: it works like a normal mean
                             error_model="poisson")
        self.assertTrue(numpy.allclose(ref.radial, obt.position), "radial matches")



        thres = 1e-4
        thres_cnt = 1


        # self.assertLessEqual(numpy.sum(abs(ref.sum_signal-obt.signal)>thres), thres_cnt, "signal matches")
        # self.assertLessEqual(numpy.sum(abs(ref.sum_variance-obt.variance)>thres), thres_cnt, "variance matches")
        # self.assertLessEqual(numpy.sum(abs(ref.sum_normalization-obt.normalization)>thres), thres_cnt, "normalization matches")
        # self.assertLessEqual(numpy.sum(abs(ref.sum_normalization2-obt.norm_sq)>thres), thres_cnt, "norm_sq matches")
        self.assertTrue(numpy.allclose(engine._indptr[1:]-engine._indptr[:-1], obt.count), "count matches") # not valid with pixel splitting
        self.assertTrue(numpy.allclose(ref.sum_signal, obt.signal, atol=1e-4, rtol=1e-6), "signal matches")
        self.assertTrue(numpy.allclose(ref.sum_variance, obt.variance, atol=1e-4, rtol=1e-6), "variance matches")
        self.assertTrue(numpy.allclose(ref.sum_normalization, obt.normalization, atol=1e-4, rtol=1e-6), "normalization matches")
        self.assertTrue(numpy.allclose(ref.sum_normalization2, obt.norm_sq, atol=1e-2, rtol=1e-3), "norm_sq matches")

        self.assertLessEqual(numpy.sum(abs(ref.intensity-obt.intensity)>thres), thres_cnt, "intensity matches")
        self.assertLessEqual(numpy.sum(abs(ref.sigma-obt.sigma)>thres), thres_cnt, "sigma matches")
        self.assertLessEqual(numpy.sum(abs(ref.std-obt.std>thres)), thres_cnt, "std matches")
        self.assertLessEqual(numpy.sum(abs(ref.sem-obt.sem>thres)), thres_cnt, "sem matches")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestMedfilt))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
