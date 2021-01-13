#!/usr/bin/env python3
# coding: utf-8
#
#    Project: Fast Azimuthal Integration
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

"""Test suites for multi_geometry modules"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "13/01/2021"

import unittest
import logging
from .utilstest import UtilsTest
logger = logging.getLogger(__name__)
import numpy
from ..azimuthalIntegrator import AzimuthalIntegrator
from ..multi_geometry import MultiGeometry
from ..detectors import Detector

import fabio


class TestMultiGeometry(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = fabio.open(UtilsTest.getimage("mock.tif")).data
        cls.lst_data = [cls.data[:250,:300], cls.data[250:,:300], cls.data[:250, 300:], cls.data[250:, 300:]]
        cls.det = Detector(1e-4, 1e-4)
        cls.det.max_shape = (500, 600)
        cls.sub_det = Detector(1e-4, 1e-4)
        cls.sub_det.max_shape = (250, 300)
        cls.ai = AzimuthalIntegrator(0.1, 0.03, 0.03, detector=cls.det)
        cls.range = (0, 23)
        cls.ais = [AzimuthalIntegrator(0.1, 0.030, 0.03, detector=cls.sub_det),
                   AzimuthalIntegrator(0.1, 0.005, 0.03, detector=cls.sub_det),
                   AzimuthalIntegrator(0.1, 0.030, 0.00, detector=cls.sub_det),
                   AzimuthalIntegrator(0.1, 0.005, 0.00, detector=cls.sub_det),
                   ]
        cls.mg = MultiGeometry(cls.ais, radial_range=cls.range, unit="2th_deg")
        cls.N = 390
        cls.method = ("full", "histogram", "cython")

    @classmethod
    def tearDownClass(cls):
        cls.data = cls.lst_data = cls.det = cls.sub_det = cls.ai = None
        cls.range = cls.ais = cls.mg = cls.N = None

    def setUp(self):
        """
        Python2.6 compatibility !!!
        """
        unittest.TestCase.setUp(self)
        if "data" not in dir(self):
            self.setUpClass()

    def tearDown(self):
        self.data = self.lst_data = self.det = self.sub_det = self.ai = None
        self.range = self.ais = self.mg = self.N = None

    def test_integrate1d(self):
        res = self.ai.integrate1d_ng(self.data, radial_range=self.range,
                                                           npt=self.N, unit="2th_deg",
                                                           method=self.method,
                                                           variance=numpy.ones_like(self.data))
        tth_ref, I_ref, sigma_ref = res
        lst_var = [numpy.ones_like(i) for i in self.lst_data]
        obt = self.mg.integrate1d(self.lst_data, self.N, lst_variance=lst_var,
                                  method=self.method)
        tth_obt, I_obt, sigma_obt = obt

        self.assertEqual(abs(tth_ref - tth_obt).max(), 0, "Bin position is the same")
        # intensity need to be scaled by solid angle 1e-4*1e-4/0.1**2 = 1e-6
        delta = (abs(I_obt * 1e-6 - I_ref).max())
        self.assertTrue(delta < 9e-5, "Intensity is the same delta=%s" % delta)

        delta = (abs(sigma_obt * 1e-6 - sigma_ref).max())
        self.assertTrue(delta < 9e-5, "Standard deviation is the same delta=%s" % delta)

    def test_integrate1d_withpol(self):
        tth_ref, I_ref = self.ai.integrate1d_ng(self.data, radial_range=self.range,
                                                npt=self.N, unit="2th_deg", method="splitpixel",
                                                polarization_factor=0.9)
        obt = self.mg.integrate1d(self.lst_data, self.N, polarization_factor=0.9, method=self.method)
        tth_obt, I_obt = obt
        self.assertEqual(abs(tth_ref - tth_obt).max(), 0, "Bin position is the same")
        # intensity need to be scaled by solid angle 1e-4*1e-4/0.1**2 = 1e-6
        delta = (abs(I_obt * 1e-6 - I_ref).max())
        self.assertTrue(delta < 9e-5, "Intensity is the same delta=%s" % delta)

    def test_integrate2d(self):
        ref = self.ai.integrate2d_ng(self.data, self.N, 360, radial_range=self.range, azimuth_range=(-180, 180), unit="2th_deg",
                                     method=self.method)
        obt = self.mg.integrate2d(self.lst_data, self.N, 360, method=self.method)
        self.assertEqual(abs(ref.radial - obt.radial).max(), 0, "Bin position is the same")
        self.assertEqual(abs(ref.azimuthal - obt.azimuthal).max(), 0, "Bin position is the same")
        # intensity need to be scaled by solid angle 1e-4*1e-4/0.1**2 = 1e-6
        mask = obt.count <= 0.1  # restrict on valid pixel
        mask[:, 0:2] = True
        delta = abs(obt.intensity * 1e-6 - ref.intensity)
        delta_norm = abs(obt.sum_normalization * 1e6 - ref.sum_normalization)
        delta_sum = abs(obt.sum_signal - ref.sum_signal)
        delta[mask] = 0
        delta_norm[mask] = 0
        delta_sum[mask] = 0
        if delta.max() > 0:
            logger.warning("TestMultiGeometry.test_integrate2d gave difference "
                           "of intensity: %s, count: %s cum: %s",
                           delta.max(), delta_norm.max(), delta_sum.max())

        self.assertTrue(delta_norm.max() < 0.001, "pixel normalization is the same delta=%s" % delta_norm.max())
        self.assertTrue(delta_sum.max() < 0.04, "pixel sum is the same delta=%s" % delta_sum.max())
        self.assertTrue(delta.max() < 0.007, "pixel intensity is the same (for populated pixels) delta=%s" % delta.max())


def suite():
    testsuite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite.addTest(loader(TestMultiGeometry))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
