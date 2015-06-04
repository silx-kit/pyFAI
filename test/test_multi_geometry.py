#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal Integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#


"""
Test suites for multi_geometry modules
"""


import unittest, numpy, os, sys, time, logging
if sys.version_info[0] > 2:
    raw_input = input
if __name__ == '__main__':
    import pkgutil
    __path__ = pkgutil.extend_path([os.path.dirname(__file__)], "pyFAI.test")
from .utilstest import UtilsTest, getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]

from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.multi_geometry import MultiGeometry
from pyFAI.detectors import Detector
import fabio


class TestMultiGeometry(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.data = fabio.open(UtilsTest.getimage("1788/moke.tif")).data
        self.lst_data = [self.data[:250, :300], self.data[250:, :300], self.data[:250, 300:], self.data[250:, 300:]]
        self.det = Detector(1e-4, 1e-4)
        self.det.max_shape = (500, 600)
        self.sub_det = Detector(1e-4, 1e-4)
        self.sub_det.max_shape = (250, 300)
        self.ai = AzimuthalIntegrator(0.1, 0.03, 0.03, detector=self.det)
        self.range = (0, 23)
        self.ais = [AzimuthalIntegrator(0.1, 0.030, 0.03, detector=self.sub_det),
                    AzimuthalIntegrator(0.1, 0.005, 0.03, detector=self.sub_det),
                    AzimuthalIntegrator(0.1, 0.030, 0.00, detector=self.sub_det),
                    AzimuthalIntegrator(0.1, 0.005, 0.00, detector=self.sub_det),
                    ]
        self.mg = MultiGeometry(self.ais, radial_range=self.range, unit="2th_deg")
        self.N = 390

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.data = None
        self.lst_data = None
        self.det = None
        self.sub_det = None
        self.ai = None
        self.ais = None
        self.mg = None

    def test_integrate1d(self):
        tth_ref, I_ref = self.ai.integrate1d(self.data, radial_range=self.range, npt=self.N, unit="2th_deg", method="splitpixel")
        obt = self.mg.integrate1d(self.lst_data, self.N)
        tth_obt, I_obt = obt
        self.assertEqual(abs(tth_ref - tth_obt).max(), 0, "Bin position is the same")
        # intensity need to be scaled by solid angle 1e-4*1e-4/0.1**2 = 1e-6
        delta = (abs(I_obt * 1e6 - I_ref).max())
        self.assert_(delta < 5e-5, "Intensity is the same delta=%s" % delta)

    def test_integrate2d(self):
        ref = self.ai.integrate2d(self.data, self.N, 360, radial_range=self.range, azimuth_range=(-180, 180), unit="2th_deg", method="splitpixel", all=True)
        obt = self.mg.integrate2d(self.lst_data, self.N, 360, all=True)
        self.assertEqual(abs(ref["radial"] - obt["radial"]).max(), 0, "Bin position is the same")
        self.assertEqual(abs(ref["azimuthal"] - obt["azimuthal"]).max(), 0, "Bin position is the same")
        # intensity need to be scaled by solid angle 1e-4*1e-4/0.1**2 = 1e-6
        delta = abs(obt["I"] * 1e6 - ref["I"])[obt["count"] >= 1e-6]  # restrict on valid pixel
        delta_cnt = abs(obt["count"] - ref["count"])
        delta_sum = abs(obt["sum"] * 1e6 - ref["sum"])
        if delta.max() > 0:
            logger.warning("TestMultiGeometry.test_integrate2d gave intensity difference of %s" % delta.max())
            if logger.level <= logging.DEBUG:
                from matplotlib import pyplot as plt
                f = plt.figure()
                a1 = f.add_subplot(2, 2, 1)
                a1.imshow(ref["sum"])
                a2 = f.add_subplot(2, 2, 2)
                a2.imshow(obt["sum"])
                a3 = f.add_subplot(2, 2, 3)
                a3.imshow(delta_sum)
                a4 = f.add_subplot(2, 2, 4)
                a4.plot(delta_sum.sum(axis=0))
                f.show()
                raw_input()

        self.assert_(delta_cnt.max() < 0.001, "pixel count is the same delta=%s" % delta_cnt.max())
        self.assert_(delta_sum.max() < 0.03, "pixel sum is the same delta=%s" % delta_sum.max())
        self.assert_(delta.max() < 0.004, "pixel intensity is the same (for populated pixels) delta=%s" % delta.max())


def test_suite_all_multi_geometry():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestMultiGeometry("test_integrate1d"))
    testSuite.addTest(TestMultiGeometry("test_integrate2d"))
    return testSuite


if __name__ == '__main__':
    mysuite = test_suite_all_multi_geometry()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
