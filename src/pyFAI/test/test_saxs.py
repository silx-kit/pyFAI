#!/usr/bin/env python
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
__date__ = "21/08/2026"

import logging
import unittest

import fabio

from ..detectors import Pilatus1M
from ..integrator.azimuthal import AzimuthalIntegrator
from ..utils import mathutil
from .utilstest import UtilsTest

logger = logging.getLogger(__name__)

if logger.getEffectiveLevel() <= logging.INFO:
    import pylab


class TestSaxs(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        img = UtilsTest.getimage("Pilatus1M.edf")
        with fabio.open(img) as fimg:
            self.data = fimg.data
        self.ai = AzimuthalIntegrator(1.58323111834, 0.0334170169115, 0.0412277798782, 0.00648735642526, 0.00755810191106, 0.0, detector=Pilatus1M())
        self.ai.wavelength = 1e-10
        self.npt = 1000

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.data = self.ai = self.npt = None

    def testMask(self):
        ss = self.ai.mask.sum()
        self.assertTrue(ss == 73533, f"masked pixel = {ss} expected 73533")

    @unittest.skipIf(UtilsTest.low_mem, "test using >100Mb")
    def testNumpy(self):
        method = ("no", "histogram", "python")
        qref, Iref, _ = self.ai.integrate1d_ng(self.data, self.npt, error_model="poisson")
        q, intensity, s = self.ai.integrate1d_ng(self.data, self.npt, error_model="poisson", method=method)
        self.assertTrue(q[0] > 0, f"q[0]>0 {q[0]}")
        self.assertTrue(q[-1] < 8, f"q[-1] < 8, got {q[-1]}")
        self.assertTrue(s.min() >= 0, f"s.min() >= 0 got {s.min()}")
        self.assertTrue(s.max() < 21, f"s.max() < 21 got {s.max()}")
        self.assertTrue(intensity.max() < 52000, f"I.max() < 52000 got {intensity.max()}")
        self.assertTrue(intensity.min() >= 0, f"I.min() >= 0 got {intensity.min()}")
        R = mathutil.rwp((q, intensity), (qref, Iref))
        if R > 20:
            logger.error("Numpy has R=%s", R)
        if logger.getEffectiveLevel() == logging.DEBUG:
            pylab.errorbar(q, intensity, s, label=f"Numpy R={R:.1f}")
            pylab.yscale("log")
        self.assertTrue(R < 20, f"Numpy: Measure R={R}<2")

    @unittest.skipIf(UtilsTest.low_mem, "skipping test using >100M")
    def testCython(self):
        method = ("no", "histogram", "cython")
        qref, Iref, _s = self.ai.integrate1d_ng(self.data, self.npt, error_model="poisson")
        q, intensity, s = self.ai.integrate1d_ng(self.data, self.npt, error_model="poisson", method=method)
        self.assertTrue(q[0] > 0, f"q[0]>0 {q[0]}")
        self.assertTrue(q[-1] < 8, f"q[-1] < 8, got {q[-1]}")
        self.assertTrue(s.min() >= 0, f"s.min() >= 0 got {s.min()}")
        self.assertTrue(s.max() < 21, f"s.max() < 21 got {s.max()}")
        self.assertTrue(intensity.max() < 52000, f"I.max() < 52000 got {intensity.max()}")
        self.assertTrue(intensity.min() >= 0, f"I.min() >= 0 got {intensity.min()}")
        R = mathutil.rwp((q, intensity), (qref, Iref))
        if R > 20:
            logger.error("Cython has R=%s", R)
        if logger.getEffectiveLevel() == logging.DEBUG:
            pylab.errorbar(q, intensity, s, label=f"Cython R={R:.1f}")
            pylab.yscale("log")
        self.assertTrue(R < 20, f"Cython: Measure R={R}<2")

    def testSplitBBox(self):
        method = ("bbox", "histogram", "cython")
        qref, Iref, _s = self.ai.integrate1d_ng(self.data, self.npt, error_model="poisson")
        q, intensity, s = self.ai.integrate1d_ng(self.data, self.npt, error_model="poisson", method=method)
        self.assertTrue(q[0] > 0, f"q[0]>0 {q[0]}")
        self.assertTrue(q[-1] < 8, f"q[-1] < 8, got {q[-1]}")
        self.assertTrue(s.min() >= 0, f"s.min() >= 0 got {s.min()}")
        self.assertTrue(s.max() < 21, f"s.max() < 21 got {s.max()}")
        self.assertTrue(intensity.max() < 52000, f"I.max() < 52000 got {intensity.max()}")
        self.assertTrue(intensity.min() >= 0, f"I.min() >= 0 got {intensity.min()}")
        R = mathutil.rwp((q, intensity), (qref, Iref))
        if R > 20:
            logger.error("SplitPixel has R=%s", R)
        if logger.getEffectiveLevel() == logging.DEBUG:
            pylab.errorbar(q, intensity, s, label=f"SplitBBox R={R:.1f}")
            pylab.yscale("log")
        self.assertEqual(R < 20, True, f"SplitBBox: Measure R={R}<20")

    def testSplitPixel(self):
        method = ("full", "histogram", "cython")
        qref, Iref, _s = self.ai.integrate1d_ng(self.data, self.npt, error_model="poisson")
        q, intensity, s = self.ai.integrate1d_ng(self.data, self.npt, error_model="poisson", method=method)
        self.assertTrue(q[0] > 0, f"q[0]>0 {q[0]}")
        self.assertTrue(q[-1] < 8, f"q[-1] < 8, got {q[-1]}")
        self.assertTrue(s.min() >= 0, f"s.min() >= 0 got {s.min()}")
        self.assertTrue(s.max() < 21, f"s.max() < 21 got {s.max()}")
        self.assertTrue(intensity.max() < 52000, f"I.max() < 52000 got {intensity.max()}")
        self.assertTrue(intensity.min() >= 0, f"I.min() >= 0 got {intensity.min()}")
        R = mathutil.rwp((q, intensity), (qref, Iref))
        if R > 20:
            logger.error("SplitPixel has R=%s", R)
        if logger.getEffectiveLevel() == logging.DEBUG:
            pylab.errorbar(q, intensity, s, label=f"SplitPixel R={R:.1f}")
            pylab.yscale("log")
        self.assertEqual(R < 20, True, f"SplitPixel: Measure R={R}<20")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestSaxs))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
    if logger.getEffectiveLevel() == logging.DEBUG:
        pylab.legend()
        pylab.show()
        input()
        pylab.clf()
