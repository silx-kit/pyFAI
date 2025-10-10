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

import unittest
import logging
import fabio
from .utilstest import UtilsTest
from ..integrator.azimuthal import AzimuthalIntegrator
from ..detectors import Pilatus1M
from ..utils import mathutil
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
        self.assertTrue(ss == 73533, "masked pixel = %s expected 73533" % ss)

    @unittest.skipIf(UtilsTest.low_mem, "test using >100Mb")
    def testNumpy(self):
        method = ("no", "histogram", "python")
        qref, Iref, _ = self.ai.integrate1d_ng(self.data, self.npt, error_model="poisson")
        q, intensity, s = self.ai.integrate1d_ng(self.data, self.npt, error_model="poisson", method=method)
        self.assertTrue(q[0] > 0, "q[0]>0 %s" % q[0])
        self.assertTrue(q[-1] < 8, "q[-1] < 8, got %s" % q[-1])
        self.assertTrue(s.min() >= 0, "s.min() >= 0 got %s" % (s.min()))
        self.assertTrue(s.max() < 21, "s.max() < 21 got %s" % (s.max()))
        self.assertTrue(intensity.max() < 52000, "I.max() < 52000 got %s" % (intensity.max()))
        self.assertTrue(intensity.min() >= 0, "I.min() >= 0 got %s" % (intensity.min()))
        R = mathutil.rwp((q, intensity), (qref, Iref))
        if R > 20:
            logger.error("Numpy has R=%s", R)
        if logger.getEffectiveLevel() == logging.DEBUG:
            pylab.errorbar(q, intensity, s, label="Numpy R=%.1f" % R)
            pylab.yscale("log")
        self.assertTrue(R < 20, "Numpy: Measure R=%s<2" % R)

    @unittest.skipIf(UtilsTest.low_mem, "skipping test using >100M")
    def testCython(self):
        method = ("no", "histogram", "cython")
        qref, Iref, _s = self.ai.integrate1d_ng(self.data, self.npt, error_model="poisson")
        q, intensity, s = self.ai.integrate1d_ng(self.data, self.npt, error_model="poisson", method=method)
        self.assertTrue(q[0] > 0, "q[0]>0 %s" % q[0])
        self.assertTrue(q[-1] < 8, "q[-1] < 8, got %s" % q[-1])
        self.assertTrue(s.min() >= 0, "s.min() >= 0 got %s" % (s.min()))
        self.assertTrue(s.max() < 21, "s.max() < 21 got %s" % (s.max()))
        self.assertTrue(intensity.max() < 52000, "I.max() < 52000 got %s" % (intensity.max()))
        self.assertTrue(intensity.min() >= 0, "I.min() >= 0 got %s" % (intensity.min()))
        R = mathutil.rwp((q, intensity), (qref, Iref))
        if R > 20:
            logger.error("Cython has R=%s", R)
        if logger.getEffectiveLevel() == logging.DEBUG:
            pylab.errorbar(q, intensity, s, label="Cython R=%.1f" % R)
            pylab.yscale("log")
        self.assertTrue(R < 20, "Cython: Measure R=%s<2" % R)

    def testSplitBBox(self):
        method = ("bbox", "histogram", "cython")
        qref, Iref, _s = self.ai.integrate1d_ng(self.data, self.npt, error_model="poisson")
        q, intensity, s = self.ai.integrate1d_ng(self.data, self.npt, error_model="poisson", method=method)
        self.assertTrue(q[0] > 0, "q[0]>0 %s" % q[0])
        self.assertTrue(q[-1] < 8, "q[-1] < 8, got %s" % q[-1])
        self.assertTrue(s.min() >= 0, "s.min() >= 0 got %s" % (s.min()))
        self.assertTrue(s.max() < 21, "s.max() < 21 got %s" % (s.max()))
        self.assertTrue(intensity.max() < 52000, "I.max() < 52000 got %s" % (intensity.max()))
        self.assertTrue(intensity.min() >= 0, "I.min() >= 0 got %s" % (intensity.min()))
        R = mathutil.rwp((q, intensity), (qref, Iref))
        if R > 20:
            logger.error("SplitPixel has R=%s", R)
        if logger.getEffectiveLevel() == logging.DEBUG:
            pylab.errorbar(q, intensity, s, label="SplitBBox R=%.1f" % R)
            pylab.yscale("log")
        self.assertEqual(R < 20, True, "SplitBBox: Measure R=%s<20" % R)

    def testSplitPixel(self):
        method = ("full", "histogram", "cython")
        qref, Iref, _s = self.ai.integrate1d_ng(self.data, self.npt, error_model="poisson")
        q, intensity, s = self.ai.integrate1d_ng(self.data, self.npt, error_model="poisson", method=method)
        self.assertTrue(q[0] > 0, "q[0]>0 %s" % q[0])
        self.assertTrue(q[-1] < 8, "q[-1] < 8, got %s" % q[-1])
        self.assertTrue(s.min() >= 0, "s.min() >= 0 got %s" % (s.min()))
        self.assertTrue(s.max() < 21, "s.max() < 21 got %s" % (s.max()))
        self.assertTrue(intensity.max() < 52000, "I.max() < 52000 got %s" % (intensity.max()))
        self.assertTrue(intensity.min() >= 0, "I.min() >= 0 got %s" % (intensity.min()))
        R = mathutil.rwp((q, intensity), (qref, Iref))
        if R > 20:
            logger.error("SplitPixel has R=%s", R)
        if logger.getEffectiveLevel() == logging.DEBUG:
            pylab.errorbar(q, intensity, s, label="SplitPixel R=%.1f" % R)
            pylab.yscale("log")
        self.assertEqual(R < 20, True, "SplitPixel: Measure R=%s<20" % R)


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
