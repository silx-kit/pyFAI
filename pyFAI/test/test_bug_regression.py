#!/usr/bin/python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) 2015 European Synchrotron Radiation Facility, Grenoble, France
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

from __future__ import absolute_import, division, print_function

__doc__ = """test suite for non regression on some bugs.

Please refer to their respective bug number
https://github.com/kif/pyFAI/issues
"""
__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "12/04/2016"

import sys
import os
import unittest
import numpy
import subprocess
from .utilstest import getLogger, UtilsTest  # , Rwp, getLogger
logger = getLogger(__file__)
import fabio
from .. import load
from ..azimuthalIntegrator import AzimuthalIntegrator
from .. import detectors


class TestBug170(unittest.TestCase):
    """
    Test a mar345 image with 2300 pixels size
    """

    def setUp(self):
        ponitxt = """
Detector: Mar345
PixelSize1: 0.00015
PixelSize2: 0.00015
Distance: 0.446642915189
Poni1: 0.228413453499
Poni2: 0.272291324302
Rot1: 0.0233130647508
Rot2: 0.0011735285628
Rot3: -7.22446379865e-08
SplineFile: None
Wavelength: 7e-11
"""
        self.ponifile = os.path.join(UtilsTest.tempdir, "bug170.poni")
        with open(self.ponifile, "w") as poni:
            poni.write(ponitxt)
        self.data = numpy.random.random((2300, 2300))

    def tearDown(self):
        if os.path.exists(self.ponifile):
            os.unlink(self.ponifile)
        self.data = None

    def test_bug170(self):
        ai = load(self.ponifile)
        logger.debug(ai.mask.shape)
        logger.debug(ai.detector.pixel1)
        logger.debug(ai.detector.pixel2)
        ai.integrate1d(self.data, 2000)


class TestBug211(unittest.TestCase):
    """
    Check the quantile filter in pyFAI-average
    """
    def setUp(self):
        shape = (100, 100)
        dtype = numpy.float32
        self.image_files = []
        self.outfile = os.path.join(UtilsTest.tempdir, "out.edf")
        res = numpy.zeros(shape, dtype=dtype)
        for i in range(5):
            fn = os.path.join(UtilsTest.tempdir, "img_%i.edf" % i)
            if i == 3:
                data = numpy.zeros(shape, dtype=dtype)
            elif i == 4:
                data = numpy.ones(shape, dtype=dtype)
            else:
                data = numpy.random.random(shape).astype(dtype)
                res += data
            e = fabio.edfimage.edfimage(data=data)
            e.write(fn)
            self.image_files.append(fn)
        self.res = res / 3.0
        self.exe, self.env = UtilsTest.script_path("pyFAI-average")

    def tearDown(self):
        for fn in self.image_files:
            os.unlink(fn)
        if os.path.exists(self.outfile):
            os.unlink(self.outfile)
        self.image_files = None
        self.res = None
        self.exe = self.env = None

    def test_quantile(self):
        if not os.path.exists(self.exe):
            print("Error with executable: %s" % self.exe)
            print(os.listdir(os.path.dirname(self.exe)))
        p = subprocess.call([sys.executable, self.exe, "--quiet", "-q", "0.2-0.8", "-o", self.outfile] + self.image_files,
                            shell=False, env=self.env)
        if p:
            l = [sys.executable, self.exe, "--quiet", "-q", "0.2-0.8", "-o", self.outfile] + self.image_files
            logger.error(os.linesep + (" ".join(l)))
            env = "Environment:"
            for k, v in self.env.items():
                env += "%s    %s: %s" % (os.linesep, k, v)
            logger.error(env)
        self.assertEqual(p, 0, msg="pyFAI-average return code %i != 0" % p)
        self.assert_(numpy.allclose(fabio.open(self.outfile).data, self.res),
                         "pyFAI-average with quantiles gives good results")


class TestBug232(unittest.TestCase):
    """
    Check the copy and deepcopy methods of Azimuthal integrator
    """
    def test(self):
        det = detectors.ImXPadS10()
        ai = AzimuthalIntegrator(dist=1, detector=det)
        data = numpy.random.random(det.shape)
        tth, I = ai.integrate1d(data, 100, unit="r_mm")
        import copy
        ai2 = copy.copy(ai)
        self.assertNotEqual(id(ai), id(ai2), "copy instances are different")
        self.assertEqual(id(ai.ra), id(ai2.ra), "copy arrays are the same after copy")
        self.assertEqual(id(ai.detector), id(ai2.detector), "copy detector are the same after copy")
        ai3 = copy.deepcopy(ai)
        self.assertNotEqual(id(ai), id(ai3), "deepcopy instances are different")
        self.assertNotEqual(id(ai.ra), id(ai3.ra), "deepcopy arrays are different after copy")
        self.assertNotEqual(id(ai.detector), id(ai3.detector), "deepcopy arrays are different after copy")


class TestBug174(unittest.TestCase):
    """
    wavelength change not taken into account (memoization error)
    """
    def test(self):
        ai = load(UtilsTest.getimage("1893/Pilatus1M.poni"))
        data = fabio.open(UtilsTest.getimage("1883/Pilatus1M.edf")).data
        wl1 = 1e-10
        wl2 = 2e-10
        ai.wavelength = wl1
        q1, i1 = ai.integrate1d(data, 1000)
#         ai.reset()
        ai.wavelength = wl2
        q2, i2 = ai.integrate1d(data, 1000)
        dq = (abs(q1 - q2).max())
        di = (abs(i1 - i2).max())
#         print(dq)
        self.assertAlmostEqual(dq, 3.79, 2, "Q-scale difference should be around 3.8, got %s" % dq)


def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(TestBug170("test_bug170"))
    testsuite.addTest(TestBug211("test_quantile"))
    testsuite.addTest(TestBug232("test"))
    testsuite.addTest(TestBug174("test"))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())


