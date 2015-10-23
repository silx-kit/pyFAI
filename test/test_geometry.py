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

__doc__ = """tests for Jon's geometry changes
FIXME : make some tests that the functions do what is expected
"""
__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "23/10/2015"


import unittest
import os
import sys
import time
import numpy
if __name__ == '__main__':
    import pkgutil
    __path__ = pkgutil.extend_path([os.path.dirname(__file__)], "pyFAI.test")
from .utilstest import UtilsTest, getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]

from pyFAI import geometry
from pyFAI import AzimuthalIntegrator
import fabio


class TestSolidAngle(unittest.TestCase):
    """
    Test case for solid angle compared to Fit2D results

    Masked region have values set to 0 (not negative) and native mask from pilatus desactivated
    Detector Pilatus6M     PixelSize= 1.720e-04, 1.720e-04 m
    Wavelength= 1.072274e-10m
    SampleDetDist= 1.994993e-01m    PONI= 2.143248e-01, 2.133315e-01m    rot1=0.007823  rot2= 0.006716  rot3= -0.000000 rad
    DirectBeamDist= 199.510mm    Center: x=1231.226, y=1253.864 pix    Tilt=0.591 deg  tiltPlanRotation= 139.352 deg
    integration in 2theta between 0 and 56 deg in 1770 points
    """
    fit2dFile = '2548/powder_200_2_0001.chi'
    pilatusFile = '2549/powder_200_2_0001.cbf'
    ai = None
    fit2d = None

    def setUp(self):
        """Download files"""
        self.fit2dFile = UtilsTest.getimage(self.__class__.fit2dFile)
        self.pilatusFile = UtilsTest.getimage(self.__class__.pilatusFile)
        self.tth_fit2d, self.I_fit2d = numpy.loadtxt(self.fit2dFile, unpack=True)
        self.ai = AzimuthalIntegrator(dist=1.994993e-01,
                                      poni1=2.143248e-01,
                                      poni2=2.133315e-01,
                                      rot1=0.007823,
                                      rot2=0.006716,
                                      rot3=0,
                                      pixel1=172e-6,
                                      pixel2=172e-6)
        self.data = fabio.open(self.pilatusFile).data
        self.data[self.data < 0] = 0  # discard negative pixels

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.fit2dFile = self.pilatusFile = self.tth_fit2d = self.I_fit2d = self.ai = self.data = None

    def testSolidAngle(self):
        """
        This dataset goes up to 56deg, very good to test the solid angle correction
        any error will show off.
        fit2d makes correction in 1/cos^3(2th) (without tilt). pyFAI used to correct in 1/cos(2th)
        """
        tth, I_nogood = self.ai.integrate1d(self.data, 1770, unit="2th_deg", radial_range=[0, 56], method="splitBBox", correctSolidAngle=False)
        delta_tth = abs(tth - self.tth_fit2d).max()
        delta_I = abs(I_nogood - self.I_fit2d).max()
        I = abs(I_nogood - self.I_fit2d).mean()
        self.assert_(delta_tth < 1e-5, 'Error on 2th position: %s <1e-5' % delta_tth)
        self.assert_(delta_I > 100, 'Error on (wrong) I are large: %s >100' % delta_I)
        self.assert_(I > 2, 'Error on (wrong) I are large: %s >2' % I)
        tth, I_good = self.ai.integrate1d(self.data, 1770, unit="2th_deg", radial_range=[0, 56], method="splitBBox", correctSolidAngle=3)
        delta_tth = abs(tth - self.tth_fit2d).max()
        delta_I = abs(I_good - self.I_fit2d).max()
        I = abs(I_good - self.I_fit2d).mean()
        self.assert_(delta_tth < 1e-5, 'Error on 2th position: %s <1e-5' % delta_tth)
        self.assert_(delta_I < 5, 'Error on (good) I are small: %s <5' % delta_I)
        self.assert_(I < 0.05, 'Error on (good) I are small: %s <0.05' % I)


class TestBug88SolidAngle(unittest.TestCase):
    """
    Test case for solid angle where data got modified inplace.
    
    https://github.com/kif/pyFAI/issues/88
    """

    def testSolidAngle(self):
        img = numpy.ones((1000, 1000), dtype=numpy.float32)
        ai = pyFAI.AzimuthalIntegrator(dist=0.01, detector="Titan", wavelength=1e-10)
        t = ai.integrate1d(img, 1000, method="numpy")[1].max()
        f = ai.integrate1d(img, 1000, method="numpy", correctSolidAngle=False)[1].max()
        self.assertAlmostEqual(f, 1, 5, "uncorrected flat data are unchanged")
        self.assertNotAlmostEqual(f, t, 1, "corrected and uncorrected flat data are different")


class TestRecprocalSpacingSquarred(unittest.TestCase):
    """
    """
    def setUp(self):
        from pyFAI.detectors import Detector
        self.shape = (50, 49)
        size = (50, 60)
        det = Detector(*size)
        det.max_shape = self.shape
        self.geo = geometry.Geometry(detector=det, wavelength=1e-10)

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.geo = None
        self.size = None

    def test_center(self):
        rd2 = self.geo.rd2Array(self.shape)
        q = self.geo.qArray(self.shape)
        self.assert_(numpy.allclose(rd2, (q / (2 * numpy.pi)) ** 2), "center rd2 = (q/2pi)**2")

    def test_corner(self):
        rd2 = self.geo.cornerRd2Array(self.shape)[:, :, :, 0]
        q = self.geo.cornerQArray(self.shape)[:, :, :, 0]
        self.assert_(numpy.allclose(rd2, (q / (2 * numpy.pi)) ** 2), "corners rd2 = (q/2pi)**2")

    def test_delta(self):

        drd2a = self.geo.deltaRd2(self.shape)
        rd2 = self.geo.rd2Array(self.shape)
        rc = self.geo.cornerRd2Array(self.shape)[:, :, :, 0]
        drd2 = self.geo.deltaRd2(self.shape)
        self.assert_(numpy.allclose(drd2, drd2a, atol=1e-5), "delta rd2 = (q/2pi)**2, one formula with another")
        delta2 = abs(rc - numpy.atleast_3d(rd2)).max(axis=-1)
        self.assert_(numpy.allclose(drd2, delta2, atol=1e-5), "delta rd2 = (q/2pi)**2")


class ParameterisedTestCase(unittest.TestCase):
    """ TestCase classes that want to be parameterised should
        inherit from this class.
        From Eli Bendersky's website
        http://eli.thegreenplace.net/2011/08/02/python-unit-testing-parametrized-test-cases/
    """
    def __init__(self, methodName='runTest', param=None):
        super(ParameterisedTestCase, self).__init__(methodName)
        self.param = param

    @staticmethod
    def parameterise(testcase_klass, param=None):
        """ Create a suite containing all tests taken from the given
            subclass, passing them the parameter 'param'.
        """
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(testcase_klass)
        suite = unittest.TestSuite()
        for name in testnames:
            suite.addTest(testcase_klass(name, param=param))
        return suite


class TestGeometry(ParameterisedTestCase):

    def testGeometryFunctions(self):
        func, statargs, varargs, kwds, expectedFail = self.param
        kwds["pixel1"] = 1
        kwds["pixel2"] = 1
        g = geometry.Geometry(**kwds)
        g.wavelength = 1e-10
        t0 = time.time()
        oldret = getattr(g, func)(*statargs, path=varargs[0])
        t1 = time.time()
        newret = getattr(g, func)(*statargs, path=varargs[1])
        t2 = time.time()
        logger.debug("TIMINGS\t meth: %s t=%.3fs\t meth: %s t=%.3fs" % (varargs[0], t1 - t0, varargs[1], t2 - t1))
        maxDelta = abs(oldret - newret).max()
        msg = "geo=%s%s max delta=%.3f" % (g, os.linesep, maxDelta)
        if expectedFail:
            self.assertNotAlmostEquals(maxDelta, 0, 3, msg)
        else:
            self.assertAlmostEquals(maxDelta, 0, 3, msg)
        logger.info(msg)

size = 1024
d1, d2 = numpy.mgrid[-size:size:32, -size:size:32]

TESTCASES = [
 ("tth", (d1, d2), ("cos", "tan"), {'dist':1, 'rot1':0, 'rot2':0, 'rot3':0}, False),
 ("tth", (d1, d2), ("cos", "tan"), {'rot1':-1, 'rot2':1, 'rot3':1}, False),
 ("tth", (d1, d2), ("cos", "tan"), {'rot1':-.2, 'rot2':1, 'rot3':-.1}, False),
 ("tth", (d1, d2), ("cos", "tan"), {'rot1':-1, 'rot2':-.2, 'rot3':1}, False),
 ("tth", (d1, d2), ("cos", "tan"), {'rot1':1, 'rot2':5, 'rot3':.4}, False),
 ("tth", (d1, d2), ("cos", "tan"), {'rot1':-1.2, 'rot2':1, 'rot3':1}, False),
 ("tth", (d1, d2), ("cos", "tan"), {'dist':1e10, 'rot1':-2, 'rot2':2, 'rot3':1}, False),
 ("tth", (d1, d2), ("cos", "tan"), {'dist':1, 'rot1':3, 'rot2':0, 'rot3':0}, False),
 ("tth", (d1, d2), ("cos", "tan"), {'rot1':-1, 'rot2':1, 'rot3':3}, False),
 ("tth", (d1, d2), ("cos", "tan"), {'rot1':-.2, 'rot2':1, 'rot3':-.1}, False),
 ("tth", (d1, d2), ("cos", "tan"), {'rot1':-3, 'rot2':-.2, 'rot3':1}, False),
 ("tth", (d1, d2), ("cos", "tan"), {'rot1':1, 'rot2':5, 'rot3':.4}, False),
 ("tth", (d1, d2), ("cos", "tan"), {'rot1':-1.2, 'rot2':1.6, 'rot3':1}, False),
 ("tth", (d1, d2), ("cos", "tan"), {'dist':1e10, 'rot1':0, 'rot2':0, 'rot3':0}, False),
 ("tth", (d1, d2), ("tan", "cython"), {'dist':1, 'rot1':0, 'rot2':0, 'rot3':0}, False),
 ("tth", (d1, d2), ("tan", "cython"), {'rot1':-1, 'rot2':1, 'rot3':1}, False),
 ("tth", (d1, d2), ("tan", "cython"), {'rot1':-.2, 'rot2':1, 'rot3':-.1}, False),
 ("tth", (d1, d2), ("tan", "cython"), {'rot1':-1, 'rot2':-.2, 'rot3':1}, False),
 ("tth", (d1, d2), ("tan", "cython"), {'rot1':1, 'rot2':5, 'rot3':.4}, False),
 ("tth", (d1, d2), ("tan", "cython"), {'rot1':-1.2, 'rot2':1, 'rot3':1}, False),
 ("tth", (d1, d2), ("tan", "cython"), {'dist':1e10, 'rot1':-2, 'rot2':2, 'rot3':1}, False),
 ("tth", (d1, d2), ("tan", "cython"), {'dist':1, 'rot1':3, 'rot2':0, 'rot3':0}, False),
 ("tth", (d1, d2), ("tan", "cython"), {'rot1':-1, 'rot2':1, 'rot3':3}, False),
 ("tth", (d1, d2), ("tan", "cython"), {'rot1':-.2, 'rot2':1, 'rot3':-.1}, False),
 ("tth", (d1, d2), ("tan", "cython"), {'rot1':-3, 'rot2':-.2, 'rot3':1}, False),
 ("tth", (d1, d2), ("tan", "cython"), {'rot1':1, 'rot2':5, 'rot3':.4}, False),
 ("tth", (d1, d2), ("tan", "cython"), {'rot1':-1.2, 'rot2':1.6, 'rot3':1}, False),
 ("tth", (d1, d2), ("tan", "cython"), {'dist':1e10, 'rot1':0, 'rot2':0, 'rot3':0}, False),

 ("qFunction", (d1, d2), ("cython", "tan"), {'dist':1, 'rot1':0, 'rot2':0, 'rot3':0}, False),
 ("qFunction", (d1, d2), ("cython", "tan"), {'rot1':-1, 'rot2':1, 'rot3':1}, False),
 ("qFunction", (d1, d2), ("cython", "tan"), {'rot1':-.2, 'rot2':1, 'rot3':-.1}, False),
 ("qFunction", (d1, d2), ("cython", "tan"), {'rot1':-1, 'rot2':-.2, 'rot3':1}, False),
 ("qFunction", (d1, d2), ("cython", "tan"), {'rot1':1, 'rot2':5, 'rot3':.4}, False),
 ("qFunction", (d1, d2), ("cython", "tan"), {'rot1':-1.2, 'rot2':1, 'rot3':1}, False),
 ("qFunction", (d1, d2), ("cython", "tan"), {'dist':1e10, 'rot1':-2, 'rot2':2, 'rot3':1}, False),
 ("qFunction", (d1, d2), ("cython", "tan"), {'dist':1, 'rot1':3, 'rot2':0, 'rot3':0}, False),
 ("qFunction", (d1, d2), ("cython", "tan"), {'rot1':-1, 'rot2':1, 'rot3':3}, False),
 ("qFunction", (d1, d2), ("cython", "tan"), {'rot1':-.2, 'rot2':1, 'rot3':-.1}, False),
 ("qFunction", (d1, d2), ("cython", "tan"), {'rot1':-3, 'rot2':-.2, 'rot3':1}, False),
 ("qFunction", (d1, d2), ("cython", "tan"), {'rot1':1, 'rot2':5, 'rot3':.4}, False),
 ("qFunction", (d1, d2), ("cython", "tan"), {'rot1':-1.2, 'rot2':1.6, 'rot3':1}, False),
 ("qFunction", (d1, d2), ("cython", "tan"), {'dist':1e10, 'rot1':0, 'rot2':0, 'rot3':0}, False),

 ("rFunction", (d1, d2), ("cython", "numpy"), {'dist':1, 'rot1':0, 'rot2':0, 'rot3':0}, False),
 ("rFunction", (d1, d2), ("cython", "numpy"), {'rot1':-1, 'rot2':1, 'rot3':1}, False),
 ("rFunction", (d1, d2), ("cython", "numpy"), {'rot1':-.2, 'rot2':1, 'rot3':-.1}, False),
 ("rFunction", (d1, d2), ("cython", "numpy"), {'rot1':-1, 'rot2':-.2, 'rot3':1}, False),
 ("rFunction", (d1, d2), ("cython", "numpy"), {'rot1':1, 'rot2':5, 'rot3':.4}, False),
 ("rFunction", (d1, d2), ("cython", "numpy"), {'rot1':-1.2, 'rot2':1, 'rot3':1}, False),
 ("rFunction", (d1, d2), ("cython", "numpy"), {'dist':1e10, 'rot1':-2, 'rot2':2, 'rot3':1}, False),
 ("rFunction", (d1, d2), ("cython", "numpy"), {'dist':1, 'rot1':3, 'rot2':0, 'rot3':0}, False),
 ("rFunction", (d1, d2), ("cython", "numpy"), {'rot1':-1, 'rot2':1, 'rot3':3}, False),
 ("rFunction", (d1, d2), ("cython", "numpy"), {'rot1':-.2, 'rot2':1, 'rot3':-.1}, False),
 ("rFunction", (d1, d2), ("cython", "numpy"), {'rot1':-3, 'rot2':-.2, 'rot3':1}, False),
 ("rFunction", (d1, d2), ("cython", "numpy"), {'rot1':1, 'rot2':5, 'rot3':.4}, False),
 ("rFunction", (d1, d2), ("cython", "numpy"), {'rot1':-1.2, 'rot2':1.6, 'rot3':1}, False),
 ("rFunction", (d1, d2), ("cython", "numpy"), {'dist':1e10, 'rot1':0, 'rot2':0, 'rot3':0}, False),
 ]


def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(TestSolidAngle("testSolidAngle"))
    testsuite.addTest(TestBug88SolidAngle("testSolidAngle"))
    testsuite.addTest(TestRecprocalSpacingSquarred("test_center"))
    testsuite.addTest(TestRecprocalSpacingSquarred("test_corner"))
    testsuite.addTest(TestRecprocalSpacingSquarred("test_delta"))

    for param in TESTCASES:
        testsuite.addTest(ParameterisedTestCase.parameterise(
                TestGeometry, param))

    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
