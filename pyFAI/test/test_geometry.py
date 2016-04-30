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
__date__ = "30/04/2016"


import unittest
import os
import sys
import random
import time
import numpy
import itertools
from .utilstest import UtilsTest, getLogger, ParameterisedTestCase
logger = getLogger(__file__)

from .. import geometry
from .. import AzimuthalIntegrator
from .. import units
from ..detectors import detector_factory
import fabio

if sys.platform == "win32":
    timer = time.clock
else:
    timer = time.time


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

    def test_nonflat_center(self):
        """
        Test non flat detector cos(incidence) to be 1 (+/- 1%) when centered.

        Aarhus is a curved detector of radius 0.3m
        """
        aarhus = detector_factory("Aarhus")
        aarhus.binning = (10, 10)
        ai = AzimuthalIntegrator(aarhus.radius, detector=aarhus)
        cosa = numpy.fromfunction(ai.cosIncidance,
                                  aarhus.shape, dtype=numpy.float32)
        maxi = cosa.max()
        mini = cosa.min()
        self.assert_(maxi <= 1.0, 'Cos incidence is %s <=1.0' % maxi)
        self.assert_(mini > 0.99, 'Cos solid angle is %s >0.99' % mini)

    def test_nonflat_outside(self):
        """
        Test non flat detector cos(incidence) to be !=1 when off-centered.

        Aarhus is a curved detector of radius 0.3m, here we offset of 50%
        """
        aarhus = detector_factory("Aarhus")
        aarhus.binning = (10, 10)
        ai = AzimuthalIntegrator(aarhus.radius * 1.5, detector=aarhus)
        cosa = numpy.fromfunction(ai.cosIncidance,
                                  aarhus.shape, dtype=numpy.float32)
        maxi = cosa.max()
        mini = cosa.min()
        self.assert_(maxi <= 1.0, 'Cos incidence is %s <=1.0' % maxi)
        self.assert_(maxi > 0.99, 'Cos incidence max is %s >0.99' % maxi)
        self.assert_(mini < 0.92, 'Cos solid angle min is %s <0.92' % mini)

    def test_nonflat_inside(self):
        """
        Test non flat detector cos(incidence) to be !=1 when off-centered.

        Aarhus is a curved detector of radius 0.3m, here we offset of 50%
        """
        aarhus = detector_factory("Aarhus")
        aarhus.binning = (10, 10)
        ai = AzimuthalIntegrator(aarhus.radius * 0.5, detector=aarhus)
        cosa = numpy.fromfunction(ai.cosIncidance,
                                  aarhus.shape, dtype=numpy.float32)
        maxi = cosa.max()
        mini = cosa.min()
        self.assert_(maxi <= 1.0, 'Cos incidence is %s <=1.0' % maxi)
        self.assert_(maxi > 0.99, 'Cos incidence max is %s >0.99' % maxi)
        self.assert_(mini < 0.87, 'Cos solid angle min is %s <0.86' % mini)


class TestBug88SolidAngle(unittest.TestCase):
    """
    Test case for solid angle where data got modified inplace.

    https://github.com/kif/pyFAI/issues/88
    """

    def testSolidAngle(self):
        img = numpy.ones((1000, 1000), dtype=numpy.float32)
        ai = AzimuthalIntegrator(dist=0.01, detector="Titan", wavelength=1e-10)
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
        det = Detector(*size, max_shape=self.shape)
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
        delta = rd2 - (q / (2 * numpy.pi)) ** 2
        self.assert_(numpy.allclose(rd2, (q / (2 * numpy.pi)) ** 2), "corners rd2 = (q/2pi)**2, delat=%s" % delta)

    def test_delta(self):

        drd2a = self.geo.deltaRd2(self.shape)
        rd2 = self.geo.rd2Array(self.shape)
        rc = self.geo.cornerRd2Array(self.shape)[:, :, :, 0]
        drd2 = self.geo.deltaRd2(self.shape)
        self.assert_(numpy.allclose(drd2, drd2a, atol=1e-5), "delta rd2 = (q/2pi)**2, one formula with another")
        delta2 = abs(rc - numpy.atleast_3d(rd2)).max(axis=-1)
        self.assert_(numpy.allclose(drd2, delta2, atol=1e-5), "delta rd2 = (q/2pi)**2")


class ParamFastPath(ParameterisedTestCase):
    """Test the consistency of the geometry calculation using the Python and the
    Cython path.
    """
    detectors = ("Pilatus300k", "Xpad_flat")
    number_of_geometries = 2
    epsilon = 3e-7
    geometries = []
    for i in range(number_of_geometries):
        geo = {"dist": 0.01 + random.random(),
               "poni1": random.random() - 0.5,
               "poni2": random.random() - 0.5,
               "rot1": random.random() - 0.5,
               "rot2": random.random() - 0.5,
               "rot3": random.random() - 0.5,
               "wavelength": 1e-10}
# Provides atol = 1.08e-5
#         geo = {"dist": 0.037759112584709535,
#                "poni1": 0.005490358659182459,
#                "poni2": 0.06625690275821605,
#                "rot1": 0.20918568578536278,
#                "rot2": 0.42161920581114365,
#                "rot3": 0.38784171093239983,
#                "wavelength": 1e-10}
# Provides atol = 2.8e-5
#         geo = {'dist': 0.48459003559204783,
#                'poni2':-0.15784154756282065,
#                'poni1': 0.02783657100374448,
#                'rot3':-0.2901541134116695,
#                'rot1':-0.3927992588689394,
#                'rot2': 0.148115949280184,
#                "wavelength": 1e-10}
# Provides atol = 3.67761e-05
#         geo = {'poni1':-0.22055143279015976, 'poni2':-0.11124668733292842, 'rot1':-0.18105235367380956, 'wavelength': 1e-10, 'rot3': 0.2146474866836957, 'rot2': 0.36581323339171257, 'detector': 'Pilatus300k', 'dist': 0.7350926443000882}
        for det in detectors:
            dico = geo.copy()
            dico["detector"] = det
            geometries.append(dico)
    dunits = dict((u.REPR.split("_")[0], u) for u in units.RADIAL_UNITS)
    TESTSPACE = itertools.product(geometries, dunits.values())

    def test_corner_array(self):
        """test pyFAI.geometry.corner_array with full detectors
        """
        data, space = self.param
        geo = geometry.Geometry(**data)
        t00 = timer()
        py_res = geo.corner_array(unit=space, use_cython=False)
        t01 = timer()
        geo.reset()
        t10 = timer()
        cy_res = geo.corner_array(unit=space, use_cython=True)
        t11 = timer()
        delta = abs(py_res - cy_res).max()
        logger.info("TIMINGS\t meth: %s %s Python: %.3fs, Cython: %.3fs\t x%.3f\t delta:%s",
                    space, data["detector"], t01 - t00, t11 - t10, (t01 - t00) / (t11 - t10), delta)
        self.assert_(numpy.allclose(py_res, cy_res, atol=2.9e-5), "data:%s, space: %s delta: %s" % (data, space, delta))

    def test_XYZ(self):
        """Test the calc_pos_zyx with full detectors"""
        kwds = self.param
        geo = geometry.Geometry(**kwds)
        t0 = timer()
        py_res = geo.calc_pos_zyx(corners=True, use_cython=False)
        t1 = timer()
        cy_res = geo.calc_pos_zyx(corners=True, use_cython=True)
        t2 = timer()
        delta = numpy.array([abs(py - cy).max() for py, cy in zip(py_res, cy_res)])
        logger.info("TIMINGS\t meth: calc_pos_zyx %s, corner=True python t=%.3fs\t cython: t=%.3fs \t x%.3f delta %s", kwds["detector"], t1 - t0, t2 - t1, (t1 - t0) / (t2 - t1), delta)
        msg = "delta=%s<%s, geo= \n%s" % (delta, self.epsilon, geo)
        self.assert_(numpy.alltrue(delta.max() < self.epsilon), msg)
        logger.debug(msg)


class ParamTestGeometry(ParameterisedTestCase):
    size = 1024
    d1, d2 = numpy.mgrid[-size:size:32, -size:size:32]
    functions = [("tth", ("cos", "tan")),
                 ("tth", ("tan", "cython")),
                 ("tth", ("cos", "tan")),
                 ("tth", ("tan", "cython")),
                 ("qFunction", ("numpy", "cython")),
                 ("rFunction", ("numpy", "cython"))]
    pixels = {"pixel1": 1,
              "pixel2": 1,
              "wavelength": 1e-10}
    geometries = [{'dist': 1, 'rot1': 0, 'rot2': 0, 'rot3': 0},
                  {'dist': 1, 'rot1':-1, 'rot2': 1, 'rot3': 1},
                  {'dist': 1, 'rot1':-.2, 'rot2': 1, 'rot3':-.1},
                  {'dist': 1, 'rot1':-1, 'rot2':-.2, 'rot3': 1},
                  {'dist': 1, 'rot1': 1, 'rot2': 5, 'rot3': .4},
                  {'dist': 1, 'rot1':-1.2, 'rot2': 1, 'rot3': 1},
                  {'dist': 1e10, 'rot1':-2, 'rot2': 2, 'rot3': 1},
                  ]
    for g in geometries:
        g.update(pixels)

    TESTCASES_FUNCT = [(k[0], k[1], g) for k, g in itertools.product(functions, geometries)]
    TESTCASES_XYZ = itertools.product((False,), geometries)

    def test_geometry_functions(self):
        "test functions like tth, qFunct, rfunction, ... fake detectors"
        func, varargs, kwds = self.param
        geo = geometry.Geometry(**kwds)
        t0 = timer()
        oldret = getattr(geo, func)(self.d1, self.d2, path=varargs[0])
        t1 = timer()
        newret = getattr(geo, func)(self.d1, self.d2, path=varargs[1])
        t2 = timer()
        delta = abs(oldret - newret).max()
        logger.info("TIMINGS\t %s meth: %s %.3fs\t meth: %s %.3fs, x%.3f delta %s", func, varargs[0], t1 - t0, varargs[1], t2 - t1, (t1 - t0) / (t2 - t1), delta)
        msg = "func: %s max delta=%.3f, geo:%s" % (func, delta, geo)
        self.assertAlmostEquals(delta, 0, 3, msg)
        logger.debug(msg)

    def test_XYZ(self):
        """Test the calc_pos_zyx with fake detectors"""
        corners, kwds = self.param
        geo = geometry.Geometry(**kwds)
        t0 = timer()
        py_res = geo.calc_pos_zyx(None, self.d1, self.d2, corners=corners, use_cython=False)
        t1 = timer()
        cy_res = geo.calc_pos_zyx(None, self.d1, self.d2, corners=corners, use_cython=True)
        t2 = timer()
        delta = numpy.array([abs(py - cy).max() for py, cy in zip(py_res, cy_res)])
        logger.info("TIMINGS\t meth: calc_pos_zyx, corner=%s python t=%.3fs\t cython: t=%.3fs\t x%.3f delta %s", corners, t1 - t0, t2 - t1, (t1 - t0) / (t2 - t1), delta)
        msg = "delta=%s, geo= \n%s" % (delta, geo)
        self.assert_(numpy.allclose(numpy.vstack(cy_res), numpy.vstack(py_res)), msg)
        logger.debug(msg)


def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(TestSolidAngle("testSolidAngle"))
    testsuite.addTest(TestSolidAngle("test_nonflat_center"))
    testsuite.addTest(TestSolidAngle("test_nonflat_outside"))
    testsuite.addTest(TestSolidAngle("test_nonflat_inside"))
    testsuite.addTest(TestBug88SolidAngle("testSolidAngle"))
    testsuite.addTest(TestRecprocalSpacingSquarred("test_center"))
    testsuite.addTest(TestRecprocalSpacingSquarred("test_corner"))
    testsuite.addTest(TestRecprocalSpacingSquarred("test_delta"))

    for param in ParamTestGeometry.TESTCASES_FUNCT:
        testsuite.addTest(ParameterisedTestCase.parameterise(ParamTestGeometry, "test_geometry_functions", param))
    for param in ParamTestGeometry.TESTCASES_XYZ:
        testsuite.addTest(ParameterisedTestCase.parameterise(ParamTestGeometry, "test_XYZ", param))

    for param in ParamFastPath.geometries:
        testsuite.addTest(ParameterisedTestCase.parameterise(ParamFastPath, "test_XYZ", param))
    for param in ParamFastPath.TESTSPACE:
        testsuite.addTest(ParameterisedTestCase.parameterise(ParamFastPath, "test_corner_array", param))

    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
