#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2024 European Synchrotron Radiation Facility, Grenoble, France
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

"""tests for Jon's geometry changes
FIXME : make some tests that the functions do what is expected
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "09/10/2024"

import unittest
import random
import time
import numpy
import itertools
import logging
logger = logging.getLogger(__name__)
import os.path
import json
import fabio
from . import utilstest
from ..io.ponifile import PoniFile
from .. import geometry
from ..integrator.azimuthal import AzimuthalIntegrator
from .. import units
from ..detectors import detector_factory
from ..third_party import transformations
from .utilstest import UtilsTest
from ..utils.mathutil import allclose_mod
from ..geometry.crystfel import build_geometry, parse_crystfel_geom


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

    @unittest.skipIf(utilstest.UtilsTest.low_mem, "skipping test using >400M")
    def testSolidAngle(self):
        """
        This dataset goes up to 56deg, very good to test the solid angle correction
        any error will show off.
        """
        fit2dFile = 'powder_200_2_0001.chi'
        pilatusFile = 'powder_200_2_0001.cbf'

        fit2dFile = utilstest.UtilsTest.getimage(fit2dFile)
        pilatusFile = utilstest.UtilsTest.getimage(pilatusFile)
        tth_fit2d, I_fit2d = numpy.loadtxt(fit2dFile, unpack=True)
        ai = AzimuthalIntegrator(dist=1.994993e-01,
                                 poni1=2.143248e-01,
                                 poni2=2.133315e-01,
                                 rot1=0.007823,
                                 rot2=0.006716,
                                 rot3=0,
                                 pixel1=172e-6,
                                 pixel2=172e-6)
        with fabio.open(pilatusFile) as fimg:
            data = fimg.data
        data[data < 0] = 0  # discard negative pixels

        method = ("bbox", "histogram", "cython")
        tth, I_nogood = ai.integrate1d_ng(data, 1770, unit="2th_deg", radial_range=[0, 56], method=method, correctSolidAngle=False)
        delta_tth = abs(tth - tth_fit2d).max()
        delta_I = abs(I_nogood - I_fit2d).max()
        mean_I = abs(I_nogood - I_fit2d).mean()
        self.assertLess(delta_tth, 1e-5, 'Error on 2th position: %s <1e-5' % delta_tth)
        self.assertGreater(delta_I, 100, 'Error on (wrong) I are large: %s >100' % delta_I)
        self.assertGreater(mean_I, 2, 'Error on (wrong) I are large: %s >2' % mean_I)

        tth, I_good = ai.integrate1d_ng(data, 1770, unit="2th_deg", radial_range=[0, 56], method=method, correctSolidAngle=3)
        delta_tth = abs(tth - tth_fit2d).max()
        delta_I = abs(I_good - I_fit2d).max()
        mean_I = abs(I_good - I_fit2d).mean()
        self.assertLess(delta_tth, 1e-5, 'Error on 2th position: %s <1e-5' % delta_tth)
        self.assertLess(delta_I, 5, 'Error on (good) I are small: %s <5' % delta_I)
        self.assertLess(mean_I, 0.05, 'Error on (good) I are small: %s <0.05' % mean_I)
        ai.reset()

    def test_nonflat_center(self):
        """
        Test non flat detector cos(incidence) to be 1 (+/- 1%) when centered.

        Aarhus is a curved detector of radius 0.3m
        """
        aarhus = detector_factory("Aarhus")
        aarhus.binning = (10, 10)
        ai = AzimuthalIntegrator(aarhus.radius, detector=aarhus)
        for path in ("cython", "numexpr", "numpy"):
            cosa = numpy.fromfunction(ai.cos_incidence,
                                      aarhus.shape, dtype=numpy.float32,
                                      path=path)
            maxi = cosa.max()
            mini = cosa.min()
            self.assertLessEqual(maxi, 1.0, f'path:{path} Cos incidence is {maxi} <=1.0')
            self.assertTrue(mini > 0.99, f'path:{path} Cos incidence is {mini} >0.99')

    def test_nonflat_outside(self):
        """
        Test non flat detector cos(incidence) to be !=1 when off-centered.

        Aarhus is a curved detector of radius 0.3m, here we offset of 50%
        """
        aarhus = detector_factory("Aarhus")
        aarhus.binning = (10, 10)
        ai = AzimuthalIntegrator(aarhus.radius * 1.5, detector=aarhus)
        for path in ("cython", "numexpr", "numpy"):
            cosa = numpy.fromfunction(ai.cos_incidence,
                                      aarhus.shape, dtype=numpy.float32,
                                      path=path)
            maxi = cosa.max()
            mini = cosa.min()
            self.assertLessEqual(maxi, 1.0, f'path: {path} Cos incidence is {maxi} <=1.0')
            self.assertGreater(maxi, 0.99, f'path: {path} Cos incidence max is {maxi} >0.99')
            self.assertLess(mini, 0.92, f'path: {path} Cos incidence min is {mini} <0.92')

    def test_nonflat_inside(self):
        """
        Test non flat detector cos(incidence) to be !=1 when off-centered.

        Aarhus is a curved detector of radius 0.3m, here we offset of 50%
        """
        aarhus = detector_factory("Aarhus")
        aarhus.binning = (10, 10)
        ai = AzimuthalIntegrator(aarhus.radius * 0.5, detector=aarhus)
        for path in ("cython", "numexpr", "numpy"):
            cosa = numpy.fromfunction(ai.cos_incidence,
                                      aarhus.shape, dtype=numpy.float32,
                                      path=path)
            maxi = cosa.max()
            mini = cosa.min()
            self.assertLessEqual(maxi, 1.0, f'path: {path} Cos incidence is {maxi} <=1.0')
            self.assertGreater(maxi, 0.99, f'path: {path} Cos incidence max is {maxi} >0.99')
            self.assertLess(mini, 0.87, f'path: {path} Cos incidence min is {mini} <0.86')

    def test_flat(self):
        """test sine and cosine for the incidence angle
        """
        pilatus = detector_factory("Imxpad S10")
        ai = AzimuthalIntegrator(0.1, detector=pilatus)
        for path in ("cython", "numexpr", "numpy"):
            cosa = numpy.fromfunction(ai.cos_incidence,
                                      pilatus.shape, dtype=numpy.float64,
                                      path=path)
            sina = numpy.fromfunction(ai.sin_incidence,
                                      pilatus.shape, dtype=numpy.float64,
                                      path=path)
            one = cosa * cosa + sina * sina
            self.assertLessEqual(one.max() - 1.0, 1e-10, f"path: {path} cos2+sin2<=1")
            self.assertGreater(one.min() - 1.0, -1e-10, f"path: {path} cos2+sin2>0.99")


class TestBug88SolidAngle(unittest.TestCase):
    """
    Test case for solid angle where data got modified inplace.

    https://github.com/silx-kit/pyFAI/issues/88
    """

    def testSolidAngle(self):
        method = ("no", "histogram", "python")
        ai = AzimuthalIntegrator(dist=0.001, detector="Imxpad S10", wavelength=1e-10)
        img = numpy.ones(ai.detector.shape, dtype=numpy.float32)
        r0 = ai.integrate1d_ng(img, 100, method=method)
        t = r0[1].max()
        r1 = ai.integrate1d_ng(img, 100, method=method, correctSolidAngle=False)
        f = r1[1].max()
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
        self.former_loglevel = geometry.logger.level
        geometry.logger.setLevel(logging.ERROR)

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.geo = None
        self.size = None
        geometry.logger.setLevel(self.former_loglevel)

    def test_center(self):
        rd2 = self.geo.rd2Array(self.shape)
        q = self.geo.qArray(self.shape)
        self.assertTrue(numpy.allclose(rd2, (q / (2 * numpy.pi)) ** 2), "center rd2 = (q/2pi)**2")

    def test_corner(self):
        rd2 = self.geo.corner_array(self.shape, unit=units.RecD2_NM, scale=False)[:,:,:, 0]
        q = self.geo.corner_array(self.shape, unit=units.Q, use_cython=False, scale=False)[:,:,:, 0]
        delta = rd2 - (q / (2 * numpy.pi)) ** 2
        self.assertTrue(numpy.allclose(rd2, (q / (2 * numpy.pi)) ** 2), "corners rd2 = (q/2pi)**2, delat=%s" % delta)

    def test_delta(self):
        drd2a = self.geo.deltaRd2(self.shape)
        rd2 = self.geo.rd2Array(self.shape)
        rc = self.geo.corner_array(self.shape, unit=units.RecD2_NM, scale=False)[:,:,:, 0]
        drd2 = self.geo.deltaRd2(self.shape)
        self.assertTrue(numpy.allclose(drd2, drd2a, atol=1e-5), "delta rd2 = (q/2pi)**2, one formula with another")
        delta2 = abs(rc - numpy.atleast_3d(rd2)).max(axis=-1)
        self.assertTrue(numpy.allclose(drd2, delta2, atol=1e-5), "delta rd2 = (q/2pi)**2")


class TestFastPath(utilstest.ParametricTestCase):
    """Test the consistency of the geometry calculation using the Python and the
    Cython path.
    """
    EPSILON = 3e-7
    EPSILON_R = 1e-5
    EPSILON_A = 1e-5
    matrices = None
    geometries = None
    quaternions = None

    @classmethod
    def setUpClass(cls):
        super(TestFastPath, cls).setUpClass()
        cls.calc_geometries()

    @classmethod
    def tearDownClass(cls):
        super(TestFastPath, cls).tearDownClass()
        cls.matrices = None
        cls.geometries = None
        cls.quaternions = None

    def setUp(self):
        utilstest.ParametricTestCase.setUp(self)
        self.former_loglevel = geometry.logger.level
        geometry.logger.setLevel(logging.ERROR)

    def tearDown(self):
        geometry.logger.setLevel(self.former_loglevel)
        utilstest.ParametricTestCase.setUp(self)

    @classmethod
    def calc_geometries(cls):
        detectors = ("Pilatus100k", "ImXPadS10")
        number_of_geometries = 2

        # Here is a set of pathological cases ...
        geometries = [
            # Provides atol = 1.08e-5
            {"dist": 0.037759112584709535, "poni1": 0.005490358659182459, "poni2": 0.06625690275821605,
             "rot1": 0.20918568578536278, "rot2": 0.42161920581114365, "rot3": 0.38784171093239983,
             "wavelength": 1e-10, 'detector': 'Pilatus100k', "orientation":3},
            # Provides atol = 2.8e-5
            {'dist': 0.48459003559204783, 'poni2':-0.15784154756282065, 'poni1': 0.02783657100374448,
             'rot3':-0.2901541134116695, 'rot1':-0.3927992588689394, 'rot2': 0.148115949280184,
             "wavelength": 1e-10, 'detector': 'Pilatus100k', "orientation":3},
            # Provides atol = 3.67761e-05
            {'poni1':-0.22055143279015976, 'poni2':-0.11124668733292842, 'rot1':-0.18105235367380956,
             'wavelength': 1e-10, 'rot3': 0.2146474866836957, 'rot2': 0.36581323339171257,
             'detector': 'Pilatus100k', 'dist': 0.7350926443000882, "orientation":3},
            # Provides atol = 4.94719e-05
            {'poni2': 0.1010652698401574, 'rot3':-0.30578860159890153, 'rot1': 0.46240992613529186,
             'wavelength': 1e-10, 'detector': 'Pilatus300k', 'rot2':-0.027476969196682077,
             'dist': 0.04711960678381288, 'poni1': 0.012745759325719641, "orientation":3},
            # atol=2pi
            {'poni1': 0.07803878450256929, 'poni2': 0.2601779472529494, 'rot1':-0.33177239820033455,
             'wavelength': 1e-10, 'rot3': 0.2928945825578625, 'rot2': 0.2762729953307118,
             'detector': 'Pilatus100k', 'dist': 0.43544642285972124, "orientation":3},
            {'wavelength': 1e-10, 'dist': 0.13655542730645986, 'rot1':-0.16145635108891077,
             'poni1': 0.16271587645146157, 'rot2':-0.443426307059295, 'rot3': 0.40517456402269536,
             'poni2': 0.05248001026597382, 'detector': 'Pilatus100k', "orientation":3}
        ]

        matrices = [[[ 0.84465919, -0.29127499, -0.44912107], [ 0.34507215, 0.93768707, 0.04084325], [ 0.4092384 , -0.1894778 , 0.89253689]],
                    [[ 0.94770834, 0.21018393, -0.24014914], [-0.28296736, 0.9013857 , -0.3277702 ], [ 0.14757497, 0.37858492, 0.91372594]],
                    [[ 0.91240314, -0.27245408, -0.30543295], [ 0.19890928, 0.94736169, -0.25088028], [ 0.35770884, 0.16815051, 0.91856943]],
                    [[ 0.95324989, 0.25774197, 0.15774577], [-0.30093165, 0.85715139, 0.41800912], [-0.02747351, -0.44593785, 0.89464219]],
                    [[ 0.92110587, -0.35804283, -0.1528702 ], [ 0.27777593, 0.87954879, -0.38630875], [ 0.27277188, 0.3133676 , 0.90961324]],
                    [[ 0.83015106, -0.32566669, 0.45253776], [ 0.35605693, 0.93426745, 0.01917795], [-0.42903692, 0.14520861, 0.891539  ]]
                ]
        quaternions = [[ 0.95849924, -0.06007335, -0.2238811 , 0.16597487],
                       [ 0.96989948, 0.18206916, -0.09993925, -0.12711402],
                       [ 0.97189689, 0.10778684, -0.17057926, 0.1212483 ],
                       [ 0.96242447, -0.22441942, 0.04811268, -0.14512142],
                       [ 0.96310279, 0.18162037, -0.11048719, 0.16504437],
                       [ 0.95602792, 0.03295684, 0.23053058, 0.1782698 ]
                       ]

        for _ in range(number_of_geometries):
            random.seed(0)
            geo = {"dist": 0.01 + random.random(),
                   "poni1": random.random() - 0.5,
                   "poni2": random.random() - 0.5,
                   "rot1": (random.random() - 0.5) * numpy.pi,
                   "rot2": (random.random() - 0.5) * numpy.pi,
                   "rot3": (random.random() - 0.5) * numpy.pi,
                   "wavelength": 1e-10,
                   "orientation":3}

            for det in detectors:
                dico = geo.copy()
                dico["detector"] = det
                geometries.append(dico)
                q = transformations.quaternion_from_euler(-dico["rot1"], -dico["rot2"], dico["rot3"], axes="sxyz")
                quaternions.append(q)
                matrices.append(transformations.quaternion_matrix(q)[:3,:3])
        cls.geometries = geometries
        cls.quaternions = quaternions
        cls.matrices = matrices

    @classmethod
    def get_geometries(cls, what="geometries"):
        if what == "geometries":
            return cls.geometries
        elif what == "quaternions":
            return cls.quaternions
        elif what == "matrices":
            return cls.matrices

    def test_corner_array(self):
        """Test pyFAI.geometry.corner_array with full detectors
        """
        geometries = self.get_geometries()
        count_a = 17
        dunits = dict((u.split("_")[0], v) for u, v in units.RADIAL_UNITS.items())
        params = itertools.product(geometries, dunits.values())
        for data, space in params:
            with self.subTest(data=data, space=space):
                geo = geometry.Geometry(**data)
                t00 = time.perf_counter()
                py_res = geo.corner_array(unit=space, use_cython=False, scale=False)
                t01 = time.perf_counter()
                geo.reset()
                t10 = time.perf_counter()
                cy_res = geo.corner_array(unit=space, use_cython=True, scale=False)
                t11 = time.perf_counter()
                delta = abs(py_res - cy_res)
                # We expect precision on radial position
                delta_r = delta[..., 0].max()
                # issue with numerical stability of azimuthal position due to arctan(y,x)
                cnt_delta_a = (delta[..., 1] > self.EPSILON_A).sum()
                logger.debug("TIMINGS\t meth: %s %s Python: %.3fs, Cython: %.3fs\t x%.3f\t delta_r:%s",
                             space, data["detector"], t01 - t00, t11 - t10, (t01 - t00) / numpy.float64(t11 - t10), delta)
                self.assertLess(delta_r, self.EPSILON_R, "data=%s, space='%s' delta_r: %s" % (data, space, delta_r))
                self.assertLess(cnt_delta_a, count_a, "data:%s, space: %s cnt_delta_a: %s" % (data, space, cnt_delta_a))

    def test_XYZ(self):
        """Test the calc_pos_zyx with full detectors"""
        geometries = self.get_geometries()
        for geometryParams in geometries:
            with self.subTest(geometry=geometry):
                geo = geometry.Geometry(**geometryParams)
                t0 = time.perf_counter()
                py_res = geo.calc_pos_zyx(corners=True, use_cython=False)
                t1 = time.perf_counter()
                cy_res = geo.calc_pos_zyx(corners=True, use_cython=True)
                t2 = time.perf_counter()
                delta = numpy.array([abs(py - cy).max() for py, cy in zip(py_res, cy_res)])
                logger.debug("TIMINGS\t meth: calc_pos_zyx %s, corner=True python t=%.3fs\t cython: t=%.3fs \t x%.3f delta %s",
                             geometryParams["detector"], t1 - t0, t2 - t1, (t1 - t0) / numpy.float64(t2 - t1), delta)
                msg = "delta=%s<%s, geo= \n%s" % (delta, self.EPSILON, geo)
                self.assertTrue(numpy.all(delta.max() < self.EPSILON), msg)
                logger.debug(msg)

    def test_deltachi(self):
        """Test the deltaChi"""
        geometries = self.get_geometries()
        for geometryParams in geometries:
            with self.subTest(geometry=geometryParams):
                geo = geometry.Geometry(**geometryParams)
                t0 = time.perf_counter()
                py_res = geo.deltaChi(use_cython=False)
                # t1 = time.perf_counter()
                geo.reset()
                t1 = time.perf_counter()
                cy_res = geo.deltaChi(use_cython=True)
                t2 = time.perf_counter()
                delta = numpy.array([abs(py - cy).max() for py, cy in zip(py_res, cy_res)])
                logger.debug("TIMINGS\t meth: deltaChi %s python t=%.3fs\t cython: t=%.3fs \t x%.3f delta %s",
                             geometryParams["detector"], t1 - t0, t2 - t1, (t1 - t0) / numpy.float64(t2 - t1), delta)
                msg = "delta=%s<%s, geo= \n%s" % (delta, self.EPSILON, geo)
                self.assertTrue(numpy.all(delta.max() < self.EPSILON), msg)
                logger.debug(msg)

    def test_quaternions(self):
        "test the various geometry transformation in quaternions and rotation matrices"
        geometries = self.get_geometries()
        quaternions = self.get_geometries("quaternions")
        matrices = self.get_geometries("matrices")
        self.assertEqual(len(geometries), len(quaternions), "length is the same")
        self.assertEqual(len(geometries), len(matrices), "length is the same")
        for kwds, quat, mat in zip(geometries, quaternions, matrices):
            geo = geometry.Geometry(**kwds)
            self.assertTrue(numpy.allclose(geo.rotation_matrix(), mat), "matrice are the same %s" % kwds)
            self.assertTrue(numpy.allclose(geo.quaternion(), quat), "quaternions are the same %s" % kwds)


class TestGeometry(utilstest.ParametricTestCase):

    SIZE = 1024
    D1, D2 = numpy.mgrid[-SIZE:SIZE:32, -SIZE:SIZE:32]

    def getFunctions(self):
        functions = [("tth", ("cos", "tan")),
                     ("tth", ("tan", "cython")),
                     ("tth", ("cos", "tan")),
                     ("tth", ("tan", "cython")),
                     ("qFunction", ("numpy", "cython")),
                     ("rFunction", ("numpy", "cython")),
                     ("chi", ("numpy", "cython"))]
        return functions

    def get_geometries(self):
        pixels = {"detector": "Imxpad S10",
                  "wavelength": 1e-10}
        geometries = [{'dist': 1, 'rot1': 0, 'rot2': 0, 'rot3': 0},
                      {'dist': 1, 'rot1':-1, 'rot2': 1, 'rot3': 1},
                      {'dist': 1, 'rot1':-.2, 'rot2': 1, 'rot3':-.1},
                      {'dist': 1, 'rot1':-1, 'rot2':-.2, 'rot3': 1},
                      {'dist': 1, 'rot1': 1, 'rot2': 5, 'rot3': .4},
                      {'dist': 1, 'rot1':-1.2, 'rot2': 1, 'rot3': 1},
                      {'dist': 100, 'rot1':-2, 'rot2': 2, 'rot3': 1},
                      ]
        for g in geometries:
            g.update(pixels)
        return geometries

    def test_geometry_functions(self):
        "Test functions like tth, qFunct, rfunction... fake detectors"
        functions = self.getFunctions()
        geometries = self.get_geometries()
        params = [(k[0], k[1], g) for k, g in itertools.product(functions, geometries)]
        for func, varargs, kwds in params:
            with self.subTest(function=func, varargs=varargs, kwds=kwds):
                geo = geometry.Geometry(**kwds)
                t0 = time.perf_counter()
                oldret = getattr(geo, func)(self.D1, self.D2, path=varargs[0])
                t1 = time.perf_counter()
                newret = getattr(geo, func)(self.D1, self.D2, path=varargs[1])
                t2 = time.perf_counter()
                delta = abs(oldret - newret).max()
                logger.debug("TIMINGS\t %s meth: %s %.3fs\t meth: %s %.3fs, x%.3f delta %s",
                             func, varargs[0], t1 - t0, varargs[1], t2 - t1, (t1 - t0) / numpy.float64(t2 - t1), delta)
                msg = "func: %s max delta=%.3f, geo:%s" % (func, delta, geo)
                self.assertAlmostEqual(delta, 0, 3, msg)
                logger.debug(msg)

    def test_XYZ(self):
        """Test the calc_pos_zyx with fake detectors"""
        geometries = self.get_geometries()
        params = itertools.product((False, True), geometries)
        for corners, kwds in params:
            with self.subTest(corners=corners, kwds=kwds):
                geo = geometry.Geometry(**kwds)
                t0 = time.perf_counter()
                py_res = geo.calc_pos_zyx(None, self.D1, self.D2, corners=corners, use_cython=False)
                t1 = time.perf_counter()
                cy_res = geo.calc_pos_zyx(None, self.D1, self.D2, corners=corners, use_cython=True)
                t2 = time.perf_counter()
                delta = numpy.array([abs(py - cy).max() for py, cy in zip(py_res, cy_res)])
                logger.debug("TIMINGS\t meth: calc_pos_zyx, corner=%s python t=%.3fs\t cython: t=%.3fs\t x%.3f delta %s",
                             corners, t1 - t0, t2 - t1, (t1 - t0) / numpy.float64(t2 - t1), delta)
                msg = "delta=%s, geo= \n%s" % (delta, geo)
                self.assertTrue(numpy.allclose(numpy.vstack(cy_res), numpy.vstack(py_res)), msg)
                logger.debug(msg)

    def test_ponifile_custom_detector(self):
        config = {"pixel1": 1, "pixel2": 2, "orientation":3}
        detector = detector_factory("adsc_q315", config)
        geom = geometry.Geometry(detector=detector)
        ponifile = os.path.join(UtilsTest.tempdir, "%s.poni" % self.id())
        geom.save(ponifile)
        geom = geometry.Geometry()
        geom.load(ponifile)
        self.assertEqual(geom.detector.get_config(), config)

    def test_energy(self):
        g = geometry.Geometry()
        g.energy = 12.4
        self.assertAlmostEqual(g.wavelength, 1e-10, msg="energy conversion works", delta=1e-13)
        self.assertAlmostEqual(g.energy, 12.4, 10, msg="energy conversion is stable")

    def test_promotion(self):
        g = geometry.Geometry.sload({"detector":"pilatus200k", "poni1":0.05, "poni2":0.06})
        idmask = id(g.detector.mask)
        ai = g.promote()
        self.assertEqual(type(ai).__name__, "AzimuthalIntegrator", "Promote to AzimuthalIntegrator by default")
        ai = g.promote("FiberIntegrator")
        self.assertEqual(type(ai).__name__, "FiberIntegrator", "Promote to FiberIntegrator when requested")
        gr = g.promote("GeometryRefinement")
        self.assertEqual(type(gr).__name__, "GeometryRefinement", "Promote to GeometryRefinement when requested")
        gr = ai.promote("pyFAI.geometryRefinement.GeometryRefinement")
        self.assertEqual(type(gr).__name__, "GeometryRefinement", "Promote to GeometryRefinement when requested")
        self.assertNotEqual(id(ai.detector.mask), idmask, "detector mutable attributes got duplicated")



class TestCalcFrom(unittest.TestCase):
    """
    Test case for testing "calcfrom1d/calcfrom2d geometry
    """

    def test_calcfrom12d(self):
        det = detector_factory("pilatus300k")
        ai = AzimuthalIntegrator(0.1, 0.05, 0.04, detector=det)
        prof_1d = ai.integrate1d_ng(UtilsTest.get_rng().random(det.shape), 200, unit="2th_deg")
        sig = numpy.sinc(prof_1d.radial * 10) ** 2
        img1 = ai.calcfrom1d(prof_1d.radial, sig, dim1_unit="2th_deg", mask=det.mask, dummy=-1)
        new_prof_1d = ai.integrate1d_ng(img1, 200, unit="2th_deg")
        delta = abs((new_prof_1d.intensity - sig)).max()
        self.assertLess(delta, 2e-3, "calcfrom1d works delta=%s" % delta)
        prof_2d = ai.integrate2d(img1, 400, 360, unit="2th_deg")
        img2 = ai.calcfrom2d(prof_2d.intensity, prof_2d.radial, prof_2d.azimuthal,
                             mask=det.mask,
                             dim1_unit="2th_deg", correctSolidAngle=True, dummy=-1)
        delta2 = abs(img2 - img1).max()
        self.assertLess(delta2, 1e-3, "calcfrom2d works delta=%s" % delta2)


class TestBugRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super(TestBugRegression, cls).setUpClass()
        detector = detector_factory("Imxpad S10")  # small detectors makes calculation faster
        cls.geo = geometry.Geometry(detector=detector)
        cls.geo.setFit2D(100, detector.shape[1] // 3, detector.shape[0] // 3, tilt=1)

    @classmethod
    def tearDownClass(cls) -> None:
        super(TestBugRegression, cls).tearDownClass()
        cls.geo = None

    def test_bug747(self):
        """This bug is about PONI coordinates not subtracted from x&y coodinates in Cython"""
        # self.geo.reset()
        rc = self.geo.position_array(use_cython=True)
        rp = self.geo.position_array(use_cython=False)
        delta = abs(rp - rc).max()
        self.assertLess(delta, 1e-5, "error on position is %s" % delta)

    def test_bug2024(self):
        """This bug is about delta chi being sometimes 2pi"""
        self.geo.reset()
        deltaChi = self.geo.deltaChi()
        self.assertLess(deltaChi.max(), numpy.pi, "deltaChi is less than pi")
        self.geo.reset()
        delta_array = self.geo.delta_array(unit="chi_rad")
        self.assertLess(delta_array.max(), numpy.pi, "delta_array is less than pi")
        self.assertTrue(numpy.allclose(delta_array, deltaChi, atol=7e-6), "delta_array matches deltaChi")


class TestOrientation(unittest.TestCase):
    """Simple tests to validate the orientation of the detector"""

    @classmethod
    def setUpClass(cls) -> None:
        super(TestOrientation, cls).setUpClass()
        cls.ai1 = geometry.Geometry.sload({"detector":"pilatus100k", "detector_config":{"orientation":1},
                                           "wavelength":1e-10})
        cls.ai2 = geometry.Geometry.sload({"detector":"pilatus100k", "detector_config":{"orientation":2},
                                           "wavelength":1e-10})
        cls.ai3 = geometry.Geometry.sload({"detector":"pilatus100k", "detector_config":{"orientation":3},
                                           "wavelength":1e-10})
        cls.ai4 = geometry.Geometry.sload({"detector":"pilatus100k", "detector_config":{"orientation":4},
                                           "wavelength":1e-10})

    @classmethod
    def tearDownClass(cls) -> None:
        super(TestOrientation, cls).tearDownClass()
        cls.ai1 = cls.ai2 = cls.ai3 = cls.ai3 = None

    def test_array_from_unit_tth_center(self):
        r1 = self.ai1.array_from_unit(unit="2th_deg")
        r2 = self.ai2.array_from_unit(unit="2th_deg")
        r3 = self.ai3.array_from_unit(unit="2th_deg")
        r4 = self.ai4.array_from_unit(unit="2th_deg")

        self.assertFalse(numpy.allclose(r1, r2), "orientation 1,2 differ tth")
        self.assertFalse(numpy.allclose(r1, r3), "orientation 1,3 differ tth")
        self.assertFalse(numpy.allclose(r1, r4), "orientation 1,4 differ tth")

        self.assertTrue(numpy.allclose(r1, numpy.fliplr(r2)), "orientation 1,2 flipped match tth")
        self.assertTrue(numpy.allclose(r1, numpy.flipud(r4)), "orientation 1,4 flipped match tth")
        self.assertTrue(numpy.allclose(r2, numpy.flipud(r3)), "orientation 2,3 flipped match tth")
        self.assertTrue(numpy.allclose(r1, r3[-1::-1, -1::-1]), "orientation 1,3 inversion match tth")
        self.assertTrue(numpy.allclose(r2, r4[-1::-1, -1::-1]), "orientation 2,4 inversion match tth")

    def test_array_from_unit_chi_center(self):
        r1 = self.ai1.array_from_unit(unit="chi_deg")
        r2 = self.ai2.array_from_unit(unit="chi_deg")
        r3 = self.ai3.array_from_unit(unit="chi_deg")
        r4 = self.ai4.array_from_unit(unit="chi_deg")

        self.assertFalse(numpy.allclose(r1, r2), "orientation 1,2 differ chi")
        self.assertFalse(numpy.allclose(r1, r3), "orientation 1,3 differ chi")
        self.assertFalse(numpy.allclose(r1, r4), "orientation 1,4 differ chi")

        self.assertTrue(-180 < r1.min() < -179, "Orientation 1 lower range matches")
        self.assertTrue(-91 < r1.max() < -90, "Orientation 1 upperrange matches")
        self.assertTrue(-90 < r2.min() < -89, "Orientation 2 lower range matches")
        self.assertTrue(-1 < r2.max() < 0, "Orientation 2 upperrange matches")
        self.assertTrue(0 < r3.min() < 1, "Orientation 3 lower range matches")
        self.assertTrue(89 < r3.max() < 90, "Orientation 3 upperrange matches")
        self.assertTrue(90 < r4.min() < 91, "Orientation 4 lower range matches")
        self.assertTrue(179 < r4.max() < 180, "Orientation 4 upperrange matches")

    def test_array_from_unit_tth_corner(self):
        r1 = self.ai1.array_from_unit(unit="2th_rad", typ="corner")
        r2 = self.ai2.array_from_unit(unit="2th_rad", typ="corner")
        r3 = self.ai3.array_from_unit(unit="2th_rad", typ="corner")
        r4 = self.ai4.array_from_unit(unit="2th_rad", typ="corner")

        tth1 = r1[..., 0]
        tth2 = r2[..., 0]
        tth3 = r3[..., 0]
        tth4 = r4[..., 0]

        chi1 = r1[..., 1]
        chi2 = r2[..., 1]
        chi3 = r3[..., 1]
        chi4 = r4[..., 1]

        sin_chi1 = numpy.sin(chi1)
        sin_chi2 = numpy.sin(chi2)
        sin_chi3 = numpy.sin(chi3)
        sin_chi4 = numpy.sin(chi4)

        cos_chi1 = numpy.cos(chi1)
        cos_chi2 = numpy.cos(chi2)
        cos_chi3 = numpy.cos(chi3)
        cos_chi4 = numpy.cos(chi4)

        # Here we use complex numbers
        z1 = tth1 * cos_chi1 + tth1 * sin_chi1 * 1j
        z2 = tth2 * cos_chi2 + tth2 * sin_chi2 * 1j
        z3 = tth3 * cos_chi3 + tth3 * sin_chi3 * 1j
        z4 = tth4 * cos_chi4 + tth4 * sin_chi4 * 1j

        # the mean is not sensitive to 2pi discontinuity in azimuthal direction
        z1 = z1.mean(axis=-1)
        z2 = z2.mean(axis=-1)
        z3 = z3.mean(axis=-1)
        z4 = z4.mean(axis=-1)

        self.assertFalse(numpy.allclose(z1, z2), "orientation 1,2 differ")
        self.assertFalse(numpy.allclose(z1, z3), "orientation 1,3 differ")
        self.assertFalse(numpy.allclose(z1, z3), "orientation 1,3 differ")
        self.assertFalse(numpy.allclose(z1, z4), "orientation 1,4 differ")
        self.assertFalse(numpy.allclose(z2, z3), "orientation 2,3 differ")
        self.assertFalse(numpy.allclose(z2, z4), "orientation 2,4 differ")
        self.assertFalse(numpy.allclose(z3, z4), "orientation 3,4 differ")

        # Check that the tranformation is OK. This is with complex number thus dense & complicated !
        self.assertTrue(numpy.allclose(z1, -numpy.fliplr(z2.conj())), "orientation 1,2 flipped")
        self.assertTrue(numpy.allclose(z1, -z3[-1::-1, -1::-1]), "orientation 1,3 inversed")
        self.assertTrue(numpy.allclose(z1, numpy.flipud(z4.conj())), "orientation 1,4 flipped")
        self.assertTrue(numpy.allclose(z2, numpy.flipud(z3.conj())), "orientation 2,3 flipped")
        self.assertTrue(numpy.allclose(z2, -z4[-1::-1, -1::-1]), "orientation 2,4 inversion")
        self.assertTrue(numpy.allclose(z3, -numpy.fliplr(z4.conj())), "orientation 3,4 flipped")

    def test_chi(self):
        epsilon = 6e-3
        orient = {}
        for i in range(1, 5):
            ai = geometry.Geometry.sload({"detector":"Imxpad S10", "detector_config":{"orientation":i},
                                           "poni1":0.005, "poni2":0.005, "wavelength":1e-10})
            chi_c = ai.center_array(unit="chi_rad") / numpy.pi
            corners = ai.corner_array(unit="r_mm")
            corners_rad = corners[..., 0]
            corners_ang = corners[..., 1]
            z = corners_rad * numpy.cos(corners_ang) + corners_rad * numpy.sin(corners_ang) * 1j
            chi_m = numpy.angle(z.mean(axis=-1)) / numpy.pi

            orient[i] = {"ai": ai, "chi_c": chi_c, "chi_m": chi_m}

        for o, orien in orient.items():
            self.assertTrue(allclose_mod(orien["chi_m"], orien["chi_c"], 2), f"Orientation {o} matches")
            ai = orien["ai"]
            self.assertLess(numpy.median(ai.delta_array(unit="chi_rad")) / numpy.pi, epsilon, f"Orientation {o} delta chi small #0")
            self.assertLess(numpy.median(ai.deltaChi()) / numpy.pi, epsilon, f"Orientation {o} delta chi small #1")
            ai.reset()
            self.assertLess(numpy.median(ai.delta_array(unit="chi_rad")) / numpy.pi, epsilon, f"Orientation {o} delta chi small #2")
            ai.reset()
            self.assertLess(numpy.median(ai.deltaChi()) / numpy.pi, epsilon, f"Orientation {o} delta chi small #3")
            ai.reset()
            chiArray = ai.chiArray() / numpy.pi
            chi_center = orien["chi_c"]
            self.assertTrue(allclose_mod(chiArray, chi_center), f"Orientation {o} chiArray == center_array(chi)")


class TestOrientation2(unittest.TestCase):
    """Simple tests to validate the orientation of the detector"""

    @classmethod
    def setUpClass(cls) -> None:
        super(TestOrientation2, cls).setUpClass()
        p = detector_factory("pilatus100k")
        c = p.get_pixel_corners()
        d1 = c[..., 1].max()
        d2 = c[..., 2].max()
        cls.ai1 = geometry.Geometry.sload({"poni1":3 * d1 / 4, "poni2":3 * d2 / 4, "wavelength":1e-10,
                                           "detector":"pilatus100k", "detector_config":{"orientation":1}})
        cls.ai2 = geometry.Geometry.sload({"poni1":3 * d1 / 4, "poni2":d2 / 4, "wavelength":1e-10,
                                           "detector":"pilatus100k", "detector_config":{"orientation":2}})
        cls.ai3 = geometry.Geometry.sload({"poni1":d1 / 4, "poni2":d2 / 4, "wavelength":1e-10,
                                           "detector":"pilatus100k", "detector_config":{"orientation":3}})
        cls.ai4 = geometry.Geometry.sload({"poni1":d1 / 4, "poni2":3 * d2 / 4, "wavelength":1e-10,
                                           "detector":"pilatus100k", "detector_config":{"orientation":4}})

    @classmethod
    def tearDownClass(cls) -> None:
        super(TestOrientation2, cls).tearDownClass()
        cls.ai1 = cls.ai2 = cls.ai3 = cls.ai3 = None

    def test_positions(self):
        for cc in (True, False):
            for cm in (True, False):
                for ai in (self.ai1, self.ai2, self.ai3, self.ai4):
                    zc, yc, xc = ai.calc_pos_zyx(corners=False, use_cython=cc)
                    zco, yco, xco = ai.calc_pos_zyx(corners=True, use_cython=cm)
                    zm = zco.mean(axis=-1)
                    ym = yco.mean(axis=-1)
                    xm = xco.mean(axis=-1)

                    self.assertTrue(numpy.allclose(zc, zm, atol=1e-8), f"check Z on {ai.detector.orientation.name}, cython: center={cc}, corner={cm}")
                    self.assertTrue(numpy.allclose(yc, ym, atol=1e-8), f"check Y on {ai.detector.orientation.name}, cython: center={cc}, corner={cm}")
                    self.assertTrue(numpy.allclose(xc, xm, atol=1e-8), f"check X on {ai.detector.orientation.name}, cython: center={cc}, corner={cm}")

    def test_center_radius_center(self):
        r1 = self.ai1.array_from_unit(unit="r_m", typ="center")
        r2 = self.ai2.array_from_unit(unit="r_m", typ="center")
        r3 = self.ai3.array_from_unit(unit="r_m", typ="center")
        r4 = self.ai4.array_from_unit(unit="r_m", typ="center")
        self.assertTrue(numpy.allclose(r1, r2, atol=1e-8))
        self.assertTrue(numpy.allclose(r1, r3, atol=1e-8))
        self.assertTrue(numpy.allclose(r1, r4, atol=1e-8))
        self.assertTrue(numpy.allclose(r2, r3, atol=1e-8))
        self.assertTrue(numpy.allclose(r2, r4, atol=1e-8))
        self.assertTrue(numpy.allclose(r3, r4, atol=1e-8))

    def test_center_chi_center(self):
        r1 = self.ai1.array_from_unit(unit="chi_rad", typ="center") / numpy.pi
        r2 = self.ai2.array_from_unit(unit="chi_rad", typ="center") / numpy.pi
        r3 = self.ai3.array_from_unit(unit="chi_rad", typ="center") / numpy.pi
        r4 = self.ai4.array_from_unit(unit="chi_rad", typ="center") / numpy.pi
        self.assertTrue(numpy.allclose(r1[:, 200:], r2[:, 200:], atol=1e-8))
        self.assertTrue(numpy.allclose(r1[:, 200:], r3[:, 200:], atol=1e-8))
        self.assertTrue(numpy.allclose(r1[:, 200:], r4[:, 200:], atol=1e-8))
        self.assertTrue(numpy.allclose(r2[:, 200:], r3[:, 200:], atol=1e-8))
        self.assertTrue(numpy.allclose(r2[:, 200:], r4[:, 200:], atol=1e-8))
        self.assertTrue(numpy.allclose(r3[:, 200:], r4[:, 200:], atol=1e-8))

    def test_center_tth_center(self):
        r1 = self.ai1.array_from_unit(unit="2th_deg", typ="corner")
        r2 = self.ai2.array_from_unit(unit="2th_deg", typ="corner")
        r3 = self.ai3.array_from_unit(unit="2th_deg", typ="corner")
        r4 = self.ai4.array_from_unit(unit="2th_deg", typ="corner")
        tth1 = r1[..., 0].mean(axis=-1)
        chi1 = r1[..., 1].mean(axis=-1)
        tth2 = r2[..., 0].mean(axis=-1)
        chi2 = r2[..., 1].mean(axis=-1)
        tth3 = r3[..., 0].mean(axis=-1)
        chi3 = r3[..., 1].mean(axis=-1)
        tth4 = r4[..., 0].mean(axis=-1)
        chi4 = r4[..., 1].mean(axis=-1)

        res = []
        tths = [tth1, tth2, tth3, tth4]
        thres = 0.1
        for idx, a1 in enumerate(tths):
            for a2 in tths[:idx]:
                res.append(numpy.allclose(a1, a2, atol=thres))
        # print(res)
        self.assertTrue(numpy.all(res), "2th is OK")

        res = []
        tths = [chi1, chi2, chi3, chi4]
        thres = 0.1
        for idx, a1 in enumerate(tths):
            for a2 in tths[:idx]:
                res.append(numpy.allclose(a1[:, 200:], a2[:, 200:], atol=thres))
        # print(res)
        self.assertTrue(numpy.all(res), "2th is OK")


class TestCrystFEL(unittest.TestCase):
    """Simple tests to validate the import from CrystFEL"""

    def test_crystfel(self):
        results = {"alignment-test.geom": {"poni_version": 2.1,
                                           "detector": "Detector",
                                           "detector_config": {
                                               "pixel1": 0.0001,
                                               "pixel2": 0.0001,
                                               "max_shape": [
                                                   1025,
                                                   1025
                                                   ],
                                               "orientation": 3
                                               },
                                           "dist": 100.0,
                                           "poni1": 0.2048,
                                           "poni2": 0.2048,
                                           "rot1": 0,
                                           "rot2": 0,
                                           "rot3": 0,
                                           "wavelength": 1.3776022048133363e-10, },
                    "cspad-cxiformat.geom": None,  # -> clen in HDF5
                    "cspad-single.geom": None,  # -> clen in HDF5
                    "Eiger16M-binning2-nativefiles.geom": {  "poni_version": 2.1,
                                                            "detector": "Detector",
                                                            "detector_config": {
                                                                "pixel1": 7.500018750046876e-05,
                                                                "pixel2": 7.500018750046876e-05,
                                                                "max_shape": [
                                                                    2167,
                                                                    2070
                                                                ],
                                                                "orientation": 3
                                                            },
                                                            "dist": 0.1,
                                                            "poni1": 0.07500018750046876,
                                                            "poni2": 0.07500018750046876,
                                                            "rot1": 0,
                                                            "rot2": 0,
                                                            "rot3": 0,
                                                            "wavelength": 5.6356453833272844e-11
                                                        },
                    "ev_enum1.geom": None,
                    "ev_enum2.geom": {  "poni_version": 2.1,
                                        "detector": "Detector",
                                        "detector_config": {
                                            "pixel1": 1e-06,
                                            "pixel2": 1e-06,
                                            "max_shape": [
                                                2,
                                                1
                                            ],
                                            "orientation": 3
                                        },
                                        "dist": 50.0,
                                        "poni1": 9.999999999999999e-05,
                                        "poni2": 9.999999999999999e-05,
                                        "rot1": 0,
                                        "rot2": 0,
                                        "rot3": 0,
                                        "wavelength": 1.2398419843320025e-10
                                    },
                    "ev_enum3.geom": {  "poni_version": 2.1,
                                        "detector": "Detector",
                                        "detector_config": {
                                            "pixel1": 1e-06,
                                            "pixel2": 1e-06,
                                            "max_shape": [
                                                2,
                                                1
                                            ],
                                            "orientation": 3
                                        },
                                        "dist": 50.0,
                                        "poni1": 9.999999999999999e-05,
                                        "poni2": 9.999999999999999e-05,
                                        "rot1": 0,
                                        "rot2": 0,
                                        "rot3": 0,
                                        "wavelength": 1.2398419843320025e-10
                                    },
                    "jf-swissfel-16M.geom":{"poni_version": 2.1,
                                            "detector": "Detector",
                                            "detector_config": {
                                                "pixel1": 7.500018750046876e-05,
                                                "pixel2": 7.500018750046876e-05,
                                                "max_shape": [
                                                    16448,
                                                    1030
                                                ],
                                                "orientation": 3
                                            },
                                            "dist": 95.3,
                                            "poni1": 0.1582400308744524,
                                            "poni2": 0.20740705679872742,
                                            "rot1": 0,
                                            "rot2": 0,
                                            "rot3": 0,
                                            "wavelength": 2.713002153899349e-10 },
                    "lcls-dec.geom": {"poni_version": 2.1,
                                        "detector": "Detector",
                                        "detector_config": {
                                            "pixel1": 7.500018750046876e-05,
                                            "pixel2": 7.500018750046876e-05,
                                            "max_shape": [
                                                1024,
                                                1024
                                            ],
                                            "orientation": 3
                                        },
                                        "dist": 0.0678,
                                        "poni1": 0.058477646194115496,
                                        "poni2": 0.03690009225023063,
                                        "rot1": 0,
                                        "rot2": 0,
                                        "rot3": 0 },
                    "lcls-june-r0013-r0128.geom": { "poni_version": 2.1,
                                                    "detector": "Detector",
                                                    "detector_config": {
                                                        "pixel1": 7.500018750046876e-05,
                                                        "pixel2": 7.500018750046876e-05,
                                                        "max_shape": [
                                                            1024,
                                                            1024
                                                        ],
                                                        "orientation": 3
                                                    },
                                                    "dist": 0.06478,
                                                    "poni1": 0.06757516893792236,
                                                    "poni2": 0.03892509731274329,
                                                    "rot1": 0,
                                                    "rot2": 0,
                                                    "rot3": 0},
                    "lcls-xpp-estimate.geom": {"poni_version": 2.1,
                                                "detector": "Detector",
                                                "detector_config": {
                                                    "pixel1": 0.00011001100110011001,
                                                    "pixel2": 0.00011001100110011001,
                                                    "max_shape": [
                                                        1456,
                                                        1456
                                                    ],
                                                    "orientation": 3
                                                },
                                                "dist": 0.08,
                                                "poni1": 0.08003300330033003,
                                                "poni2": 0.08003300330033003,
                                                "rot1": 0,
                                                "rot2": 0,
                                                "rot3": 0},
                    "pilatus.geom": None,  # -> clen in HDF5
                    "simple.geom": {"poni_version": 2.1,
                                    "detector": "Detector",
                                    "detector_config": {
                                        "pixel1": 7.500018750046876e-05,
                                        "pixel2": 7.500018750046876e-05,
                                        "max_shape": [
                                            1024,
                                            1024
                                        ],
                                        "orientation": 3
                                    },
                                    "dist": 0.05,
                                    "poni1": 0.03915009787524469,
                                    "poni2": 0.038400096000240004,
                                    "rot1": 0,
                                    "rot2": 0,
                                    "rot3": 0
                                },
                    "stream_roundtrip.geom": {"poni_version": 2.1,
                                                "detector": "Detector",
                                                "detector_config": {
                                                    "pixel1": 1e-06,
                                                    "pixel2": 1e-06,
                                                    "max_shape": [
                                                        100,
                                                        100
                                                    ],
                                                    "orientation": 3
                                                },
                                                "dist": 50.0,
                                                "poni1":-0.0,
                                                "poni2": 0.00019999999999999998,
                                                "rot1": 0,
                                                "rot2": 0,
                                                "rot3": 0,
                                                "wavelength": 1.2398419843320025e-10
                                            },
                    "wavelength_geom1.geom": None,  # -> wavelength in HDF5
                    "wavelength_geom2.geom": None,  # -> photon_energy in HDF5
                    "wavelength_geom3.geom": None,  # -> photon_energy in HDF5
                    "wavelength_geom4.geom": {"poni_version": 2.1,
                                                "detector": "Detector",
                                                "detector_config": {
                                                    "pixel1": 1e-06,
                                                    "pixel2": 1e-06,
                                                    "max_shape": [
                                                        2,
                                                        1
                                                    ],
                                                    "orientation": 3
                                                },
                                                "dist": 50.0,
                                                "poni1": 9.999999999999999e-05,
                                                "poni2": 9.999999999999999e-05,
                                                "rot1": 0,
                                                "rot2": 0,
                                                "rot3": 0
                                            },
                    "wavelength_geom5.geom": {"poni_version": 2.1,
                                                "detector": "Detector",
                                                "detector_config": {
                                                    "pixel1": 1e-06,
                                                    "pixel2": 1e-06,
                                                    "max_shape": [
                                                        2,
                                                        1
                                                    ],
                                                    "orientation": 3
                                                },
                                                "dist": 50.0,
                                                "poni1": 9.999999999999999e-05,
                                                "poni2": 9.999999999999999e-05,
                                                "rot1": 0,
                                                "rot2": 0,
                                                "rot3": 0
                                            },
                    "wavelength_geom6.geom": None,  # -> parameters in HDF5
                    "wavelength_geom7.geom": {"poni_version": 2.1,
                                                "detector": "Detector",
                                                "detector_config": {
                                                    "pixel1": 1e-06,
                                                    "pixel2": 1e-06,
                                                    "max_shape": [
                                                        2,
                                                        1
                                                    ],
                                                    "orientation": 3
                                                },
                                                "dist": 50.0,
                                                "poni1": 9.999999999999999e-05,
                                                "poni2": 9.999999999999999e-05,
                                                "rot1": 0,
                                                "rot2": 0,
                                                "rot3": 0,
                                                "wavelength": 1.3776022048133363e-10
                                            },
                    "wavelength_geom8.geom": {"poni_version": 2.1,
                                                "detector": "Detector",
                                                "detector_config": {
                                                    "pixel1": 1e-06,
                                                    "pixel2": 1e-06,
                                                    "max_shape": [
                                                        2,
                                                        1
                                                    ],
                                                    "orientation": 3
                                                },
                                                "dist": 50.0,
                                                "poni1": 9.999999999999999e-05,
                                                "poni2": 9.999999999999999e-05,
                                                "rot1": 0,
                                                "rot2": 0,
                                                "rot3": 0
                                            },
                    "wavelength_geom9.geom": {  "poni_version": 2.1,
                                                "detector": "Detector",
                                                "detector_config": {
                                                    "pixel1": 1e-06,
                                                    "pixel2": 1e-06,
                                                    "max_shape": [
                                                        2,
                                                        1
                                                    ],
                                                    "orientation": 3
                                                },
                                                "dist": 50.0,
                                                "poni1": 9.999999999999999e-05,
                                                "poni2": 9.999999999999999e-05,
                                                "rot1": 0,
                                                "rot2": 0,
                                                "rot3": 0,
                                                "wavelength": 1.3776022048133363e-10
                                            },
                    "wavelength_geom10.geom": {"poni_version": 2.1,
                                                "detector": "Detector",
                                                "detector_config": {
                                                    "pixel1": 1e-06,
                                                    "pixel2": 1e-06,
                                                    "max_shape": [
                                                        2,
                                                        1
                                                    ],
                                                    "orientation": 3
                                                },
                                                "dist": 50.0,
                                                "poni1": 9.999999999999999e-05,
                                                "poni2": 9.999999999999999e-05,
                                                "rot1": 0,
                                                "rot2": 0,
                                                "rot3": 0,
                                                "wavelength": 1.3776022048133363e-10
                                            },
                    "wavelength_geom11.geom": {"poni_version": 2.1,
                                                "detector": "Detector",
                                                "detector_config": {
                                                    "pixel1": 1e-06,
                                                    "pixel2": 1e-06,
                                                    "max_shape": [
                                                        2,
                                                        1
                                                    ],
                                                    "orientation": 3
                                                },
                                                "dist": 50.0,
                                                "poni1": 9.999999999999999e-05,
                                                "poni2": 9.999999999999999e-05,
                                                "rot1": 0,
                                                "rot2": 0,
                                                "rot3": 0,
                                                "wavelength": 1.125e-10
                                            },
                    "wavelength_geom12.geom": { "poni_version": 2.1,
                                                "detector": "Detector",
                                                "detector_config": {
                                                    "pixel1": 1e-06,
                                                    "pixel2": 1e-06,
                                                    "max_shape": [
                                                        2,
                                                        1
                                                    ],
                                                    "orientation": 3
                                                },
                                                "dist": 50.0,
                                                "poni1": 9.999999999999999e-05,
                                                "poni2": 9.999999999999999e-05,
                                                "rot1": 0,
                                                "rot2": 0,
                                                "rot3": 0,
                                                "wavelength": 1.125e-10
                                            },
                   }

        for i, ref in results.items():
            geom = UtilsTest.getimage(i)
            dico = parse_crystfel_geom(geom)
            if ref is not None:
                ai = build_geometry(dico)
                poni_res = PoniFile(ai)
                poni_ref = PoniFile(ref)
                self.assertEqual(json.dumps(poni_res.as_dict()),
                                 json.dumps(poni_ref.as_dict()), f"geometry matches for {i}")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestBugRegression))
    testsuite.addTest(loader(TestSolidAngle))
    testsuite.addTest(loader(TestBug88SolidAngle))
    testsuite.addTest(loader(TestRecprocalSpacingSquarred))
    testsuite.addTest(loader(TestCalcFrom))
    testsuite.addTest(loader(TestGeometry))
    testsuite.addTest(loader(TestFastPath))
    testsuite.addTest(loader(TestOrientation))
    testsuite.addTest(loader(TestOrientation2))
    testsuite.addTest(loader(TestCrystFEL))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
