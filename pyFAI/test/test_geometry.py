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

"""tests for Jon's geometry changes
FIXME : make some tests that the functions do what is expected
"""

from __future__ import absolute_import, division, print_function

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "09/05/2019"


import unittest
import sys
import random
import time
import numpy
import itertools
import logging
import os.path

from . import utilstest
logger = logging.getLogger(__name__)

from .. import geometry
from ..azimuthalIntegrator import AzimuthalIntegrator
from .. import units
from ..detectors import detector_factory
from ..third_party import transformations
from .utilstest import UtilsTest
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
        data = fabio.open(pilatusFile).data
        data[data < 0] = 0  # discard negative pixels

        tth, I_nogood = ai.integrate1d(data, 1770, unit="2th_deg", radial_range=[0, 56], method="splitBBox", correctSolidAngle=False)
        delta_tth = abs(tth - tth_fit2d).max()
        delta_I = abs(I_nogood - I_fit2d).max()
        mean_I = abs(I_nogood - I_fit2d).mean()
        self.assertTrue(delta_tth < 1e-5, 'Error on 2th position: %s <1e-5' % delta_tth)
        self.assertTrue(delta_I > 100, 'Error on (wrong) I are large: %s >100' % delta_I)
        self.assertTrue(mean_I > 2, 'Error on (wrong) I are large: %s >2' % mean_I)

        tth, I_good = ai.integrate1d(data, 1770, unit="2th_deg", radial_range=[0, 56], method="splitBBox", correctSolidAngle=3)
        delta_tth = abs(tth - tth_fit2d).max()
        delta_I = abs(I_good - I_fit2d).max()
        mean_I = abs(I_good - I_fit2d).mean()
        self.assertTrue(delta_tth < 1e-5, 'Error on 2th position: %s <1e-5' % delta_tth)
        self.assertTrue(delta_I < 5, 'Error on (good) I are small: %s <5' % delta_I)
        self.assertTrue(mean_I < 0.05, 'Error on (good) I are small: %s <0.05' % mean_I)
        ai.reset()

    def test_nonflat_center(self):
        """
        Test non flat detector cos(incidence) to be 1 (+/- 1%) when centered.

        Aarhus is a curved detector of radius 0.3m
        """
        aarhus = detector_factory("Aarhus")
        aarhus.binning = (10, 10)
        ai = AzimuthalIntegrator(aarhus.radius, detector=aarhus)
        cosa = numpy.fromfunction(ai.cos_incidence,
                                  aarhus.shape, dtype=numpy.float32)
        maxi = cosa.max()
        mini = cosa.min()
        self.assertTrue(maxi <= 1.0, 'Cos incidence is %s <=1.0' % maxi)
        self.assertTrue(mini > 0.99, 'Cos solid angle is %s >0.99' % mini)

    def test_nonflat_outside(self):
        """
        Test non flat detector cos(incidence) to be !=1 when off-centered.

        Aarhus is a curved detector of radius 0.3m, here we offset of 50%
        """
        aarhus = detector_factory("Aarhus")
        aarhus.binning = (10, 10)
        ai = AzimuthalIntegrator(aarhus.radius * 1.5, detector=aarhus)
        cosa = numpy.fromfunction(ai.cos_incidence,
                                  aarhus.shape, dtype=numpy.float32)
        maxi = cosa.max()
        mini = cosa.min()
        self.assertTrue(maxi <= 1.0, 'Cos incidence is %s <=1.0' % maxi)
        self.assertTrue(maxi > 0.99, 'Cos incidence max is %s >0.99' % maxi)
        self.assertTrue(mini < 0.92, 'Cos solid angle min is %s <0.92' % mini)

    def test_nonflat_inside(self):
        """
        Test non flat detector cos(incidence) to be !=1 when off-centered.

        Aarhus is a curved detector of radius 0.3m, here we offset of 50%
        """
        aarhus = detector_factory("Aarhus")
        aarhus.binning = (10, 10)
        ai = AzimuthalIntegrator(aarhus.radius * 0.5, detector=aarhus)
        cosa = numpy.fromfunction(ai.cos_incidence,
                                  aarhus.shape, dtype=numpy.float32)
        maxi = cosa.max()
        mini = cosa.min()
        self.assertTrue(maxi <= 1.0, 'Cos incidence is %s <=1.0' % maxi)
        self.assertTrue(maxi > 0.99, 'Cos incidence max is %s >0.99' % maxi)
        self.assertTrue(mini < 0.87, 'Cos solid angle min is %s <0.86' % mini)


class TestBug88SolidAngle(unittest.TestCase):
    """
    Test case for solid angle where data got modified inplace.

    https://github.com/silx-kit/pyFAI/issues/88
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
        rd2 = self.geo.corner_array(self.shape, unit=units.RecD2_NM, scale=False)[:, :, :, 0]
        q = self.geo.corner_array(self.shape, unit=units.Q, use_cython=False, scale=False)[:, :, :, 0]
        delta = rd2 - (q / (2 * numpy.pi)) ** 2
        self.assertTrue(numpy.allclose(rd2, (q / (2 * numpy.pi)) ** 2), "corners rd2 = (q/2pi)**2, delat=%s" % delta)

    def test_delta(self):
        drd2a = self.geo.deltaRd2(self.shape)
        rd2 = self.geo.rd2Array(self.shape)
        rc = self.geo.corner_array(self.shape, unit=units.RecD2_NM, scale=False)[:, :, :, 0]
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
            {"dist": 0.037759112584709535, "poni1": 0.005490358659182459, "poni2": 0.06625690275821605, "rot1": 0.20918568578536278, "rot2": 0.42161920581114365, "rot3": 0.38784171093239983, "wavelength": 1e-10, 'detector': 'Pilatus100k'},
            # Provides atol = 2.8e-5
            {'dist': 0.48459003559204783, 'poni2':-0.15784154756282065, 'poni1': 0.02783657100374448, 'rot3':-0.2901541134116695, 'rot1':-0.3927992588689394, 'rot2': 0.148115949280184, "wavelength": 1e-10, 'detector': 'Pilatus100k'},
            # Provides atol = 3.67761e-05
            {'poni1':-0.22055143279015976, 'poni2':-0.11124668733292842, 'rot1':-0.18105235367380956, 'wavelength': 1e-10, 'rot3': 0.2146474866836957, 'rot2': 0.36581323339171257, 'detector': 'Pilatus100k', 'dist': 0.7350926443000882},
            # Provides atol = 4.94719e-05
            {'poni2': 0.1010652698401574, 'rot3':-0.30578860159890153, 'rot1': 0.46240992613529186, 'wavelength': 1e-10, 'detector': 'Pilatus300k', 'rot2':-0.027476969196682077, 'dist': 0.04711960678381288, 'poni1': 0.012745759325719641},
            # atol=2pi
            {'poni1': 0.07803878450256929, 'poni2': 0.2601779472529494, 'rot1':-0.33177239820033455, 'wavelength': 1e-10, 'rot3': 0.2928945825578625, 'rot2': 0.2762729953307118, 'detector': 'Pilatus100k', 'dist': 0.43544642285972124},
            {'wavelength': 1e-10, 'dist': 0.13655542730645986, 'rot1':-0.16145635108891077, 'poni1': 0.16271587645146157, 'rot2':-0.443426307059295, 'rot3': 0.40517456402269536, 'poni2': 0.05248001026597382, 'detector': 'Pilatus100k'}
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
            geo = {"dist": 0.01 + random.random(),
                   "poni1": random.random() - 0.5,
                   "poni2": random.random() - 0.5,
                   "rot1": (random.random() - 0.5) * numpy.pi,
                   "rot2": (random.random() - 0.5) * numpy.pi,
                   "rot3": (random.random() - 0.5) * numpy.pi,
                   "wavelength": 1e-10}

            for det in detectors:
                dico = geo.copy()
                dico["detector"] = det
                geometries.append(dico)
                q = transformations.quaternion_from_euler(-dico["rot1"], -dico["rot2"], dico["rot3"], axes="sxyz")
                quaternions.append(q)
                matrices.append(transformations.quaternion_matrix(q)[:3, :3])
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
                t00 = timer()
                py_res = geo.corner_array(unit=space, use_cython=False, scale=False)
                t01 = timer()
                geo.reset()
                t10 = timer()
                cy_res = geo.corner_array(unit=space, use_cython=True, scale=False)
                t11 = timer()
                delta = abs(py_res - cy_res)
                # We expect precision on radial position
                delta_r = delta[..., 0].max()
                # issue with numerical stability of azimuthal position due to arctan(y,x)
                cnt_delta_a = (delta[..., 1] > self.EPSILON_A).sum()
                logger.debug("TIMINGS\t meth: %s %s Python: %.3fs, Cython: %.3fs\t x%.3f\t delta_r:%s",
                             space, data["detector"], t01 - t00, t11 - t10, (t01 - t00) / numpy.float64(t11 - t10), delta)
                self.assertTrue(delta_r < self.EPSILON_R, "data=%s, space='%s' delta_r: %s" % (data, space, delta_r))
                self.assertTrue(cnt_delta_a < count_a, "data:%s, space: %s cnt_delta_a: %s" % (data, space, cnt_delta_a))

    def test_XYZ(self):
        """Test the calc_pos_zyx with full detectors"""
        geometries = self.get_geometries()
        for geometryParams in geometries:
            with self.subTest(geometry=geometry):
                geo = geometry.Geometry(**geometryParams)
                t0 = timer()
                py_res = geo.calc_pos_zyx(corners=True, use_cython=False)
                t1 = timer()
                cy_res = geo.calc_pos_zyx(corners=True, use_cython=True)
                t2 = timer()
                delta = numpy.array([abs(py - cy).max() for py, cy in zip(py_res, cy_res)])
                logger.debug("TIMINGS\t meth: calc_pos_zyx %s, corner=True python t=%.3fs\t cython: t=%.3fs \t x%.3f delta %s",
                             geometryParams["detector"], t1 - t0, t2 - t1, (t1 - t0) / numpy.float64(t2 - t1), delta)
                msg = "delta=%s<%s, geo= \n%s" % (delta, self.EPSILON, geo)
                self.assertTrue(numpy.alltrue(delta.max() < self.EPSILON), msg)
                logger.debug(msg)

    def test_deltachi(self):
        """Test the deltaChi"""
        geometries = self.get_geometries()
        for geometryParams in geometries:
            with self.subTest(geometry=geometryParams):
                geo = geometry.Geometry(**geometryParams)
                t0 = timer()
                py_res = geo.deltaChi(use_cython=False)
                # t1 = timer()
                geo.reset()
                t1 = timer()
                cy_res = geo.deltaChi(use_cython=True)
                t2 = timer()
                delta = numpy.array([abs(py - cy).max() for py, cy in zip(py_res, cy_res)])
                logger.debug("TIMINGS\t meth: deltaChi %s python t=%.3fs\t cython: t=%.3fs \t x%.3f delta %s",
                             geometryParams["detector"], t1 - t0, t2 - t1, (t1 - t0) / numpy.float64(t2 - t1), delta)
                msg = "delta=%s<%s, geo= \n%s" % (delta, self.EPSILON, geo)
                self.assertTrue(numpy.alltrue(delta.max() < self.EPSILON), msg)
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
            self.assert_(numpy.allclose(geo.rotation_matrix(), mat), "matrice are the same %s" % kwds)
            self.assert_(numpy.allclose(geo.quaternion(), quat), "quaternions are the same %s" % kwds)


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
        pixels = {"detector": "Pilatus100k",
                  "wavelength": 1e-10}
        geometries = [{'dist': 1, 'rot1': 0, 'rot2': 0, 'rot3': 0},
                      {'dist': 1, 'rot1': -1, 'rot2': 1, 'rot3': 1},
                      {'dist': 1, 'rot1': -.2, 'rot2': 1, 'rot3': -.1},
                      {'dist': 1, 'rot1': -1, 'rot2': -.2, 'rot3': 1},
                      {'dist': 1, 'rot1': 1, 'rot2': 5, 'rot3': .4},
                      {'dist': 1, 'rot1': -1.2, 'rot2': 1, 'rot3': 1},
                      {'dist': 100, 'rot1': -2, 'rot2': 2, 'rot3': 1},
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
                t0 = timer()
                oldret = getattr(geo, func)(self.D1, self.D2, path=varargs[0])
                t1 = timer()
                newret = getattr(geo, func)(self.D1, self.D2, path=varargs[1])
                t2 = timer()
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
                t0 = timer()
                py_res = geo.calc_pos_zyx(None, self.D1, self.D2, corners=corners, use_cython=False)
                t1 = timer()
                cy_res = geo.calc_pos_zyx(None, self.D1, self.D2, corners=corners, use_cython=True)
                t2 = timer()
                delta = numpy.array([abs(py - cy).max() for py, cy in zip(py_res, cy_res)])
                logger.debug("TIMINGS\t meth: calc_pos_zyx, corner=%s python t=%.3fs\t cython: t=%.3fs\t x%.3f delta %s",
                             corners, t1 - t0, t2 - t1, (t1 - t0) / numpy.float64(t2 - t1), delta)
                msg = "delta=%s, geo= \n%s" % (delta, geo)
                self.assertTrue(numpy.allclose(numpy.vstack(cy_res), numpy.vstack(py_res)), msg)
                logger.debug(msg)

    def test_ponifile_custom_detector(self):
        config = {"pixel1": 1, "pixel2": 2}
        detector = detector_factory("adsc_q315", config)
        geom = geometry.Geometry(detector=detector)
        ponifile = os.path.join(UtilsTest.tempdir, "%s.poni" % self.id())
        geom.save(ponifile)
        geom = geometry.Geometry()
        geom.load(ponifile)
        self.assertEqual(geom.detector.get_config(), config)


class TestCalcFrom(unittest.TestCase):
    """
    Test case for testing "calcfrom1d/calcfrom2d geometry
    """

    def test_calcfrom12d(self):
        det = detector_factory("pilatus300k")
        ai = AzimuthalIntegrator(0.1, 0.05, 0.04, detector=det)
        prof_1d = ai.integrate1d(numpy.random.random(det.shape), 200, unit="2th_deg")
        sig = numpy.sinc(prof_1d.radial * 10) ** 2
        img1 = ai.calcfrom1d(prof_1d.radial, sig, dim1_unit="2th_deg", mask=det.mask, dummy=-1)
        new_prof_1d = ai.integrate1d(img1, 200, unit="2th_deg")
        delta = abs((new_prof_1d.intensity - sig)).max()
        self.assertLess(delta, 2e-3, "calcfrom1d works delta=%s" % delta)
        prof_2d = ai.integrate2d(img1, 400, 360, unit="2th_deg")
        img2 = ai.calcfrom2d(prof_2d.intensity, prof_2d.radial, prof_2d.azimuthal,
                             mask=det.mask,
                             dim1_unit="2th_deg", correctSolidAngle=True, dummy=-1)
        delta2 = abs(img2 - img1).max()
        self.assertLess(delta2, 1e-3, "calcfrom2d works delta=%s" % delta2)


class TestBug474(unittest.TestCase):
    """This bug is about PONI coordinates not subtracted from x&y coodinates in Cython"""

    def test_regression(self):
        detector = detector_factory("Pilatus100K")  # small detectors makes calculation faster
        geo = geometry.Geometry(detector=detector)
        geo.setFit2D(100, detector.shape[1] // 3, detector.shape[0] // 3, tilt=1)
        rc = geo.position_array(use_cython=True)
        rp = geo.position_array(use_cython=False)
        delta = abs(rp - rc).max()
        self.assertLess(delta, 1e-5, "error on position is %s" % delta)


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestBug474))
    testsuite.addTest(loader(TestSolidAngle))
    testsuite.addTest(loader(TestBug88SolidAngle))
    testsuite.addTest(loader(TestRecprocalSpacingSquarred))
    testsuite.addTest(loader(TestCalcFrom))
    testsuite.addTest(loader(TestGeometry))
    testsuite.addTest(loader(TestFastPath))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
