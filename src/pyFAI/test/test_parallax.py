#!/usr/bin/env python
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2021-2025 European Synchrotron Radiation Facility, Grenoble, France
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

"""Test suites for parallax correction"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "06/10/2026"

import logging
import unittest

import numpy

from .. import load
from ..calibrant import get_calibrant
from ..detectors import Detector
from ..detectors.sensors import CdTe_MATERIAL, SensorConfig, Si_MATERIAL
from ..ext.parallax_raytracing import Raytracing
from ..io.ponifile import PoniFile
from ..geometryRefinement import GeometryRefinement
from ..parallax import BaseSensor, Beam, Parallax, ThinSensor
from ..test.utilstest import UtilsTest

logger = logging.getLogger(__name__)

class TestSensorMaterial(unittest.TestCase):
    """test pyFAI.detectors.sensors"""
    def test_Si(self):
        self.assertTrue(numpy.allclose(Si_MATERIAL.mu(20), 10.396656))
        self.assertTrue(numpy.allclose(Si_MATERIAL.mu_en(20), 9.493004))

    def test_CdTe(self):
        self.assertTrue(numpy.allclose(CdTe_MATERIAL.mu(40), 112.905))
        self.assertTrue(numpy.allclose(CdTe_MATERIAL.mu_en(40), 56.36475))

    def test_bug_2851(self):
        """test bug #2851: SensorConfig.from_dict() initializes the SenorMaterial
        while the default constructor does not"""
        c1 = SensorConfig.from_dict({"material": "Si", "thickness": 0.00045})
        c2 = SensorConfig("Si", 0.00045)
        self.assertEqual(c1, c2)


class TestParallax(unittest.TestCase):
    """Test Azimuthal integration based sparse matrix multiplication methods
    Bounding box pixel splitting
    """
    def test_beam(self):
        width = 1e-3
        for profile in ("gaussian", "circle", "square"):
            beam =  Beam(width, profile)
            x,y = beam()
            self.assertGreaterEqual(x[-1]-x[0], width, "{profile} profile is large enough")
            self.assertTrue(numpy.isclose(y.sum(), 1.0), "intensity are normalized")

    def test_decay(self):
        t = ThinSensor(450e-6, 0.3)
        self.assertTrue(isinstance(t, BaseSensor))
        self.assertTrue(t.test(), msg="autotest OK")

    def test_serialize1(self):
        beam = Beam(1e-3)
        sensor=ThinSensor(1e-3, 0.3)
        p = Parallax(beam=beam, sensor=sensor)
        q=Parallax()
        q.set_config(p.get_config())
        self.assertEqual(str(p), str(q))

    def test_serialize2(self):
        beam = Beam(1e-3)
        sensor=BaseSensor(1e-3)
        p = Parallax(beam=beam, sensor=sensor)
        q=Parallax()
        q.set_config(p.get_config())
        self.assertEqual(str(p), str(q))


class TestActivation(unittest.TestCase):
    def test_activation(self):
        a = load({"detector":"Pilatus1M", "wavelength":1e-10})
        self.assertFalse(bool(a.parallax))
        p0 = PoniFile(a)
        self.assertEqual(p0.as_dict()["poni_version"], 2.1)
        a.detector.sensor = SensorConfig(Si_MATERIAL, 320e-6)
        p1 = PoniFile(a)
        self.assertEqual(p1.as_dict()["poni_version"], 2.1)
        a.enable_parallax()
        p2 = PoniFile(a)
        self.assertGreaterEqual(p2.as_dict()["poni_version"], 3)
        a.save(UtilsTest.temp_path/"test_activation.poni")
        b = load(UtilsTest.temp_path/"test_activation.poni")
        # print("a",a)
        # print("b",b)
        # with open(UtilsTest.temp_path/"test_activation.poni") as f:
        #     print(f.read())
        self.assertEqual(PoniFile(a),PoniFile(b), "ponifiles are the same")
        self.assertEqual(str(a),str(b), "geometries are the same")


class TestParallaxRefinement(unittest.TestCase):
    """The parallax correction has to follow the geometry under refinement.

    The minimizer of `GeometryRefinement` evaluates trial parameters through
    `Geometry.tth(d1, d2, param)` **without** storing them in the geometry. A
    correction which reads the distance or the PONI from `self` instead of from
    `param` stays frozen on the starting geometry: it then acts as a constant
    offset, which biases the fit and makes the returned residual meaningless.
    """

    PIXEL = 75e-6
    SHAPE = (2167, 2070)
    WAVELENGTH = 0.9218e-10
    THICKNESS = 450e-6
    # true geometry used to build the synthetic control points
    DIST = 0.1
    PONI1 = 0.0812
    PONI2 = 0.0777
    data = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.detector = Detector(cls.PIXEL, cls.PIXEL, max_shape=cls.SHAPE)
        cls.calibrant = get_calibrant("LaB6")
        cls.calibrant.wavelength = cls.WAVELENGTH
        cls.sensor = cls.build_geometry(data=numpy.zeros((1, 3))).parallax.sensor
        cls.data = cls.control_points()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls.detector = cls.calibrant = cls.sensor = cls.data = None

    @classmethod
    def build_geometry(cls, parallax=True, data=None, **kwargs):
        """Build a GeometryRefinement, with the parallax correction enabled or not"""
        param = {"dist": cls.DIST, "poni1": cls.PONI1, "poni2": cls.PONI2,
                 "rot1": 0.0, "rot2": 0.0, "rot3": 0.0}
        param.update(kwargs)
        geo = GeometryRefinement(cls.data if data is None else data,
                                 detector=cls.detector,
                                 wavelength=cls.WAVELENGTH, calibrant=cls.calibrant,
                                 **param)
        if parallax:
            geo.enable_parallax(True, sensor_material="Si",
                                sensor_thickness=cls.THICKNESS)
        return geo

    @classmethod
    def control_points(cls):
        """Synthetic control points, displaced by the parallax effect.

        Without any tilt the incidence angle is the scattering angle, hence the
        observed radius is simply `dist*tan(2th)` plus the displacement.
        """
        chi = numpy.linspace(0, 2 * numpy.pi, 60, endpoint=False)
        cos_chi, sin_chi = numpy.cos(chi), numpy.sin(chi)
        rows, cols, rings = [], [], []
        two_th = [t for t in cls.calibrant.get_2th() if t < numpy.radians(55)]
        for idx, tth in enumerate(two_th):
            radius = cls.DIST * numpy.tan(tth) + cls.sensor.measure_displacement_integrate(tth)
            d1 = (cls.PONI1 + radius * cos_chi) / cls.PIXEL - 0.5
            d2 = (cls.PONI2 + radius * sin_chi) / cls.PIXEL - 0.5
            valid = ((d1 > 0) & (d1 < cls.SHAPE[0]) &
                     (d2 > 0) & (d2 < cls.SHAPE[1]))
            rows.append(d1[valid])
            cols.append(d2[valid])
            rings.append(numpy.full(valid.sum(), idx))
        return numpy.vstack((numpy.concatenate(rows),
                             numpy.concatenate(cols),
                             numpy.concatenate(rings))).T

    def test_displacement_follows_param(self):
        """Non regression: the correction used to be read from `self`, hence it was
        the very same for every trial geometry (bug seen with parallax refinement)."""
        geo = self.build_geometry()
        d1 = numpy.array([100.0, 600.0, 1200.0, 1800.0])
        d2 = numpy.array([100.0, 600.0, 1200.0, 1800.0])
        reference = None
        for dist in (0.05, 0.1, 0.3):
            p1, p2, p3 = geo.detector.calc_cartesian_positions(d1, d2)
            p1 = (p1 - self.PONI1).ravel()
            p2 = (p2 - self.PONI2).ravel()
            p3 = numpy.zeros(p1.size) + dist if p3 is None else (dist + p3).ravel()
            before = numpy.hypot(p1, p2)
            geo._correct_parallax(p1, p2, p3, dist)
            # the correction undoes the parallax displacement, hence it is inwards
            correction = numpy.hypot(p1, p2) - before
            self.assertTrue((correction < 0).all(), "correction is inwards")
            if reference is None:
                reference = correction
            else:
                self.assertFalse(numpy.allclose(reference, correction),
                                 "correction depends on the trial distance")

    def test_tth_follows_param(self):
        """`tth(d1, d2, param)` is the entry point of the minimizer: it goes through
        `calc_pos_zyx(param=...)`, hence the correction has to follow `param` too.
        """
        geo = self.build_geometry()
        d1 = numpy.array([100.0, 600.0, 1800.0])
        d2 = numpy.array([100.0, 600.0, 1800.0])
        # same trial geometry as the one stored: parallax must not change anything else
        param = [self.DIST, self.PONI1, self.PONI2, 0.0, 0.0, 0.0]
        reference = geo.tth(d1, d2, param)
        self.assertTrue(numpy.allclose(reference, geo.tth(d1, d2)),
                        "tth(param) matches tth() for the stored geometry")
        for dist in (0.05, 0.3):
            other = geo.tth(d1, d2, [dist] + param[1:])
            # the parallax correction scales with the distance: a geometry-only
            # calculation would give exactly arctan(r/dist) here.
            naked = numpy.arctan(geo.rFunction(d1, d2, param) / dist)
            self.assertFalse(numpy.allclose(other, naked, atol=1e-7),
                             f"the correction is applied for dist={dist}")

    def test_refinement_is_param_aware(self):
        """A single refine3() must recover the true geometry whatever the starting
        point. With the frozen correction, starting 20% off biased the distance by
        more than 20µm and left a residual 2 orders of magnitude too large."""
        for start in (0.1015, 0.105, 0.12):
            geo = self.build_geometry(dist=start, poni1=0.0805, poni2=0.0785)
            geo.refine3(10000, fix=["rot3", "wavelength"])
            npt = self.data.shape[0]
            rms = numpy.degrees(numpy.sqrt(geo.chi2() / npt)) * 1e3
            self.assertAlmostEqual(geo.dist, self.DIST, delta=2e-6,
                                   msg=f"distance recovered from {start}m")
            self.assertAlmostEqual(geo.poni1, self.PONI1, delta=2e-6,
                                   msg=f"poni1 recovered from {start}m")
            self.assertAlmostEqual(geo.poni2, self.PONI2, delta=2e-6,
                                   msg=f"poni2 recovered from {start}m")
            self.assertLess(rms, 0.1, msg=f"residual below 0.1 mdeg from {start}m")

    def test_refinement_without_parallax(self):
        """Sanity check: without the correction the fit absorbs the displacement
        into the distance, which is off by about the mean depth of absorption."""
        geo = self.build_geometry(parallax=False, dist=0.1015,
                                  poni1=0.0805, poni2=0.0785)
        self.assertIsNone(geo.parallax)
        geo.refine3(10000, fix=["rot3", "wavelength"])
        self.assertGreater(geo.dist - self.DIST, 1e-4,
                           "distance is biased by more than 100µm")


class TestRaytracing(unittest.TestCase):
    def test_extension(self):
        """Simple test that validates the extension works"""
        ai = load({"detector": "Pilatus 100k",
                         "detector_config":{"sensor": {"material":"Si", "thickness":1e-3}},
                         "distance": 1e-1,
                         "wavelength":5e-11})
        ai.enable_parallax(True)
        self.assertAlmostEqual(ai.parallax.sensor.efficiency, 0.5041, delta=1e-4)
        rt=Raytracing(ai)
        data, indices, indptr = rt.calc_csr(1)
        self.assertEqual(indptr.size-1, numpy.prod(ai.detector.shape))
        self.assertEqual((indptr[1:] - indptr[:-1]).max(), 8)
        self.assertEqual(data.size, indices.size)
        self.assertAlmostEqual(data.size/numpy.prod(ai.detector.shape), 3.56, delta=4e-3)


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestParallax))
    testsuite.addTest(loader(TestSensorMaterial))
    testsuite.addTest(loader(TestActivation))
    testsuite.addTest(loader(TestParallaxRefinement))
    testsuite.addTest(loader(TestRaytracing))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
