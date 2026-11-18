#!/usr/bin/env python
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2017-2026 European Synchrotron Radiation Facility, Grenoble, France
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

"""Test suite for Goniometer class and associated
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jérôme.Kieffer@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "18/11/2026"

import logging
import math
import os
import unittest

import fabio
import numpy

from .. import detectors, geometry
from ..calibrant import CALIBRANT_FACTORY
from ..goniometer import (
    ExtendedTransformation,
    GeometryTranslation,
    Goniometer,
    GoniometerRefinement,
    PoniParam,
    SingleGeometry,
    numexpr,
)
from ..integrator.azimuthal import AzimuthalIntegrator
from .utilstest import UtilsTest

logger = logging.getLogger(__name__)


@unittest.skipUnless(numexpr, "Numexpr package is missing")
class TestTranslation(unittest.TestCase):
    """
    Test the proper working of the translation class
    """

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.gt = GeometryTranslation(pos_names=["pos_dist", "pos_angle"],
                                      param_names=["dist_scale", "dist_offset",
                                                   "poni1", "poni2", "rot1",
                                                   "rot2_scale", "rot2_offset"],
                                      dist_expr="pos_dist * dist_scale + dist_offset",
                                      poni1_expr="poni1",
                                      poni2_expr="poni2",
                                      rot1_expr="rot1",
                                      rot2_expr="pos_angle * rot2_scale + rot2_offset",
                                      rot3_expr="0.0",
                                      )

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.gt = None

    @staticmethod
    def reference_function(gonio_params, motor_pos):
        """1-axis goniometer (vertical) with 1 translation on the distance

        :param gonio_params: 7 parameter model  dist_scale, dist_offset, poni1,
            poni2, rot1, rot2_scale, rot2_offset
        :param motor_pos: 2 parameters of the gonio: distance, angle
        :return 6-tuple representing pyFAI geometry
        """
        pos_dist, pos_angle = motor_pos
        dist_scale, dist_offset, poni1, poni2, rot1, rot2_scale, rot2_offset = gonio_params
        dist = pos_dist * dist_scale + dist_offset
        rot2 = pos_angle * rot2_scale + rot2_offset
        return dist, poni1, poni2, rot1, rot2, 0

    def test_serialize(self):
        # this is just to increase coverage
        str(self.gt)
        dico = self.gt.to_dict()
        new_gt = GeometryTranslation(**dico)
        self.assertEqual(str(self.gt), str(new_gt), "serialized have the same representation")

    def test_equivalent(self):
        rng = UtilsTest.get_rng()
        pos = rng.random(size=len(self.gt.pos_names))
        param = rng.random(size=len(self.gt.param_names))
        ref = numpy.array(self.reference_function(param, pos))
        obt = numpy.array(self.gt(param, pos))
        eps = abs(ref - obt).max()
        self.assertLess(eps, 1e-10, "Numexpr results looks OK.")

    def test_goniometer(self):
        g = Goniometer([1., 2., 3., 4., 5., 6., 7.], self.gt, "pilatus100k")
        fname = os.path.join(UtilsTest.tempdir, "gonio.json")
        g.save(fname)
        self.assertTrue(os.path.exists(fname), "json file written")
        g2 = Goniometer.sload(fname)
        self.assertEqual(str(g), str(g2), "goniometer description are the same")
        ai = g2.get_ai((1, 2))
        str(ai)
        mg = g.get_mg([(1, 2), (2, 3)])
        str(mg)
        if os.path.exists(fname):
            os.unlink(fname)

    def test_evaluate(self):
        param = self.gt([1, 2, 3, 4, 5, 6, 7], (0, 0))
        self.assertEqual(param, PoniParam(2, 3, 4, 5, 7, 0))


    def test_extended(self):
        jsons = """
{
  "content": "Goniometer calibration v2",
  "detector": "Imxpad S140",
  "detector_config": {},
  "wavelength": 6.888011024066681e-11,
  "param": [
    0.36741723973953894,
    0.02006383261980703,
    0.04928807714859762,
    0.32615779536127476,
    0.017464856972380878,
    -0.012690222849625982,
    17.8857568488636
  ],
  "param_names": [
    "dist",
    "poni1",
    "poni2",
    "rot1_offset",
    "rot1_scale",
    "rot2",
    "energy"
  ],
  "pos_names": [
    "pos"
  ],
  "trans_function": {
    "content": "ExtendedTransformation",
    "param_names": [
      "dist",
      "poni1",
      "poni2",
      "rot1_offset",
      "rot1_scale",
      "rot2",
      "energy"
    ],
    "pos_names": [
      "pos"
    ],
    "dist_expr": "dist",
    "poni1_expr": "poni1",
    "poni2_expr": "poni2",
    "rot1_expr": "-rot1_scale * pos - rot1_offset",
    "rot2_expr": "rot2",
    "rot3_expr": "pi/2",
    "wavelength_expr": "hc/energy*1e-10",
    "constants": {
      "pi": 3.141592653589793,
      "hc": 12.398419843320026,
      "q": 1.602176634e-19
    }
  }
}
        """

        fname = os.path.join(UtilsTest.tempdir, "extended.json")
        with open(fname, "w", encoding="utf-8") as of:
            of.write(jsons)
        gonio = Goniometer.sload(fname)
        self.assertTrue(isinstance(gonio.trans_function, ExtendedTransformation))
        gonio = GoniometerRefinement.sload(fname)
        self.assertTrue(isinstance(gonio.trans_function, ExtendedTransformation))
        param = gonio.trans_function([1,2,3,4,5,6,7], 0)
        self.assertAlmostEqual(param, gonio.trans_function.ParamNT(1, 2, 3, -4, 6, math.pi/2, 1.7712028347600038e-10))
        if os.path.exists(fname):
            os.unlink(fname)

    def test_extract_points_no_image(self):
        gonio = SingleGeometry(
          label="test_cps",
          detector="pilatus100k",
          image=None,
          geometry=None,
          calibrant=None,
        )
        with self.assertRaisesRegex(RuntimeError, "To perform control point extraction, a data image must be provided"):
          gonio.extract_cp()

    def test_extract_points_no_wavelength(self):
        gonio = SingleGeometry(
          label="test_cps",
          detector="pilatus100k",
          image=numpy.zeros((195, 487)),
          geometry=None,
          calibrant=None,
        )
        with self.assertRaisesRegex(RuntimeError, "To perform control point extraction, a wavelength must be provided"):
          gonio.extract_cp()


class TestExtractControlPoints(unittest.TestCase):
    """Quality of the control points extracted by `SingleGeometry.extract_cp`

    Reference values measured on the implementation of 2026.11 (20 runs, the extraction is
    randomized): they guard the quality of the result when `extract_cp` is optimized (#2999).
    """

    def test_extract_cp_pilatus1m(self):
        """Extraction on the first 5 rings of AgBh (the following one is not visible), then refinement"""
        with fabio.open(UtilsTest.getimage("Pilatus1M.edf")) as fimg:
            frame = fimg.data
        wl = 1e-10
        pilatus = detectors.detector_factory("Pilatus1M")
        behenate = CALIBRANT_FACTORY("AgBh")
        behenate.wavelength = wl
        initial = geometry.Geometry(detector=pilatus, wavelength=wl)
        initial.setFit2D(1600, 200, 300)  # distance in mm, beam center in pixels
        sg = SingleGeometry("pilatus1m", frame, calibrant=behenate, detector=pilatus, geometry=initial)
        cp = sg.extract_cp(max_rings=5)
        points = numpy.array(cp.getList())

        self.assertEqual(sorted(set(points[:, 2].astype(int))), [0, 1, 2, 3, 4], "5 rings extracted")
        # measured: 736 to 741 points
        self.assertTrue(725 <= len(points) <= 755, f"Number of control points: {len(points)}")

        # Rotations and wavelength are kept fixed: good practice for such a refinement
        sg.geometry_refinement.refine2(fix=["rot1", "rot2", "rot3", "wavelength"])
        # measured: chi2 between 1.1e-6 and 1.6e-6
        chi2 = sg.geometry_refinement.chi2()
        self.assertLess(chi2, 3e-6, f"chi2 after refinement: {chi2}")
        ai = sg.get_ai()
        for name, ref in (("dist", 1.63463), ("poni1", 0.0454399), ("poni2", 0.0309507)):
            self.assertAlmostEqual(getattr(ai, name), ref, delta=1e-3 * ref, msg=f"{name} after refinement")

    def test_extract_cp_synthetic(self):
        """Extraction on a noise-free simulated image (Pilatus300k): the exact geometry is known.
        No rotation, so that a refinement with fixed rotations can find it back."""
        wl = 1e-10
        pilatus = detectors.detector_factory("Pilatus300k")
        behenate = CALIBRANT_FACTORY("AgBh")
        behenate.wavelength = wl
        truth = AzimuthalIntegrator(dist=0.25,
                                    poni1=pilatus.shape[0] * pilatus.pixel1 * 0.4,
                                    poni2=pilatus.shape[1] * pilatus.pixel2 * 0.4,
                                    detector=pilatus, wavelength=wl)
        image = behenate.fake_calibration_image(truth)

        # Wrong guess: 2% on the distance, a few pixels on the beam center
        fit2d = truth.getFit2D()
        initial = geometry.Geometry(detector=pilatus, wavelength=wl)
        initial.setFit2D(truth.dist * 1000 * 1.02, fit2d["centerX"] + 4, fit2d["centerY"] - 3)
        sg = SingleGeometry("synthetic", image, calibrant=behenate, detector=pilatus, geometry=initial)
        cp = sg.extract_cp(max_rings=5)
        points = numpy.array(cp.getList())

        self.assertEqual(sorted(set(points[:, 2].astype(int))), [0, 1, 2, 3, 4], "5 rings extracted")
        # measured: 431 or 432 points
        self.assertTrue(400 <= len(points) <= 460, f"Number of control points: {len(points)}")
        # Every point lies on its ring of the exact geometry; measured: 0.13 mrad (a pixel is about 0.7 mrad)
        tth_rings = numpy.array([i for i in behenate.get_2th() if i is not None])
        error = numpy.abs(truth.tth(points[:, 0], points[:, 1]) - tth_rings[points[:, 2].astype(int)]).max()
        self.assertLess(error, 0.25e-3, f"Control points are off their ring by {error * 1e3:.3f} mrad")

        sg.geometry_refinement.refine2(fix=["rot1", "rot2", "rot3", "wavelength"])
        # measured: chi2 = 8.3e-8
        chi2 = sg.geometry_refinement.chi2()
        self.assertLess(chi2, 3e-7, f"chi2 after refinement: {chi2}")
        ai = sg.get_ai()
        # measured: 2e-5 m on the distance, 3e-7 m on the PONI
        self.assertAlmostEqual(ai.dist, truth.dist, delta=1e-3 * truth.dist, msg="dist after refinement")
        self.assertAlmostEqual(ai.poni1, truth.poni1, delta=2e-6, msg="poni1 after refinement")
        self.assertAlmostEqual(ai.poni2, truth.poni2, delta=2e-6, msg="poni2 after refinement")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestTranslation))
    testsuite.addTest(loader(TestExtractControlPoints))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
