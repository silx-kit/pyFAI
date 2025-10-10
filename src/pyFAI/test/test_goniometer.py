#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2017-2025 European Synchrotron Radiation Facility, Grenoble, France
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
__date__ = "10/10/2025"

import os
import unittest
import logging
from .utilstest import UtilsTest
import numpy
from ..goniometer import GeometryTranslation, Goniometer, numexpr, \
                         ExtendedTransformation, GoniometerRefinement
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
        with open(fname, "w") as of:
            of.write(jsons)
        gonio = Goniometer.sload(fname)
        self.assertTrue(isinstance(gonio.trans_function, ExtendedTransformation))
        gonio = GoniometerRefinement.sload(fname)
        self.assertTrue(isinstance(gonio.trans_function, ExtendedTransformation))
        if os.path.exists(fname):
            os.unlink(fname)

def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestTranslation))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
