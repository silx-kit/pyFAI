#!/usr/bin/env python
# coding: utf-8
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
__date__ = "21/11/2025"

import unittest
import numpy
import logging
from ..parallax import Beam, ThinSensor, BaseSensor, Parallax
from ..detectors.sensors import Si_MATERIAL, CdTe_MATERIAL, SensorConfig
from .. import load
from ..io.ponifile import PoniFile
from ..test.utilstest import UtilsTest
from ..ext.parallax_raytracing import Raytracing

logger = logging.getLogger(__name__)

class TestSensorMaterial(unittest.TestCase):
    """test pyFAI.detectors.sensors"""
    def test_Si(self):
        self.assertTrue(numpy.allclose(Si_MATERIAL.mu(20), 10.396656))
        self.assertTrue(numpy.allclose(Si_MATERIAL.mu_en(20), 9.493004))

    def test_CdTe(self):
        self.assertTrue(numpy.allclose(CdTe_MATERIAL.mu(40), 112.905))
        self.assertTrue(numpy.allclose(CdTe_MATERIAL.mu_en(40), 56.36475))



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
    testsuite.addTest(loader(TestRaytracing))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
