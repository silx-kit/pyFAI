#!/usr/bin/env python3
#
#    Project: Fast Azimuthal Integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2026-2026 European Synchrotron Radiation Facility, Grenoble, France
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

"""Test suite for the multi_module detectors, i.e. detectors made of several
modules whose position is refined from powder diffraction data."""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "08/09/2026"

import logging
import os
import unittest

import numpy
from scipy import optimize

from .. import detector_factory
from ..detectors import Detector
from ..detectors.multi_module import (ModuleParam, MultiModule,
                                      MultiModuleRefinement, PoniParam)
from ..io.ponifile import PoniFile
from .utilstest import UtilsTest

logger = logging.getLogger(__name__)


class TestMultiModule(unittest.TestCase):
    """Tests based on a small synthetic detector split into 4 modules by a cross-shaped mask"""

    @classmethod
    def setUpClass(cls):
        cls.pixel = 1e-4
        cls.shape = (21, 21)
        detector = Detector(pixel1=cls.pixel, pixel2=cls.pixel, max_shape=cls.shape)
        mask = numpy.zeros(cls.shape, dtype=numpy.int8)
        mask[10,:] = 1  # horizontal gap
        mask[:, 10] = 1  # vertical gap
        detector.mask = mask
        cls.detector = detector

    @classmethod
    def tearDownClass(cls):
        cls.detector = None

    def build(self):
        """Build a fresh MultiModule: the tests do modify the module parameters"""
        mm = MultiModule.from_detector(self.detector)
        self.assertEqual(mm.nb_modules, 4, "the cross-shaped mask defines 4 modules")
        return mm

    def test_identity(self):
        """Without any displacement, the pixel corners are those of the parent detector"""
        mm = self.build()
        new = mm.to_detector()
        ref = self.detector.get_pixel_corners(correct_binning=True)
        self.assertIsNotNone(new._pixel_corners, "_pixel_corners is defined")
        self.assertEqual(new._pixel_corners.shape, (self.shape[0], self.shape[1], 4, 3),
                         "pixel corners have the expected shape")
        self.assertTrue(numpy.allclose(new._pixel_corners, ref),
                        "identity transformation leaves the corners unchanged")
        self.assertFalse(new.uniform_pixel, "pixel corners enforce a non uniform detector")
        self.assertFalse(new.IS_CONTIGUOUS, "pixel corners enforce a non contiguous detector")

    def test_translation(self):
        """A pure translation shifts all the pixels of one module and only those"""
        mm = self.build()
        mm.modules[1].param.set((0.5, -0.25, 0.0))
        new = mm.to_detector()
        ref = self.detector.get_pixel_corners(correct_binning=True)
        delta = new._pixel_corners.astype(numpy.float64) - ref
        mask = mm.modules[1].mask
        self.assertTrue(numpy.allclose(delta[mask][..., 1], 0.5 * self.pixel),
                        "module #1 is translated along the slow dimension")
        self.assertTrue(numpy.allclose(delta[mask][..., 2], -0.25 * self.pixel),
                        "module #1 is translated along the fast dimension")
        self.assertTrue(numpy.allclose(delta[numpy.logical_not(mask)], 0.0),
                        "the other modules and the gaps did not move")
        self.assertTrue(numpy.allclose(delta[..., 0], 0.0),
                        "the out-of-plane coordinate is left unchanged")

    def test_displacement_map(self):
        """The detector reproduces the displacement map used by the refinement"""
        mm = self.build()
        rng = numpy.random.default_rng(1234)
        for module in mm.modules.values():
            module.param.set((rng.normal(scale=0.3), rng.normal(scale=0.3), rng.normal(scale=1e-2)))
        new = mm.to_detector()
        # the displacement map provides the position of the pixel centers, in pixel units
        ref1, ref2 = mm.calc_displacement_map()
        pos1, pos2, pos3 = new.calc_cartesian_positions()
        self.assertTrue(pos3 is None or not pos3.any(), "the detector stays flat")
        # _pixel_corners is stored as float32, hence the 1e-3 pixel tolerance
        self.assertLess(abs(pos1 / self.pixel - ref1).max(), 1e-3,
                        "pixel centers match the displacement map along the slow dimension")
        self.assertLess(abs(pos2 / self.pixel - ref2).max(), 1e-3,
                        "pixel centers match the displacement map along the fast dimension")

    def test_fixed_module(self):
        """A fixed module does not move and is skipped when reading the parameter vector"""
        mm = self.build()
        mm.modules[2].fixed = True
        self.assertEqual(mm.free_modules, 3, "3 modules are still free")
        # one distinct translation per free module, in the order of the free modules
        param = numpy.array([1.0, 0.0, 0.0,
                             3.0, 0.0, 0.0,
                             4.0, 0.0, 0.0])
        new = mm.to_detector(param=param)
        delta = new._pixel_corners.astype(numpy.float64) - self.detector.get_pixel_corners(correct_binning=True)
        for module_id, expected in ((1, 1.0), (2, 0.0), (3, 3.0), (4, 4.0)):
            obtained = delta[mm.modules[module_id].mask][..., 1] / self.pixel
            self.assertTrue(numpy.allclose(obtained, expected),
                            f"module #{module_id} is shifted by {expected} pixel")

    def test_param_vector(self):
        """Passing the refined parameters is equivalent to storing them in the modules"""
        mm = self.build()
        mm.modules[3].fixed = True
        rng = numpy.random.default_rng(5678)
        param = rng.normal(scale=0.3, size=ModuleParam.nb_param * mm.free_modules)
        from_vector = mm.to_detector(param=param)

        idx = 0
        for module in mm.modules.values():
            if module.fixed:
                continue
            module.param.set(param[ModuleParam.nb_param * idx: ModuleParam.nb_param * (idx + 1)])
            idx += 1
        from_modules = mm.to_detector()
        self.assertTrue(numpy.allclose(from_vector._pixel_corners, from_modules._pixel_corners),
                        "both ways to provide the module parameters agree")

        # the poni-parameters, appended by the refinement, are ignored
        padded = numpy.concatenate((param, rng.normal(size=5)))
        self.assertTrue(numpy.allclose(mm.to_detector(param=padded)._pixel_corners,
                                       from_vector._pixel_corners),
                        "trailing poni-parameters are ignored")
        self.assertRaises(ValueError, mm.to_detector, param[:-1])

    def test_inheritance(self):
        """Pixel size, shape, orientation, mask and sensor come from the parent detector"""
        parent = detector_factory("Pilatus300k")
        parent.sensor = {"material": "Si", "thickness": 450e-6}
        mm = MultiModule.from_detector(parent)
        self.assertGreater(mm.nb_modules, 1, "the Pilatus300k is made of several modules")
        new = mm.to_detector()
        self.assertEqual(type(new), Detector,
                         "a generic detector is returned: subclasses may ignore the pixel corners")
        self.assertEqual(new.shape, parent.shape, "same shape")
        self.assertEqual((new.pixel1, new.pixel2), (parent.pixel1, parent.pixel2), "same pixel size")
        self.assertEqual(new.orientation, parent.orientation, "same orientation")
        self.assertEqual(new.sensor, parent.sensor, "same sensor")
        self.assertTrue(numpy.array_equal(new.mask, parent.mask), "same mask")

    def test_nexus_roundtrip(self):
        """The refined detector can be saved and read back"""
        mm = self.build()
        mm.modules[4].param.set((0.4, -0.6, 5e-3))
        new = mm.to_detector()
        filename = os.path.join(UtilsTest.tempdir, "multi_module.h5")
        if os.path.exists(filename):
            os.unlink(filename)
        new.save(filename)
        back = detector_factory(filename)
        self.assertTrue(numpy.allclose(back.get_pixel_corners(), new.get_pixel_corners()),
                        "pixel corners survive the NeXus round-trip")
        self.assertTrue(numpy.array_equal(back.mask, new.mask), "mask survives the NeXus round-trip")


class TestUncertainties(unittest.TestCase):
    """Tests for the estimation of the precision of the refinement"""

    @classmethod
    def setUpClass(cls):
        detector = Detector(pixel1=1e-4, pixel2=1e-4, max_shape=(21, 21))
        mask = numpy.zeros((21, 21), dtype=numpy.int8)
        mask[10,:] = 1
        mask[:, 10] = 1
        detector.mask = mask
        cls.mm = MultiModuleRefinement.from_detector(detector)

    @classmethod
    def tearDownClass(cls):
        cls.mm = None

    def test_diagonal_jacobian(self):
        """With a diagonal jacobian, the covariance matrix is known analytically"""
        jac = numpy.zeros((6, 3))
        jac[:3,:3] = numpy.diag([1.0, 2.0, 4.0])
        # sigma_i = sqrt(2 * cost / (npt - nparam)) / jac_ii, here the prefactor is 1
        result = optimize.OptimizeResult(jac=jac, cost=1.5)
        sigma = self.mm.calc_uncertainties(result)
        self.assertTrue(numpy.allclose(sigma, [1.0, 0.5, 0.25]),
                        f"unexpected standard deviation: {sigma}")

    def test_scalar_result(self):
        """A result which does not come from a least-squares optimizer is rejected"""
        result = optimize.OptimizeResult(x=numpy.zeros(3), fun=1.0)
        self.assertRaises(RuntimeError, self.mm.calc_uncertainties, result)


class TestParameterVector(unittest.TestCase):
    """Tests for the handling of the parameter vector: init_param / set_param / names"""

    @classmethod
    def setUpClass(cls):
        detector = Detector(pixel1=1e-4, pixel2=1e-4, max_shape=(21, 21))
        mask = numpy.zeros((21, 21), dtype=numpy.int8)
        mask[10,:] = 1
        mask[:, 10] = 1
        detector.mask = mask
        cls.detector = detector

    @classmethod
    def tearDownClass(cls):
        cls.detector = None

    def build(self):
        mm = MultiModuleRefinement.from_detector(self.detector)
        mm.modules[2].fixed = True
        return mm

    def test_roundtrip(self):
        """set_param followed by init_param returns the very same vector"""
        mm = self.build()
        self.assertEqual(mm.nb_param, ModuleParam.nb_param * 3,
                         "3 free modules, no registered geometry")
        self.assertTrue(numpy.array_equal(mm.init_param(), numpy.zeros(mm.nb_param)),
                        "the initial displacement is null")
        rng = numpy.random.default_rng(24)
        param = rng.normal(size=mm.nb_param)
        self.assertTrue(numpy.array_equal(mm.set_param(param), param), "set_param echoes back")
        self.assertTrue(numpy.array_equal(mm.init_param(), param),
                        "init_param reads the state back")
        self.assertTrue(numpy.array_equal(mm.param, param), "the vector is kept in `param`")
        # the fixed module did not receive anything
        self.assertEqual(mm.modules[2].param.get(), (0.0, 0.0, 0.0), "fixed module untouched")

    def test_names(self):
        """One name per parameter, the fixed module being skipped"""
        mm = self.build()
        names = mm.param_names
        self.assertEqual(len(names), mm.nb_param, "one name per parameter")
        self.assertEqual(names[0], "module1.d0", f"unexpected first name: {names[0]}")
        self.assertFalse([n for n in names if n.startswith("module2.")],
                         "the fixed module has no parameter")

    def test_wrong_size(self):
        """A vector of the wrong size is rejected"""
        mm = self.build()
        self.assertRaises(ValueError, mm.set_param, numpy.zeros(mm.nb_param + 1))
        self.assertRaises(ValueError, mm.set_param, numpy.zeros(mm.nb_param - 1))

    def test_poni_roundtrip(self):
        """The poni-parameters are stored back into the PoniFile of every geometry"""
        mm = self.build()
        key = "fake.npt"
        mm.ponis[key] = PoniFile({"poni_version": 2.1, "dist": 0.1,
                                  "poni1": 0.01, "poni2": 0.02,
                                  "rot1": 0.0, "rot2": 0.0, "rot3": 0.0,
                                  "detector": "Detector",
                                  "detector_config": {"pixel1": 1e-4, "pixel2": 1e-4}})
        mm.calibrants[key] = None  # only the number of geometries matters here
        self.assertEqual(mm.nb_param, ModuleParam.nb_param * 3 + PoniParam.nb_param,
                         "3 free modules and one geometry")
        param = numpy.arange(mm.nb_param, dtype=numpy.float64)
        mm.set_param(param)
        self.assertTrue(numpy.array_equal(mm.init_param(), param),
                        "the poni-parameters are read back from the PoniFile")
        poni = mm.ponis[key]
        self.assertAlmostEqual(poni.dist, param[-5], msg="dist updated")
        self.assertAlmostEqual(poni.poni1, param[-4], msg="poni1 updated")
        self.assertAlmostEqual(poni.rot2, param[-1], msg="rot2 updated")
        self.assertEqual(mm.param_names[-5], f"{key}.dist", "name of the poni-parameter")

    def test_to_detector_uses_the_state(self):
        """to_detector() without argument uses the parameters stored by set_param"""
        mm = self.build()
        rng = numpy.random.default_rng(25)
        param = rng.normal(scale=0.3, size=mm.nb_param)
        mm.set_param(param)
        from_state = mm.to_detector()
        from_vector = mm.to_detector(param)
        self.assertTrue(numpy.array_equal(from_state.get_pixel_corners(),
                                          from_vector.get_pixel_corners()),
                        "both ways provide the same detector")


def suite():
    testsuite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite.addTest(loader(TestMultiModule))
    testsuite.addTest(loader(TestUncertainties))
    testsuite.addTest(loader(TestParameterVector))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
