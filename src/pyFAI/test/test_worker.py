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

"Test suite for worker"

__author__ = "Valentin Valls"
__contact__ = "valentin.valls@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "07/01/2025"

import unittest
import logging
import os.path
import shutil
import numpy

from .. import units
from .. import worker as worker_mdl
from ..worker import Worker, PixelwiseWorker
from ..integrator.azimuthal import AzimuthalIntegrator
from ..containers import Integrate1dResult
from ..containers import Integrate2dResult
from ..io.integration_config import ConfigurationReader, WorkerConfig
from ..io.ponifile import PoniFile
from .. import detector_factory
from . import utilstest

logger = logging.getLogger(__name__)


class AzimuthalIntegratorMocked():

    def __init__(self, result=None):
        self._integrate1d_called = 0
        self._integrate2d_called = 0
        self._result = result
        self._csr_integrator = 0
        self._lut_integrator = numpy.array([0])

    def integrate1d(self, **kargs):
        self._integrate1d_called += 1
        self._integrate1d_kargs = kargs
        if isinstance(self._result, Exception):
            raise self._result
        return self._result

    integrate1d_ng = integrate1d_legacy = integrate1d

    def integrate2d(self, **kargs):
        self._integrate2d_called += 1
        self._integrate2d_kargs = kargs
        if isinstance(self._result, Exception):
            raise self._result
        return self._result

    integrate2d_ng = integrate2d_legacy = integrate2d


class MockedAiWriter():

    def __init__(self, result=None):
        self._write_called = 0
        self._close_called = 0
        self._write_kargs = {}

    def write(self, data):
        self._write_called += 1
        self._write_kargs["data"] = data

    def close(self):
        self._close_called += 1


class TestWorker(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super(TestWorker, cls).setUpClass()
        cls.rng = utilstest.UtilsTest.get_rng()

    @classmethod
    def tearDownClass(cls) -> None:
        super(TestWorker, cls).tearDownClass()
        cls.rng = None

    def test_constructor_ai(self):
        ai = AzimuthalIntegrator()
        w = Worker(ai)
        self.assertIsNotNone(w)

    def test_constructor(self):
        w = Worker()
        self.assertIsNotNone(w)

    def test_process_1d(self):
        ai_result = Integrate1dResult(numpy.array([0, 1]), numpy.array([2, 3]))
        ai = AzimuthalIntegratorMocked(result=ai_result)
        worker = Worker(ai, shapeOut=(1, 10))
        data = numpy.array([0])
        worker.unit = units.TTH_DEG
        worker.dummy = "b"
        worker.delta_dummy = "c"
        worker.method = "lut"
        worker.polarization_factor = "e"
        worker.correct_solid_angle = "f"
        worker.nbpt_rad = "g"
        worker.output = "numpy"
        worker.update_processor()
        result = worker.process(data)

        # ai calls
        self.assertEqual(ai._integrate1d_called, 1)
        self.assertEqual(ai._integrate2d_called, 0)
        ai_args = ai._integrate1d_kargs
        self.assertEqual(ai_args["unit"], worker.unit)
        self.assertEqual(ai_args["dummy"], worker.dummy)
        self.assertEqual(ai_args["delta_dummy"], worker.delta_dummy)
        self.assertTrue(worker.method in str(ai_args["method"]).lower())
        self.assertEqual(ai_args["polarization_factor"], worker.polarization_factor)
        self.assertEqual(ai_args["safe"], True)
        self.assertEqual(ai_args["data"], data)
        self.assertEqual(ai_args["correctSolidAngle"], worker.correct_solid_angle)
        self.assertEqual(ai_args["npt"], worker.nbpt_rad)

        # result
        self.assertEqual(result.tolist(), [[0, 2], [1, 3]])
        self.assertEqual(worker.radial.tolist(), [0, 1])
        self.assertEqual(worker.azimuthal, None)

    def test_process_2d(self):
        ai_result = Integrate2dResult(numpy.array([0]), numpy.array([1]), numpy.array([2]))
        ai = AzimuthalIntegratorMocked(result=ai_result)
        worker = Worker(ai, shapeOut=(10, 10))
        data = numpy.array([0])
        worker.unit = units.TTH_RAD
        worker.dummy = "b"
        worker.delta_dummy = "c"
        worker.method = "lut"
        worker.polarization_factor = "e"
        worker.correct_solid_angle = "f"
        worker.nbpt_rad = "g"
        worker.output = "numpy"
        worker.update_processor()
        result = worker.process(data)

        # ai calls
        self.assertEqual(ai._integrate1d_called, 0)
        self.assertEqual(ai._integrate2d_called, 1)
        ai_args = ai._integrate2d_kargs
        self.assertEqual(ai_args["unit"], worker.unit)
        self.assertEqual(ai_args["dummy"], worker.dummy)
        self.assertEqual(ai_args["delta_dummy"], worker.delta_dummy)
        self.assertTrue(worker.method in str(ai_args["method"]).lower())
        self.assertEqual(ai_args["polarization_factor"], worker.polarization_factor)
        self.assertEqual(ai_args["safe"], True)
        self.assertEqual(ai_args["data"], data)
        self.assertEqual(ai_args["correctSolidAngle"], worker.correct_solid_angle)
        self.assertEqual(ai_args["npt_rad"], worker.nbpt_rad)
        self.assertEqual(ai_args["npt_azim"], worker.nbpt_azim)

        # result
        self.assertEqual(result.tolist(), [0])
        self.assertEqual(worker.radial.tolist(), [1])
        self.assertEqual(worker.azimuthal.tolist(), [2])

    def test_1d_writer(self):
        ai_result = Integrate1dResult(numpy.array([0]), numpy.array([1]))
        ai = AzimuthalIntegratorMocked(result=ai_result)
        data = numpy.array([0])
        worker = Worker(ai, shapeOut=(1, 10))
        worker.unit = units.TTH_DEG
        worker.dummy = "b"
        worker.delta_dummy = "c"
        worker.method = "d"
        worker.polarization_factor = "e"
        worker.correct_solid_angle = "f"
        worker.nbpt_rad = "g"
        # worker.nbpt_azim = 1
        worker.output = "numpy"

        writer = MockedAiWriter()
        _result = worker.process(data, writer=writer)

        self.assertEqual(writer._write_called, 1)
        self.assertEqual(writer._close_called, 0)
        self.assertIs(writer._write_kargs["data"], ai_result)

    def test_2d_writer(self):
        ai_result = Integrate2dResult(numpy.array([0]), numpy.array([1]), numpy.array([2]))
        ai = AzimuthalIntegratorMocked(result=ai_result)
        data = numpy.array([0])
        worker = Worker(ai, shapeOut=(1, 10))
        worker.unit = units.TTH_DEG
        worker.dummy = "b"
        worker.delta_dummy = "c"
        worker.method = "d"
        worker.polarization_factor = "e"
        worker.correct_solid_angle = "f"
        worker.nbpt_rad = "g"
        # worker.nbpt_azim = 1
        worker.output = "numpy"

        writer = MockedAiWriter()
        _result = worker.process(data, writer=writer)

        self.assertEqual(writer._write_called, 1)
        self.assertEqual(writer._close_called, 0)
        self.assertIs(writer._write_kargs["data"], ai_result)

    def test_process_exception(self):
        ai = AzimuthalIntegratorMocked(result=Exception("Out of memory"))
        worker = Worker(ai)
        data = numpy.array([0])
        # worker.nbpt_azim = 2
        try:
            worker.process(data)
        except Exception:
            pass

    def test_process_error_model(self):
        ai_result = Integrate1dResult(numpy.array([0]), numpy.array([1]))
        ai = AzimuthalIntegratorMocked(result=ai_result)
        worker = Worker(ai, shapeOut=(1, 10))
        data = numpy.array([0])
        # worker.nbpt_azim = 1
        for error_model in ["poisson", "azimuthal", None]:
            worker.error_model = error_model
            worker.process(data)
            if error_model:
                self.assertIn("error_model", ai._integrate1d_kargs)
                self.assertEqual(ai._integrate1d_kargs["error_model"], error_model)
            else:
                self.assertNotIn("error_model", ai._integrate1d_kargs)

    def test_process_no_output(self):
        ai_result = Integrate1dResult(numpy.array([0]), numpy.array([1]))
        ai = AzimuthalIntegratorMocked(result=ai_result)
        worker = Worker(ai, shapeOut=(1, 10))
        data = numpy.array([0])
        worker.output = None
        # worker.nbpt_azim = 1
        worker.do_poisson = True
        result = worker.process(data)
        self.assertIsNone(result)

    def test_pixelwiseworker(self):
        shape = (5, 7)
        size = numpy.prod(shape)
        ref = self.rng.uniform(0, 60000, size=size).reshape(shape).astype("uint16")
        dark = self.rng.poisson(10, size=size).reshape(shape).astype("uint16")
        raw = ref + dark
        flat = self.rng.normal(1.0, 0.1, size=size).reshape(shape)
        signal = ref / flat
        precision = 1e-2

        # Without error propagation
        # Numpy path
        worker_mdl.USE_CYTHON = False
        pww = PixelwiseWorker(dark=dark, flat=flat, dummy=-5, dtype="float64")
        res_np = pww.process(raw, normalization_factor=6.0)
        err = abs(res_np - signal / 6.0).max()
        self.assertLess(err, precision, "Numpy calculation are OK: %s" % err)

        # Cython path
        worker_mdl.USE_CYTHON = True
        res_cy = pww.process(raw, normalization_factor=7.0)
        err = abs(res_cy - signal / 7.0).max()
        self.assertLess(err, precision, "Cython calculation are OK: %s" % err)

        # With Poissonian errors
        # Numpy path
        worker_mdl.USE_CYTHON = False
        pww = PixelwiseWorker(dark=dark)
        res_np, err_np = pww.process(raw, variance=ref, normalization_factor=2.0)
        delta_res = abs(res_np - ref / 2.0).max()
        delta_err = abs(err_np - numpy.sqrt(ref) / 2.0).max()

        self.assertLess(delta_res, precision, "Numpy intensity calculation are OK: %s" % err)
        self.assertLess(delta_err, precision, "Numpy error calculation are OK: %s" % err)

        # Cython path
        worker_mdl.USE_CYTHON = True
        res_cy, err_cy = pww.process(raw, variance=ref, normalization_factor=2.0)
        delta_res = abs(res_cy - ref / 2.0).max()
        delta_err = abs(err_cy - numpy.sqrt(ref) / 2.0).max()
        self.assertLess(delta_res, precision, "Cython intensity calculation are OK: %s" % err)
        self.assertLess(delta_err, precision, "Cython error calculation are OK: %s" % err)

    def test_sigma_clip(self):
        ai = AzimuthalIntegrator.sload({"detector": "Imxpad S10", "wavelength":1e-10})
        worker = Worker(azimuthalIntegrator=ai,
                        extra_options={"thres":2, "error_model":"azimuthal"},
                        integrator_name="sigma_clip_ng",
                        shapeOut=(1, 100))

        self.assertEqual(worker.error_model.as_str(), "azimuthal", "error model is OK")
        img = self.rng.random(ai.detector.shape)
        worker(img)

    def test_sigma_clip_legacy(self):
        "Non regression test for #1825"
        ai = AzimuthalIntegrator.sload({"detector": "Imxpad S10", "wavelength":1e-10})
        worker = Worker(azimuthalIntegrator=ai,
                        extra_options={"thres":2},
                        integrator_name="sigma_clip_legacy",
                        shapeOut=(1, 100))
        img = self.rng.random(ai.detector.shape)
        worker(img)

    def test_default_shape(self):
        "Non regression test for #2084"
        ai = AzimuthalIntegrator.sload({"detector":"Eiger1M",
                                        "distance":0.1,
                                        "wavelength":1e-10})
        w = Worker(ai)
        self.assertEqual(w.shape, ai.detector.shape, "detector shape matches")

class TestWorkerConfig(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.directory = os.path.join(utilstest.test_options.tempdir, cls.__name__)
        os.makedirs(cls.directory)

        cls.a = os.path.join(cls.directory, "a.npy")
        cls.b = os.path.join(cls.directory, "b.npy")
        cls.c = os.path.join(cls.directory, "c.npy")
        cls.d = os.path.join(cls.directory, "d.npy")

        cls.shape = (2, 2)
        ones = numpy.ones(shape=cls.shape)
        numpy.save(cls.a, ones)
        numpy.save(cls.b, ones * 2)
        numpy.save(cls.c, ones * 3)
        numpy.save(cls.d, ones * 4)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory)

    def test_flatdark(self):
        config = {"version": 2,
                  "application": "pyfai-integrate",
                  "dark_current": [self.a, self.b, self.c],
                  "flat_field": [self.a, self.b, self.d],
                  "poni": utilstest.UtilsTest.getimage("Pilatus1M.poni"),
                  "detector": "Detector",
                  "detector_config": {"pixel1": 1, "pixel2": 1, "max_shape": (2, 2)},
                  "do_2D": False,
                  "nbpt_rad": 2,
                  "do_solid_angle": False,
                  "method": "splitbbox"}
        worker = Worker()
        worker.validate_config(config)
        worker.set_config(config)
        data = numpy.ones(shape=self.shape)
        worker.process(data=data)
        self.assertTrue(numpy.isclose(worker.ai.detector.get_darkcurrent()[0, 0], (1 + 2 + 3) / 3))
        self.assertTrue(numpy.isclose(worker.ai.detector.get_flatfield()[0, 0], (1 + 2 + 4) / 3))

    def test_reload(self):
        config = {"version": 2,
                  "application": "pyfai-integrate",
                  "dark_current": [self.a, self.b, self.c],
                  "flat_field": [self.a, self.b, self.d],
                  "poni": utilstest.UtilsTest.getimage("Pilatus1M.poni"),
                  "detector": "Detector",
                  "detector_config": {"pixel1": 1, "pixel2": 1, "max_shape": (2, 2)},
                  "do_2D": False,
                  "nbpt_rad": 2,
                  "do_solid_angle": False,
                  "method": "splitbbox"}
        worker = Worker()
        worker.validate_config(config)
        worker.set_config(config)
        new_config = worker.get_config()
        new_worker = Worker()
        new_worker.validate_config(new_config)
        new_worker.set_config(new_config)
        # test ai
        ai = AzimuthalIntegrator.sload(new_config)
        self.assertEqual(ai.detector.shape, (2,2), "detector shape matches")

    def test_old(self):
        """bug 1991"""
        config = {'unit': 'q_nm^-1',
                 'dist': 0.1999693237019301,
                 'poni1': 0.12279243634743776,
                 'poni2': 0.11803581502718556,
                 'rot1': 0.013977781483742164,
                 'rot3': -7.470130596383977e-05,
                 'rot2': -0.013837145398466972,
                 'pixel1': 7.5e-05,
                 'pixel2': 7.5e-05,
                 'splineFile': None,
                 'wavelength': 3.7380000000000004e-11,
                 'nbpt_azim': 1,
                 'nbpt_rad': 5000,
                 'polarization_factor': 0.99,
                 'dummy': None,
                 'delta_dummy': None,
                 'correct_solid_angle': True,
                 'dark_current_image': None,
                 'flat_field_image': None,
                 'mask_image': 'DAC-04-mask.npy',
                 'error_model': 'poisson',
                 'shape': [3262, 3108],
                 'method': [1, 'full', 'csr', 'opencl', 'gpu'],
                 'do_azimuth_range': False,
                 'do_radial_range': False}

        worker = Worker()
        worker.validate_config(config)
        worker.set_config(config)

        ai = worker.ai
        self.assertTrue(numpy.allclose(config["shape"], ai.detector.max_shape))

    def test_regression_2227(self):
        """pixel size got lost with generic detector"""
        worker_generic = Worker()
        integration_options_generic = {'poni_version': 2.1,
                                       'detector': 'Detector',
                                       'detector_config': {'pixel1': 1e-4,
                                                           'pixel2': 1e-4,
                                                           'max_shape': [576, 748],
                                                           'orientation': 3},
                                       'dist': 0.16156348926909264,
                                       'poni1': 1.4999999999999998,
                                       'poni2': 1.4999999999999998,
                                       'rot1': 1.0822864853552985,
                                       'rot2': -0.41026007690779387,
                                       'rot3': 0.0,
                                       'wavelength': 1.0332016536100021e-10}
        worker_generic.set_config(integration_options_generic)
        self.assertEqual(worker_generic.ai.detector.pixel1, 1e-4, "Pixel1 size matches")
        self.assertEqual(worker_generic.ai.detector.pixel2, 1e-4, "Pixel2 size matches")
        self.assertEqual(worker_generic.ai.detector.shape, [576, 748], "Shape matches")
        self.assertEqual(worker_generic.ai.detector.orientation, 3, "Orientation matches")

    def test_regression_v4(self):
        """loading poni dictionary as a separate key in configuration"""
        detector = detector_factory(name='Eiger2_4M', config={"orientation" : 3})
        ai = AzimuthalIntegrator(dist=0.1,
                                 poni1=0.1,
                                 poni2=0.1,
                                 wavelength=1e-10,
                                 detector=detector,
                                 )
        worker = Worker(azimuthalIntegrator=ai)

        self.assertEqual(ai.dist, worker.ai.dist, "Distance matches")
        self.assertEqual(ai.poni1, worker.ai.poni1, "PONI1 matches")
        self.assertEqual(ai.poni2, worker.ai.poni2, "PONI2 matches")
        self.assertEqual(ai.rot1, worker.ai.rot1, "Rot1 matches")
        self.assertEqual(ai.rot2, worker.ai.rot2, "Rot2 matches")
        self.assertEqual(ai.rot3, worker.ai.rot3, "Rot3 matches")
        self.assertEqual(ai.wavelength, worker.ai.wavelength, "Wavelength matches")
        self.assertEqual(ai.detector, worker.ai.detector, "Detector matches")

        config = worker.get_config()
        config_reader = ConfigurationReader(config)

        detector_from_reader = config_reader.pop_detector()
        self.assertEqual(detector, detector_from_reader, "Detector from reader matches")

        config = worker.get_config()
        config_reader = ConfigurationReader(config)
        poni = config_reader.pop_ponifile()

        self.assertEqual(ai.dist, poni.dist, "Distance matches")
        self.assertEqual(ai.poni1, poni.poni1, "PONI1 matches")
        self.assertEqual(ai.poni2, poni.poni2, "PONI2 matches")
        self.assertEqual(ai.rot1, poni.rot1, "Rot1 matches")
        self.assertEqual(ai.rot2, poni.rot2, "Rot2 matches")
        self.assertEqual(ai.rot3, poni.rot3, "Rot3 matches")
        self.assertEqual(ai.wavelength, poni.wavelength, "Wavelength matches")
        self.assertEqual(ai.detector, poni.detector, "Detector matches")

    def test_v3_equal_to_v4(self):
        """checking equivalence between v3 and v4"""
        config_v3 = {
            "application": "pyfai-integrate",
            "version": 3,
            "wavelength": 1e-10,
            "dist": 0.1,
            "poni1": 0.1,
            "poni2": 0.2,
            "rot1": 1,
            "rot2": 2,
            "rot3": 3,
            "detector": "Eiger2_4M",
            "detector_config": {
                "orientation": 3
            },
        }

        config_v4 = {
            "application": "pyfai-integrate",
            "version": 4,
            "poni": {
                "wavelength": 1e-10,
                "dist": 0.1,
                "poni1": 0.1,
                "poni2": 0.2,
                "rot1": 1,
                "rot2": 2,
                "rot3": 3,
                "detector": "Eiger2_4M",
                "detector_config": {
                    "orientation": 3
                }
            },
        }

        worker_v3 = Worker()
        worker_v3.set_config(config=config_v3)
        worker_v4 = Worker()
        worker_v4.set_config(config=config_v4)
        self.assertEqual(worker_v3.get_config(), worker_v4.get_config(), "Worker configs match")

        ai_config_v3 = worker_v3.ai.get_config()
        ai_config_v4 = worker_v4.ai.get_config()
        self.assertEqual(ai_config_v3, ai_config_v4, "AI configs match")

        poni_v3 = PoniFile(data=ai_config_v3)
        poni_v4 = PoniFile(data=ai_config_v4)
        self.assertEqual(poni_v3.as_dict(), poni_v4.as_dict(), "PONI dictionaries match")

        poni_v3_from_config = PoniFile(data=config_v3)
        poni_v4_from_config = PoniFile(data=config_v4)
        self.assertEqual(poni_v3_from_config.as_dict(), poni_v4_from_config.as_dict(), "PONI dictionaries from config match")

    def test_bug2230(self):
        integration_options = {'version': 2,
                               'poni_version': 2,
                               'detector': 'Maxipix2x2',
                               'detector_config': {},
                               'dist': 0.029697341310368504,
                               'poni1': 0.03075830660872356,
                               'poni2': 0.008514191495847496,
                               'rot1': 0.2993182762142924,
                               'rot2': 0.11405876271088071,
                               'rot3': -4.950942273187188e-07,
                               'wavelength': 1.0332016449700598e-10,
                               'integrator_name' : "sigma_clip",
                               'extra_options' : {"max_iter": 3, "thres": 0} }
        worker = Worker()
        worker.set_config(integration_options)
        result = worker.get_config()
        self.assertEqual(result['extra_options'],  integration_options['extra_options'], "'extra_options' matches")
        self.assertEqual(result['integrator_method'],  integration_options['integrator_name'], "'integrator_name' matches")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestWorker))
    testsuite.addTest(loader(TestWorkerConfig))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
