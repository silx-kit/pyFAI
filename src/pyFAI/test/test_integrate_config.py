#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2025 European Synchrotron Radiation Facility, Grenoble, France
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
__date__ = "28/04/2025"

import os
import json
import unittest
import logging
from ..io import integration_config, ponifile

from . import utilstest

logger = logging.getLogger(__name__)


class TestIntegrationConfigV1(unittest.TestCase):

    def test_poni(self):
        config = {"poni": utilstest.UtilsTest.getimage("Pilatus1M.poni"),
                  "wavelength": 50}
        config = integration_config.normalize(config)
        # self.assertNotIn("poni", config)
        # From PONI file
        self.assertEqual(config["poni"]["dist"], 1.58323111834)
        # From dict
        self.assertEqual(config["poni"]["wavelength"], 50)

    def test_single_flatdark(self):
        config = {"dark_current": "abc",
                  "flat_field": "abd"}
        config = integration_config.normalize(config)
        self.assertEqual(config["dark_current"], "abc")
        self.assertEqual(config["flat_field"], "abd")

    def test_coma_flatdark(self):
        config = {"dark_current": "a,b,c",
                  "flat_field": "a,b,d"}
        config = integration_config.normalize(config)
        self.assertEqual(config["dark_current"], ["a", "b", "c"])
        self.assertEqual(config["flat_field"], ["a", "b", "d"])

    def test_pilatus(self):
        config = {"detector": "pilatus2m"}
        config = integration_config.normalize(config)
        self.assertEqual(config["poni"]["detector"], "Pilatus2M")
        self.assertIsInstance(config["poni"]["detector_config"], dict)

    def test_detector(self):
        config = {"detector": "detector",
                  "pixel1": 1.0,
                  "pixel2": 1.0}
        config = integration_config.normalize(config)
        self.assertNotIn("pixel1", config)
        self.assertNotIn("pixel2", config)
        self.assertEqual(config["poni"]["detector"], "Detector")
        self.assertEqual(config["poni"]["detector_config"]["pixel1"], 1.0)
        self.assertEqual(config["poni"]["detector_config"]["pixel2"], 1.0)

    def test_frelon(self):
        spline_file = utilstest.UtilsTest.getimage("frelon.spline")
        config = {"detector": "frelon",
                  "splineFile": spline_file}
        config = integration_config.normalize(config)
        self.assertNotIn("splineFile", config)
        self.assertEqual(config["poni"]["detector"], "FReLoN")
        self.assertEqual(config["poni"]["detector_config"]["splineFile"], spline_file)

    def test_opencl(self):
        config = {"do_OpenCL": True}
        config = integration_config.normalize(config)
        self.assertNotIn("do_OpenCL", config)
        self.assertEqual(config["method"], ('*', 'csr', 'opencl'))


class TestIntegrationConfigV2(unittest.TestCase):

    def test_opencl_device(self):
        config = {
            "version": 2,
            "application": "pyfai-integrate",
            "method": "csrocl_1,1"}
        config = integration_config.normalize(config)
        self.assertNotIn("do_OpenCL", config)
        self.assertEqual(config["method"], ('*', 'csr', 'opencl'))
        self.assertEqual(config["opencl_device"], (1, 1))

    def test_opencl_cpu_device(self):
        config = {
            "version": 2,
            "application": "pyfai-integrate",
            "method": "lutocl_cpu"}
        config = integration_config.normalize(config)
        self.assertNotIn("do_OpenCL", config)
        self.assertEqual(config["method"], ('*', 'lut', 'opencl'))
        self.assertEqual(config["opencl_device"], "cpu")


class TestRegression(unittest.TestCase):

    def test_2132(self):
        """issue #2132:
        when parsing a config json file in diffmap + enforce the usage of the GPU, the splitting gets changed
        """
        from ..diffmap import DiffMap
        from ..opencl import ocl

        expected_without_gpu = ("no", "lut", "cython")
        expected_with_gpu = ("no", "lut", "opencl")
        config = {"ai": {"method": expected_without_gpu}}
        config_file = os.path.join(utilstest.UtilsTest.tempdir, "test_2132.json")
        with open(config_file, "w") as fp:
            json.dump(config, fp)

        # without GPU option -g
        dm = DiffMap(1, 1)
        _, parsed_config = dm.parse(sysargv=["--config", config_file], with_config=dict)
        self.assertEqual(parsed_config["ai"]["method"], expected_without_gpu, "method matches without -g option")

        # with GPU option -g
        dm = DiffMap(1, 1)
        _, parsed_config = dm.parse(sysargv=["-g", "--config", config_file], with_config=dict)
        expected = expected_with_gpu if ocl else expected_without_gpu
        self.assertEqual(parsed_config["ai"]["method"], expected, "method match with -g option")

    def test_dataclass(self):
        test_files = "0.14_verson0.json  id11_v0.json  id13_v0.json  id15_1_v0.json  id15_v0.json  id16_v3.json  id21_v0.json  version0.json    version3.json  version4.json"
        for fn in test_files.split():
            js = utilstest.UtilsTest.getimage(fn)
            print(fn)
            with utilstest.TestLogging(logger='pyFAI.io.integrarion_config', warning=0):
                wc = integration_config.WorkerConfig.from_file(js)
            wc.poni.API_VERSION = ponifile.PoniFile.API_VERSION
            self.assertEqual(str(wc), str(integration_config.WorkerConfig.from_dict(wc.as_dict())), f"Idempotent {fn}")

    def test_nested_dataclasses(self):
        from ..containers import PolarizationDescription, ErrorModel
        from ..units import to_unit, Unit
        # Polarization
        w = integration_config.WorkerConfig.from_dict({})
        w.polarization_factor = 1
        w.polarization_offset = 0.5
        self.assertEqual(w.polarization_description, (1, 0.5), "Polarization description")
        self.assertTrue(isinstance(w.polarization_description, PolarizationDescription))

        w = integration_config.WorkerConfig.from_dict({"polarization_description":(-1, -0.5)})
        self.assertEqual(w.polarization_factor, -1, "polarization factor")
        self.assertEqual(w.polarization_offset, -0.5, "polarization offset")
        self.assertTrue(isinstance(w.polarization_description, PolarizationDescription))

        # units:
        w = integration_config.WorkerConfig.from_dict({})
        w.unit = to_unit("2th_deg")
        self.assertEqual(str(w.unit), "2th_deg", "units")
        self.assertTrue(isinstance(w.unit, Unit))
        self.assertEqual(w.as_dict()["unit"], "2th_deg", "units")

        w = integration_config.WorkerConfig.from_dict({"unit":"q_A^-1"})
        self.assertTrue(isinstance(w.unit, Unit))
        self.assertEqual(str(w.unit), "q_A^-1", "units")
        self.assertEqual(w.as_dict()["unit"], "q_A^-1", "units")

        # Error Model
        w = integration_config.WorkerConfig.from_dict({})
        w.error_model = ErrorModel(3)
        self.assertEqual(w.error_model.as_str(), "azimuthal", "Error Model")
        self.assertEqual(w.as_dict()["error_model"], "azimuthal", "Error Model")
        w = integration_config.WorkerConfig.from_dict({"error_model": "poisson"})
        self.assertTrue(isinstance(w.error_model, ErrorModel))
        self.assertEqual(w.error_model.as_str(), "poisson", "Error Model")
        self.assertEqual(w.as_dict()["error_model"], "poisson", "Error Model")

        # PoniFile
        w = integration_config.WorkerConfig.from_dict({})
        w.poni = ponifile.PoniFile({"detector": "Titan"})
        self.assertEqual(w.poni.as_dict()["detector"], "Titan", "PoniFile")
        self.assertEqual(w.as_dict()["poni"]["detector"], "Titan", "PoniFile")
        w = integration_config.WorkerConfig.from_dict({"poni": {"detector": "Frelon"}})
        self.assertTrue(isinstance(w.poni, ponifile.PoniFile))
        self.assertEqual(w.poni.as_dict()["detector"], "FReLoN", "PoniFile")
        self.assertEqual(w.as_dict()["poni"]["detector"], "FReLoN", "PoniFile")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestIntegrationConfigV1))
    testsuite.addTest(loader(TestIntegrationConfigV2))
    testsuite.addTest(loader(TestRegression))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
