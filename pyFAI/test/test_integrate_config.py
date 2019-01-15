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

"Test suite for worker"

from __future__ import absolute_import, division, print_function

__author__ = "Valentin Valls"
__contact__ = "valentin.valls@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "07/01/2019"


import unittest
import logging
from ..io import integration_config
from . import utilstest


logger = logging.getLogger(__name__)


class TestIntegrationConfigV1(unittest.TestCase):

    def test_poni(self):
        config = {"poni": utilstest.UtilsTest.getimage("Pilatus1M.poni"),
                  "wavelength": 50}
        config = integration_config.normalize(config)
        self.assertNotIn("poni", config)
        # From PONI file
        self.assertEqual(config["dist"], 1.58323111834)
        # From dict
        self.assertEqual(config["wavelength"], 50)

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
        self.assertEqual(config["detector"], "Pilatus2M")
        self.assertIsInstance(config["detector_config"], dict)

    def test_detector(self):
        config = {"detector": "detector",
                  "pixel1": 1.0,
                  "pixel2": 1.0}
        config = integration_config.normalize(config)
        self.assertNotIn("pixel1", config)
        self.assertNotIn("pixel2", config)
        self.assertEqual(config["detector"], "Detector")
        self.assertEqual(config["detector_config"]["pixel1"], 1.0)
        self.assertEqual(config["detector_config"]["pixel2"], 1.0)

    def test_frelon(self):
        spline_file = utilstest.UtilsTest.getimage("frelon.spline")
        config = {"detector": "frelon",
                  "splineFile": spline_file}
        config = integration_config.normalize(config)
        self.assertNotIn("splineFile", config)
        self.assertEqual(config["detector"], "FReLoN")
        self.assertEqual(config["detector_config"]["splineFile"], spline_file)

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


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestIntegrationConfigV1))
    testsuite.addTest(loader(TestIntegrationConfigV2))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
