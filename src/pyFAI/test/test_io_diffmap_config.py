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

"""Test suites for diffmap config serialization"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "11/04/2025"

import unittest
import numpy
import json
import os
from dataclasses import fields
import logging
logger = logging.getLogger(__name__)
from ..io.diffmap_config import DiffmapConfig, MotorRange, WorkerConfig, ListDataSet, DataSet, CURRENT_VERSION

test_data = """
{
  "ai": {
    "application": "pyfai-integrate",
    "version": 5,
    "poni": {
      "poni_version": 2.1,
      "dist": 2.826838431400575,
      "poni1": 0.19391282381518082,
      "poni2": 0.09713342904802076,
      "rot1": 0.0,
      "rot2": 0.0,
      "rot3": 0.0,
      "detector": "Pilatus2M",
      "detector_config": {
        "orientation": 3
      },
      "wavelength": 9.918735791712574e-11
    },
    "nbpt_rad": 1000,
    "nbpt_azim": null,
    "unit": "q_nm^-1",
    "chi_discontinuity_at_0": false,
    "polarization_description": null,
    "normalization_factor": 1.0,
    "val_dummy": null,
    "delta_dummy": null,
    "correct_solid_angle": true,
    "dark_current": null,
    "flat_field": null,
    "mask_file": "fabio:/tmp/bm29/mask_28thFeb25.npy",
    "error_model": "no",
    "method": [
      "full",
      "csr",
      "opencl"
    ],
    "opencl_device": [
      0,
      1
    ],
    "azimuth_range": null,
    "radial_range": null,
    "integrator_class": "AzimuthalIntegrator",
    "integrator_method": null,
    "extra_options": null,
    "monitor_name": null,
    "shape": [
      1679,
      1475
    ]
  },
  "experiment_title": "Diffraction Mapping",
  "fast_motor_name": "fast",
  "slow_motor_name": "slow",
  "nbpt_fast": 21,
  "nbpt_slow": 31,
  "offset": null,
  "zigzag_scan": true,
  "output_file": "scan17_b.h5",
  "input_data": [
    [
      "/tmp/bm29/scan0017/scan_channel_chip_00020000.h5",
      "",
      null
    ],
    [
      "/tmp/bm29/scan0017/scan_channel_chip_00020001.h5",
      "",
      null
    ],
    [
      "/tmp/bm29/scan0017/scan_channel_chip_00020002.h5",
      "",
      null
    ],
    [
      "/tmp/bm29/scan0017/scan_channel_chip_00020003.h5",
      "",
      null
    ],
    [
      "/tmp/bm29/scan0017/scan_channel_chip_00020004.h5",
      "",
      null
    ],
    [
      "/tmp/bm29/scan0017/scan_channel_chip_00020005.h5",
      "",
      null
    ],
    [
      "/tmp/bm29/scan0017/scan_channel_chip_00020006.h5",
      "",
      null
    ]
  ],
  "fast_motor_range": [
    0.0,
    1.0
  ],
  "slow_motor_range": [
    0.0,
    1.0
  ]
}
""" # data obtained with pyFAI 2025.

class TestDiffmapConfig(unittest.TestCase):
    """Test diffmap config
    """
    @classmethod
    def setUpClass(cls):
        cls.inp = json.loads(test_data)
    @classmethod
    def tearDownClass(cls):
        cls.inp = None

    def test_parse(self):
        parsed = DiffmapConfig.from_dict(self.inp)
        for field in fields(DiffmapConfig):
            value = parsed.__getattribute__(field.name)
            msg = f"{field.name} is type {type(value).__name__}, expected {field.type.__name__}"
            if value is None:
                logger.warning(msg)
            else:
                self.assertTrue(isinstance(value, field.type), msg)
        #just a few test to validate the parsing...
        self.assertEqual(parsed.output_file, self.inp["output_file"])
        self.assertEqual(parsed.diffmap_config_version, CURRENT_VERSION)
        self.assertEqual(parsed.experiment_title, self.inp["experiment_title"])
        self.assertEqual(parsed.offset, self.inp["offset"])
        self.assertEqual(parsed.zigzag_scan, self.inp["zigzag_scan"])

    def test_consistency(self):
        ref = DiffmapConfig.from_dict(self.inp)
        obt = DiffmapConfig.from_dict(ref.as_dict())
        for field in fields(DiffmapConfig):
            # logger.info("%s: %s %s", field.name, ref.__getattribute__(field.name), obt.__getattribute__(field.name))
            self.assertEqual(ref.__getattribute__(field.name), obt.__getattribute__(field.name),
                    f"{field.name}: {ref.__getattribute__(field.name)} ≠ {obt.__getattribute__(field.name)}")

class TestListDataSet(unittest.TestCase):
    """Test ListDataSet serialization
    """

    def test_empty(self):
        empty = ListDataSet.from_serialized([])
        self.assertEqual(empty.commonroot(), "")


    def test_single(self):

        single = ListDataSet([DataSet("/a/b/c")])
        self.assertEqual(single.commonroot(), os.path.normpath("/a/b/c"))

    def test_multiple(self):
        multi = ListDataSet.from_serialized([(os.path.normpath("/a/b/c"),None,None, (10,11)),
                                             (os.path.normpath("/a/b/d"), None, 2)])
        self.assertEqual(multi.commonroot(), os.path.normpath("/a/b")+os.sep)
        self.assertEqual(multi.nframes, 3)
        self.assertEqual(multi.shape, (10,11))


def suite():

    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestDiffmapConfig))
    testsuite.addTest(loader(TestListDataSet))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
