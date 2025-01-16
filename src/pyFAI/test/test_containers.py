#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2025-2025 European Synchrotron Radiation Facility, Grenoble, France
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

"""Test suite for container module"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jérôme.Kieffer@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "07/01/2025"

import unittest
import numpy
import logging
logger = logging.getLogger(__name__)
import fabio
from .utilstest import UtilsTest
from .. import load as pyFAI_load
from .. import containers


class TestContainer(unittest.TestCase):

    def test_rebin1d(self):
        img = fabio.open(UtilsTest.getimage("moke.tif")).data
        ai = pyFAI_load({
    "poni_version": 2.1,
    "detector": "Detector",
    "detector_config": {
      "pixel1": 1e-4,
      "pixel2": 1e-4,
      "max_shape": [
        500,
        600
      ],
      "orientation": 3
    },
    "dist": 0.1,
    "poni1": 0.03,
    "poni2": 0.03,
    "rot1": 0.0,
    "rot2": 0.0,
    "rot3": 0.0,
    "wavelength": 1.0178021533473887e-10
  })
        method = ("no", "histogram", "cython")
        res2d = ai.integrate2d(img, 500, 360, method=method, error_model="poisson")
        ref1d = ai.integrate1d(img, 500, method=method, error_model="poisson")
        res1d = containers.rebin1d(res2d)
        self.assertTrue(numpy.allclose(res1d[0], ref1d[0]), "radial matches")
        self.assertTrue(numpy.allclose(res1d[1], ref1d[1]), "intensity matches")
        self.assertTrue(numpy.allclose(res1d[2], ref1d[2]), "sem matches")



def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestContainer))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
