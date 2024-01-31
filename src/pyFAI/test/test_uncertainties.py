#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2024-2024 European Synchrotron Radiation Facility, Grenoble, France
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

"""test suite on uncertainty propagation

"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@esrf.fr"
__license__ = "MIT"
__copyright__ = "2024-2024 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "31/01/2024"

import sys
import os
import unittest
import numpy
import logging
logger = logging.getLogger(__name__)
from .utilstest import UtilsTest
import fabio
from .. import load


class TestUncertainties(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super(TestUncertainties, cls).setUpClass()
        cls.ai = load({"detector": "Pilatus 100k",
                       "dist": 0.1,
                       "poni1": 0.01,
                       "poni2": 0.01,
                       "wavelength": 1e-10,
                       })
        cls.img = UtilsTest.get_rng().poisson(1000, cls.ai.detector.shape)
        cls.npt = 500

    @classmethod
    def tearDownClass(cls) -> None:
        super(TestUncertainties, cls).tearDownClass()
        cls.ai = cls.img = cls.npt = None

    def _test(self, split="no", error_model="poisson",
              algos=("histogram", "csr", "csc", "lut"),
              check=("radial", "intensity", "std", "sem", "count", "sum_signal", "sum_variance", "sum_normalization", "sum_normalization2")):

        res = {}
        for m in algos:
            tmp = self.ai.integrate1d(self.img, self.npt, error_model=error_model, method=(split, m, "cython"))
            if not res:
                res[m] = tmp
                ref = tmp
                ref_name = m
            else:
                for what in check:
                    # print(m[:3], what, getattr(tmp, what), getattr(ref, what))
                    if isinstance(check, dict):
                        epsilon =  check.get(what)
                        self.assertTrue(numpy.allclose(getattr(ref,what), getattr(tmp,what), rtol=epsilon),
                                        f"{what} matches for {m} with {ref_name}, split: {split}, error_model: {error_model}, epsilon: {epsilon}")
                    else:
                        self.assertTrue(numpy.allclose(getattr(ref, what), getattr(tmp, what)),
                                        f"{what} matches for {m} with {ref_name}, split: {split}, error_model: {error_model}")

    def test_poisson_model(self):
        """ LUT used to gives different uncertainties
        Issue #2053 on Poisson error model

        """
        self._test(split="no", error_model="poisson", algos=("histogram", "csr", "csc", "lut"),
                   check=("radial", "intensity", "std", "sem", "count", "sum_signal", "sum_variance", "sum_normalization", "sum_normalization2"))

        self._test(split="bbox", error_model="poisson", algos=("histogram", "csr", "csc", "lut"),
                   check=("radial", "intensity", "std", "sem", "count", "sum_signal", "sum_variance", "sum_normalization", "sum_normalization2"))

        self._test(split="full", error_model="poisson", algos=("histogram", "csr", "csc", "lut"),
                   check=("radial", "intensity", "std", "sem", "count", "sum_signal", "sum_variance", "sum_normalization", "sum_normalization2"))


    def test_azimuthal_model(self):
        """ histogram and csc are not producing uncertainties ...
        Issue #2061 on azimuthal error model

        """

        self._test(split="no", error_model="azimuthal", algos=("csr", "lut", "csc", "histogram"),
                   check=("radial", "intensity", "std", "sem", "count", "sum_signal", "sum_variance", "sum_normalization", "sum_normalization2"))

        self._test(split="bbox", error_model="azimuthal", algos=("csr", "lut", "csc", "histogram"),
                   check=("radial", "intensity", "std", "sem", "count", "sum_signal", "sum_variance", "sum_normalization", "sum_normalization2"))

        self._test(split="full", error_model="azimuthal", algos=("csr", "lut", "csc", "histogram"),
                   check=("radial", "intensity", "std", "sem", "count", "sum_signal", "sum_variance", "sum_normalization", "sum_normalization2"))



def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestUncertainties))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
