#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2023 European Synchrotron Radiation Facility, Grenoble, France
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

"""
tests units, used to generate geometries
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "30/04/2024"

import unittest
import numpy
import logging
from .utilstest import UtilsTest
from .. import load
from .. import units
logger = logging.getLogger(__name__)


class TestUnits(unittest.TestCase):
    def test_corner(self):
        ai = load({"detector":"Imxpad S10"})
        res = ai.array_from_unit(typ="corner", unit=("chi_rad"))
        #no fast path, just checks numexpr gives correct values.
        self.assertTrue(numpy.allclose(res[...,0], res[...,1]), "numexpr formula OK")

    def test_all(self):
        rng = UtilsTest.get_rng()
        shape = (9, 11)
        x = rng.uniform(-2, 2, shape)
        y = rng.uniform(-2, 2, shape)
        z = rng.uniform(0.1, 2, shape)
        λ = rng.uniform(1e-11, 1e-9, shape)
        for k,u in units.ANY_UNITS.items():
            if callable(u._equation) and callable(u.equation) and (u._equation!=u.equation):
                ref = u.equation(x,y,z,λ)
                obt = u._equation(x,y,z,λ)
                self.assertTrue(numpy.allclose(ref,obt), f"Equation and formula do NOT match for {k}")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestUnits))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
