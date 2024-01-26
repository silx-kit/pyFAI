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
__date__ = "26/01/2024"

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

    def test_poisson_model(self):
        """ LUT gives different uncertainties in issue #2053
        
        """
        ai = load(UtilsTest.getimage("Pilatus1M.poni"))
        img = fabio.open(UtilsTest.getimage("Pilatus1M.edf")).data
        res = {}
        for m in ("histogram", "csr", "csc", "lut"):
            res[m] = ai.integrate1d(img, 100, error_model="poisson", method=("no", m, "cython"))
            if m == "histogram":
                ref = res[m].sigma
            else:
                self.assertTrue(numpy.allclose(ref, res[m].sigma), f"sigma matches for {m}")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestUncertainties))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
