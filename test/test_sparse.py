#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal Integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#


"""
Test suites for sparse matrix multiplication modules 
"""


import unittest, numpy, os, sys, time
from utilstest import UtilsTest, getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]

from pyFAI import splitBBox
from pyFAI import splitBBoxCSR
from pyFAI import splitBBoxLUT
import fabio


class TestSparseBBox(unittest.TestCase):
    """Test Azimuthal integration based sparse matrix mutiplication methods
    Bounding box pixel splitting
    """
    N = 1000
    unit = "2th_deg"
    ai = pyFAI.load(UtilsTest.getimage("1893/Pilatus1M.poni"))
    data = fabio.open(UtilsTest.getimage("1883/Pilatus1M.edf")).data
    ref = ai.integrate1d(data, N, correctSolidAngle=False, unit=unit, method="splitBBox")[1]
    cython = splitBBox.histoBBox1d(data, ai._ttha, ai._dttha, bins=N)

    def test_LUT(self):
        obt = self.ai.integrate1d(self.data, self.N, correctSolidAngle=False, unit=self.unit, method="LUT")[1]
        logger.debug("delta on global result: %s" % (abs(obt - self.ref) / self.ref).max())
        self.assert_(numpy.allclose(obt, self.ref))

        cython = self.ai._lut_integrator.integrate(self.data)
        for ref, obt in zip(self.cython, cython):
            logger.debug("delta on cython result: %s" % (abs(obt - ref) / ref).max())
            self.assert_(numpy.allclose(obt, ref))

    def test_CSR(self):
        obt = self.ai.integrate1d(self.data, self.N, correctSolidAngle=False, unit=self.unit, method="CSR")[1]
        logger.debug("delta on global result: %s" % (abs(obt - self.ref) / self.ref).max())
        self.assert_(numpy.allclose(obt, self.ref))

        cython = self.ai._csr_integrator.integrate(self.data)
        for ref, obt in zip(self.cython, cython):
            logger.debug("delta on cython result: %s" % (abs(obt - ref) / ref).max())
            self.assert_(numpy.allclose(obt, ref))


def test_suite_all_sparse():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestSparseBBox("test_LUT"))
    testSuite.addTest(TestSparseBBox("test_CSR"))
    return testSuite


if __name__ == '__main__':
    mysuite = test_suite_all_sparse()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
