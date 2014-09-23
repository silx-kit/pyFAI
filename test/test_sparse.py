#!/usr/bin/env python

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
