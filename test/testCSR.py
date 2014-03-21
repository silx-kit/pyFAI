#!/usr/bin/env python

"""tests for Jon's geometry changes
FIXME : make some tests that the functions do what is expected

"""


import unittest, numpy, os, sys, time
from utilstest import UtilsTest, getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]

from pyFAI import splitBBox
from pyFAI import splitBBoxCSR
from pyFAI import ocl_azim_csr
import fabio



ai = pyFAI.load(UtilsTest.getimage("1893/Pilatus1M.poni"))
data = fabio.open(UtilsTest.getimage("1883/Pilatus1M.edf")).data
ai.xrpd_LUT(data, 1000)



class ParameterisedTestCase(unittest.TestCase):
    """ TestCase classes that want to be parameterised should
        inherit from this class.
        From Eli Bendersky's website
        http://eli.thegreenplace.net/2011/08/02/python-unit-testing-parametrized-test-cases/
    """
    def __init__(self, methodName='runTest', param=None):
        super(ParameterisedTestCase, self).__init__(methodName)
        self.param = param

    @staticmethod
    def parameterise(testcase_klass, param=None):
        """ Create a suite containing all tests taken from the given
            subclass, passing them the parameter 'param'.
        """
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(testcase_klass)
        suite = unittest.TestSuite()
        for name in testnames:
            suite.addTest(testcase_klass(name, param=param))
        return suite

class TestCSR(ParameterisedTestCase):
        
    def testCSR(self):
        workgroup_size, padded = self.param        

        out_ref = pyFAI.splitBBox.histoBBox1d(data, ai._ttha, ai._dttha, bins=1000)[1]
        
        if padded:
            csr = pyFAI.splitBBoxCSR.HistoBBox1d(ai._ttha, ai._dttha, bins=1000, unit="2th_deg", padding=workgroup_size)
        else:
            csr = pyFAI.splitBBoxCSR.HistoBBox1d(ai._ttha, ai._dttha, bins=1000, unit="2th_deg")

        ocl_csr = ocl_azim_csr.OCL_CSR_Integrator(csr.lut, data.size, "GPU",profile=True, padded=padded, block_size=workgroup_size)
        out_ocl_csr = ocl_csr.integrate(data)[0]
        print "Testing olc_csr with workgroup_size=", workgroup_size, " and padded=", padded
        self.assertTrue(numpy.allclose(out_ref, out_ocl_csr),"Test Failed at workgroup_size=" + str(workgroup_size)+" and padded="+str(padded))
        csr=None
        ocl_csr=None
        out_ocl_csr=None
        out_ref=None

TESTCASES = [
 (16, False),
 (16, True),
 (32, False),
 (32, True),
 (64, False),
 (64, True),
 (128, False),
 (128, True),
 (256, False),
 (256, True)
 ]


def test_suite_all_CSR():
    testSuite = unittest.TestSuite()
    for param in TESTCASES:
        testSuite.addTest(ParameterisedTestCase.parameterise(
                TestCSR, param))

    return testSuite



if __name__ == '__main__':
    mysuite = test_suite_all_CSR()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
