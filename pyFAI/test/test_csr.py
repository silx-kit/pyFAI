#!/usr/bin/python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) 2015 European Synchrotron Radiation Facility, Grenoble, France
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

from __future__ import absolute_import, division, print_function

__doc__ = """tests for Jon's geometry changes
FIXME : make some tests that the functions do what is expected
"""


import unittest
import numpy
import os
import sys
import time
from .utilstest import UtilsTest, getLogger, diff_img, diff_crv
logger = getLogger(__file__)
from .. import opencl
from ..ext import splitBBox
from ..ext import splitBBoxCSR
from ..azimuthalIntegrator import AzimuthalIntegrator
if opencl.ocl:
    from .. import ocl_azim_csr

import fabio

N = 1000
ai = AzimuthalIntegrator.sload(UtilsTest.getimage("1893/Pilatus1M.poni"))
data = fabio.open(UtilsTest.getimage("1883/Pilatus1M.edf")).data
ai.xrpd_LUT(data, N)


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


class ParamOpenclCSR(ParameterisedTestCase):

    def test_csr(self):
        workgroup_size = self.param
        out_ref = splitBBox.histoBBox1d(data, ai.ttha, ai._cached_array["2th_delta"], bins=N)
        csr = splitBBoxCSR.HistoBBox1d(ai.ttha, ai._cached_array["2th_delta"], bins=N, unit="2th_deg")
        if not opencl.ocl:
            skip = True
        else:
            try:
                ocl_csr = ocl_azim_csr.OCL_CSR_Integrator(csr.lut, data.size, "ALL", profile=True, block_size=workgroup_size)
                out_ocl_csr = ocl_csr.integrate(data)
            except (opencl.pyopencl.MemoryError, MemoryError):
                logger.warning("Skipping test due to memory error on device")
                skip = True
            else:
                skip = False
        out_cyt_csr = csr.integrate(data)
        cmt = "Testing ocl_csr with workgroup_size= %s" % (workgroup_size)
        logger.debug(cmt)
        if skip:
            for ref, cyth in zip(out_ref, out_cyt_csr):
                self.assertTrue(numpy.allclose(ref, cyth), cmt + ": hist vs csr")
        else:
            for ref, ocl, cyth in zip(out_ref[1:], out_ocl_csr, out_cyt_csr[1:]):
                self.assertTrue(numpy.allclose(ref, ocl), cmt + ": hist vs ocl_csr")
                self.assertTrue(numpy.allclose(ref, cyth), cmt + ": hist vs csr")
                self.assertTrue(numpy.allclose(cyth, ocl), cmt + ": csr vs ocl_csr")
        csr = None
        ocl_csr = None
        out_ocl_csr = None
        out_ref = None

TESTCASES = [8 * 2 ** i for i in range(6)]  # [8, 16, 32, 64, 128, 256]


class Test_CSR(unittest.TestCase):
    def test_2d_splitbbox(self):
        ai.reset()
        img, tth, chi = ai.integrate2d(data, N, unit="2th_deg", method="splitbbox_LUT")
        img_csr, tth_csr, chi_csr = ai.integrate2d(data, N, unit="2th_deg", method="splitbbox_csr")
        self.assertTrue(numpy.allclose(tth, tth_csr), " 2Th are the same")
        self.assertTrue(numpy.allclose(chi, chi_csr), " Chi are the same")
        # TODO: align on splitbbox rather then splitbbox_csr
        diff_img(img, img_csr, "splitbbox")
        self.assertTrue(numpy.allclose(img, img_csr), " img are the same")

    def test_2d_nosplit(self):
        ai.reset()
        img, tth, chi = ai.integrate2d(data, N, unit="2th_deg", method="histogram")
        img_csr, tth_csr, chi_csr = ai.integrate2d(data, N, unit="2th_deg", method="nosplit_csr")
#        diff_crv(tth, tth_csr, "2th")
#        self.assertTrue(numpy.allclose(tth, tth_csr), " 2Th are the same")
#        self.assertTrue(numpy.allclose(chi, chi_csr), " Chi are the same")
        diff_img(img, img_csr, "no split")
        self.assertTrue(numpy.allclose(img, img_csr), " img are the same")


def suite():
    testsuite = unittest.TestSuite()
    if opencl.ocl:
        for param in TESTCASES:
            testsuite.addTest(ParameterisedTestCase.parameterise(
                    ParamOpenclCSR, param))
    # if no opencl: no test
#    testsuite.addTest(Test_CSR("test_2d_splitbbox"))
#    testsuite.addTest(Test_CSR("test_2d_nosplit"))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
