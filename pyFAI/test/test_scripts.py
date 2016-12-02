#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
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

__doc__ = """test suite to scripts
"""
__author__ = "Valentin Valls"
__contact__ = "valentin.valls@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "28/11/2016"

import sys
import os
import unittest
import subprocess

from .utilstest import getLogger, UtilsTest  # , Rwp, getLogger
logger = getLogger(__file__)

try:
    from ..gui import qt
except:
    qt = None

try:
    import PyMca
except:
    PyMca = None

try:
    import silx
except:
    silx = None


class TestScriptsHelp(unittest.TestCase):

    def executeScipt(self, scriptName):
        scriptPath, env = UtilsTest.script_path(scriptName)
        p = subprocess.Popen(
            [sys.executable, scriptPath, "--help"],
            shell=False,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)

        out, err = p.communicate()
        if p.returncode != 0:
            logger.error("Error while requesting help")
            logger.error("Stdout:")
            logger.error(out)
            logger.error("Stderr:")
            logger.error(err)
            envString = "Environment:"
            for k, v in self.env.items():
                env += "%s    %s: %s" % (os.linesep, k, v)
            logger.error(envString)
            self.fail()

    def testCheckCalib(self):
        if qt is None:
            self.skipTest("Library Qt is not available")
        self.executeScipt("check_calib")

    def testDetector2Nexus(self):
        self.executeScipt("detector2nexus")

    def testDiffMap(self):
        self.executeScipt("diff_map")

    def testDiffTomo(self):
        self.executeScipt("diff_tomo")

    def testEigerMask(self):
        self.executeScipt("eiger-mask")

    def testMxcalibrate(self):
        if qt is None:
            self.skipTest("Library Qt is not available")
        self.executeScipt("MX-calibrate")

    def testPyfaiAverage(self):
        self.executeScipt("pyFAI-average")

    def testPyfaiBenchmark(self):
        self.executeScipt("pyFAI-benchmark")

    def testPyfaiCalib(self):
        if qt is None:
            self.skipTest("Library Qt is not available")
        self.executeScipt("pyFAI-calib")

    def testPyfaiDrawmask(self):
        if qt is None or (PyMca is None and silx is None):
            self.skipTest("Library Qt, PyMca and silx are not available")
        self.executeScipt("pyFAI-drawmask")

    def testPyfaiIntegrate(self):
        self.executeScipt("pyFAI-integrate")

    def testPyfaiRecalib(self):
        if qt is None:
            self.skipTest("Library Qt is not available")
        self.executeScipt("pyFAI-recalib")

    def testPyfaiSaxs(self):
        self.executeScipt("pyFAI-saxs")

    def testPyfaiWaxs(self):
        self.executeScipt("pyFAI-waxs")


def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestScriptsHelp))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
