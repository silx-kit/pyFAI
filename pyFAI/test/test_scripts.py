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
__date__ = "19/07/2017"

import sys
import unittest
import runpy

from .utilstest import getLogger
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

    def executeAppHelp(self, module):
        old_sys_argv = list(sys.argv)
        try:
            sys.argv = [None, "--help"]
            runpy.run_module(mod_name=module, run_name="__main__", alter_sys=True)
        except SystemExit as e:
            self.assertEquals(e.args[0], 0)
        finally:
            sys.argv = old_sys_argv

    def testCheckCalib(self):
        if qt is None:
            self.skipTest("Library Qt is not available")
        self.executeAppHelp("pyFAI.app.check_calib")

    def testDetector2Nexus(self):
        self.executeAppHelp("pyFAI.app.detector2nexus")

    def testDiffMap(self):
        self.executeAppHelp("pyFAI.app.diff_map")

    def testDiffTomo(self):
        self.executeAppHelp("pyFAI.app.diff_tomo")

    def testEigerMask(self):
        self.executeAppHelp("pyFAI.app.eiger_mask")

    def testMxcalibrate(self):
        if qt is None:
            self.skipTest("Library Qt is not available")
        self.executeAppHelp("pyFAI.app.mx_calibrate")

    def testPyfaiAverage(self):
        self.executeAppHelp("pyFAI.app.average")

    def testPyfaiBenchmark(self):
        self.executeAppHelp("pyFAI.app.benchmark")

    def testPyfaiCalib(self):
        if qt is None:
            self.skipTest("Library Qt is not available")
        self.executeAppHelp("pyFAI.app.calib")

    def testPyfaiDrawmask(self):
        if qt is None or (PyMca is None and silx is None):
            self.skipTest("Library Qt, PyMca and silx are not available")
        self.executeAppHelp("pyFAI.app.drawmask")

    def testPyfaiIntegrate(self):
        self.executeAppHelp("pyFAI.app.integrate")

    def testPyfaiRecalib(self):
        if qt is None:
            self.skipTest("Library Qt is not available")
        self.executeAppHelp("pyFAI.app.recalib")

    def testPyfaiSaxs(self):
        self.executeAppHelp("pyFAI.app.saxs")

    def testPyfaiWaxs(self):
        self.executeAppHelp("pyFAI.app.waxs")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestScriptsHelp))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
