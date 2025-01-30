#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2018 European Synchrotron Radiation Facility, Grenoble, France
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

"""Test suite to scripts"""

__author__ = "Valentin Valls"
__contact__ = "valentin.valls@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "28/09/2024"

import sys
import unittest
import logging
import subprocess
from ..test.utilstest import test_options

_logger = logging.getLogger(__name__)


class TestScriptsHelp(unittest.TestCase):

    def executeCommandLine(self, command_line, env):
        """Execute a command line.

        Log output as debug in case of bad return code.
        """
        _logger.info("Execute: %s", " ".join(command_line))
        p = subprocess.Popen(command_line,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             env=env)
        out, err = p.communicate()
        _logger.info("Return code: %d", p.returncode)
        try:
            out = out.decode('utf-8')
        except UnicodeError:
            pass
        try:
            err = err.decode('utf-8')
        except UnicodeError:
            pass

        if p.returncode != 0:
            _logger.warning("stdout:")
            _logger.warning("%s", out)
            _logger.warning("stderr:")
            _logger.warning("%s", err)
        else:
            _logger.debug("stdout:")
            _logger.debug("%s", out)
            _logger.debug("stderr:")
            _logger.debug("%s", err)
        self.assertEqual(p.returncode, 0)

    def executeAppHelp(self, script_name, module_name, function="main", help="--help"):
        script = f"import {module_name}; {module_name}.{function}(['{help}'])"
        env = test_options.get_test_env()
        _logger.info(script)
        command_line = [sys.executable, "-c", script]
        self.executeCommandLine(command_line, env)

    def testDetector2Nexus(self):
        self.executeAppHelp("detector2nexus", "pyFAI.app.detector2nexus")

    def testDiffMap(self):
        self.executeAppHelp("diff_map", "pyFAI.app.diff_map")

    def testDiffTomo(self):
        self.executeAppHelp("diff_tomo", "pyFAI.app.diff_tomo")

    def testEigerMask(self):
        self.executeAppHelp("eiger-mask", "pyFAI.app.eiger_mask")

    def testPyfaiAverage(self):
        self.executeAppHelp("pyFAI-average", "pyFAI.app.average")

    @unittest.skipIf(not test_options.gui, "no GUI")
    def testPyfaiBenchmark(self):
        self.executeAppHelp("pyFAI-benchmark", "pyFAI.app.benchmark")

    @unittest.skipIf(not test_options.gui, "no GUI")
    def testPyfaiIntegrate(self):
        self.executeAppHelp("pyFAI-integrate", "pyFAI.app.integrate")

    def testPeakfinder(self):
        self.executeAppHelp("peakfinder", "pyFAI.app.peakfinder")

    @unittest.skipIf(not test_options.gui, "no GUI")
    def testPilx(self):
        self.executeAppHelp("pyFAI-diffmap-view", "pyFAI.app.pilx")

    def testPyfaiSaxs(self):
        self.executeAppHelp("pyFAI-saxs", "pyFAI.app.saxs")

    def testSparsify(self):
        self.executeAppHelp("sparsify-Bragg", "pyFAI.app.sparsify")

    def testPyfaiWaxs(self):
        self.executeAppHelp("pyFAI-waxs", "pyFAI.app.waxs")

    @unittest.skipIf(not test_options.gui, "no GUI")
    def testCheckCalib(self):
        self.executeAppHelp("check_calib", "pyFAI.app.check_calib")

    @unittest.skipIf(not test_options.gui, "no GUI")
    def testMxcalibrate(self):
        self.executeAppHelp("MX-calibrate", "pyFAI.app.mx_calibrate")

    @unittest.skipIf(not test_options.gui, "no GUI")
    def testPyfaiCalib(self):
        self.executeAppHelp("pyFAI-calib", "pyFAI.app.calib")

    @unittest.skipIf(not test_options.gui, "no GUI")
    def testPyfaiDrawmask(self):
        self.executeAppHelp("pyFAI-drawmask", "pyFAI.app.drawmask")

    @unittest.skipIf(not test_options.gui, "no GUI")
    def testPyfaiRecalib(self):
        self.executeAppHelp("pyFAI-recalib", "pyFAI.app.recalib")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestScriptsHelp))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
