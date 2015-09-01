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
from __future__ import absolute_import, division, print_function
"""
Test suite for all pyFAI modules with timing and memory profiling
"""

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "01/09/2015"


import sys
import os
import unittest
import time
if __name__ == '__main__':
    import pkgutil
    __path__ = pkgutil.extend_path([os.path.dirname(__file__)], "pyFAI.test")
from .utilstest import UtilsTest, getLogger

from .test_all import test_suite_all

import resource
import logging
profiler = logging.getLogger("memProf")
profiler.setLevel(logging.DEBUG)
profiler.handlers.append(logging.FileHandler("profile.log"))


class TestResult(unittest.TestResult):

    def startTest(self, test):
        self.__mem_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.__time_start = time.time()
        unittest.TestResult.startTest(self, test)

    def stopTest(self, test):
        unittest.TestResult.stopTest(self, test)
        profiler.info("Time: %.3fs \t RAM: %.3f Mb\t%s" % (time.time() - self.__time_start,
                                                          (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - self.__mem_start) / 1e3,
                                                          test.id()))

class ProfileTestRunner(unittest.TextTestRunner):
    def _makeResult(self):
        return TestResult(stream=sys.stderr, descriptions=True, verbosity=1)

if __name__ == '__main__':
    mysuite = test_suite_all()
    runner = ProfileTestRunner()
    testresult = runner.run(mysuite)
    if runner.run(mysuite).wasSuccessful():
        UtilsTest.clean_up()
    else:
        sys.exit(1)
