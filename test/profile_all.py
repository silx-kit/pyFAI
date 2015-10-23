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
__doc__ = """Test suite for all pyFAI modules with timing and memory profiling"""
__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "22/10/2015"


import sys
import os
import unittest
import time
if __name__ == '__main__':
    import pkgutil
    __path__ = pkgutil.extend_path([os.path.dirname(__file__)], "pyFAI.test")
from .utilstest import UtilsTest, getLogger

from . import test_all

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
    suite = test_all.suite()
    runner = ProfileTestRunner()
    testresult = runner.run(suite)
    if testresult.wasSuccessful():
        UtilsTest.clean_up()
    else:
        sys.exit(1)
