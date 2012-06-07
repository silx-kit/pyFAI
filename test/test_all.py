#!/usr/bin/env python
# coding: utf8
#
#    Project: pyFAI tests class utilities
#             http://forge.epn-campus.eu/projects/azimuthal
#
#    File: "$Id:$"
#
#    Copyright (C) 2010 European Synchrotron Radiation Facility
#                       Grenoble, France
#
#    Principal authors: Jérôme KIEFFER (jerome.kieffer@esrf.fr)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as
#    published by the Free Software Foundation, either version 3 of the
#    License, or (at your option) any later version.
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
Test suite for all pyFAI modules.
"""

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__data__ = "2011-07-06"

import unittest
import os
import logging
import sys

force_build = False

for opts in sys.argv[:]:
    if opts in ["-d", "--debug"]:
        logging.basicConfig(level=logging.DEBUG)
        sys.argv.pop(sys.argv.index(opts))
    elif opts in ["-i", "--info"]:
        logging.basicConfig(level=logging.INFO)
        sys.argv.pop(sys.argv.index(opts))
    elif opts in ["-f", "--force"]:
        force_build = True
        sys.argv.pop(sys.argv.index(opts))

try:
    logging.debug("tests loaded from file: %s" % __file__)
except:
    __file__ = os.getcwd()

from utilstest import UtilsTest

if force_build:
    UtilsTest.forceBuild()

from testGeometryRefinement   import test_suite_all_GeometryRefinement
from testAzimuthalIntegrator  import test_suite_all_AzimuthalIntegration
from testHistogram            import test_suite_all_Histogram
from testPeakPicking          import test_suite_all_PeakPicking

def test_suite_all():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_suite_all_Histogram())
    testSuite.addTest(test_suite_all_GeometryRefinement())
    testSuite.addTest(test_suite_all_AzimuthalIntegration())
    testSuite.addTest(test_suite_all_PeakPicking())
    return testSuite

if __name__ == '__main__':

    mysuite = test_suite_all()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)

