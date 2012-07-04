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

from utilstest import UtilsTest, getLogger
logger = getLogger(__file__)

from testGeometryRefinement   import test_suite_all_GeometryRefinement
from testAzimuthalIntegrator  import test_suite_all_AzimuthalIntegration
from testHistogram            import test_suite_all_Histogram
from testPeakPicking          import test_suite_all_PeakPicking
from testGeometry             import test_suite_all_Geometry
from testMask                 import test_suite_all_Mask
from testOpenCL               import test_suite_all_OpenCL

def test_suite_all():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_suite_all_Histogram())
    testSuite.addTest(test_suite_all_GeometryRefinement())
    testSuite.addTest(test_suite_all_AzimuthalIntegration())
    testSuite.addTest(test_suite_all_PeakPicking())
    testSuite.addTest(test_suite_all_Geometry())
    testSuite.addTest(test_suite_all_Mask())
    testSuite.addTest(test_suite_all_OpenCL())

    return testSuite

if __name__ == '__main__':

    mysuite = test_suite_all()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)

