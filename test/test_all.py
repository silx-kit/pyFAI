#!/usr/bin/env python
# coding: utf-8
#
#    Project: pyFAI tests class utilities
#             https://github.com/kif/pyFAI
#
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
__data__ = "2014-03-08"

import sys
import unittest

from utilstest import UtilsTest, getLogger
logger = getLogger("test_all")

from test_geometry_refinement  import test_suite_all_GeometryRefinement
from test_azimuthal_integrator import test_suite_all_AzimuthalIntegration
from test_histogram            import test_suite_all_Histogram
from test_peak_picking         import test_suite_all_PeakPicking
from test_geometry             import test_suite_all_Geometry
from test_mask                 import test_suite_all_Mask
from test_openCL               import test_suite_all_OpenCL
from test_export               import test_suite_all_Export
from test_saxs                 import test_suite_all_Saxs
from test_integrate            import test_suite_all_Integrate1d
from test_bilinear             import test_suite_all_bilinear
from test_distortion           import test_suite_all_distortion
from test_flat                 import test_suite_all_Flat
from test_utils                import test_suite_all_Utils
from test_polarization         import test_suite_all_Polarization
from test_detector             import test_suite_all_detectors
from test_convolution          import test_suite_all_convolution
from test_sparse               import test_suite_all_sparse
from test_csr                  import test_suite_all_OpenCL_CSR
from test_blob_detection       import test_suite_all_blob_detection

def test_suite_all():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_suite_all_Histogram())
    testSuite.addTest(test_suite_all_GeometryRefinement())
    testSuite.addTest(test_suite_all_AzimuthalIntegration())
    testSuite.addTest(test_suite_all_PeakPicking())
    testSuite.addTest(test_suite_all_Geometry())
    testSuite.addTest(test_suite_all_Mask())
    testSuite.addTest(test_suite_all_OpenCL())
    testSuite.addTest(test_suite_all_Export())
    testSuite.addTest(test_suite_all_Saxs())
    testSuite.addTest(test_suite_all_Integrate1d())
    testSuite.addTest(test_suite_all_bilinear())
    testSuite.addTest(test_suite_all_distortion())
    testSuite.addTest(test_suite_all_Flat())
    testSuite.addTest(test_suite_all_Utils())
    testSuite.addTest(test_suite_all_detectors())
    testSuite.addTest(test_suite_all_convolution())
    testSuite.addTest(test_suite_all_sparse())
    testSuite.addTest(test_suite_all_OpenCL_CSR())
    testSuite.addTest(test_suite_all_blob_detection())
    return testSuite

if __name__ == '__main__':

    mysuite = test_suite_all()
    runner = unittest.TextTestRunner()
    if not runner.run(mysuite).wasSuccessful():
        sys.exit(1)

