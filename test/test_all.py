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
Test suite for all pyFAI modules.
"""

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "27/05/2015"

import sys
import os
import unittest
if __name__ == '__main__':
    import pkgutil
    __path__ = pkgutil.extend_path([os.path.dirname(__file__)], "pyFAI.test")
from .utilstest import UtilsTest, getLogger

logger = getLogger("test_all")


from .test_dummy import test_suite_all_dummy
from .test_geometry_refinement import test_suite_all_GeometryRefinement
from .test_azimuthal_integrator import test_suite_all_AzimuthalIntegration
from .test_histogram import test_suite_all_Histogram
from .test_peak_picking import test_suite_all_PeakPicking
from .test_geometry import test_suite_all_Geometry
from .test_mask import test_suite_all_Mask
from .test_openCL import test_suite_all_OpenCL
from .test_export import test_suite_all_Export
from .test_saxs import test_suite_all_Saxs
from .test_integrate import test_suite_all_Integrate1d
from .test_bilinear import test_suite_all_bilinear
from .test_distortion import test_suite_all_distortion
from .test_flat import test_suite_all_Flat
from .test_utils import test_suite_all_Utils
from .test_polarization import test_suite_all_polarization
from .test_detector import test_suite_all_detectors
from .test_convolution import test_suite_all_convolution
from .test_sparse import test_suite_all_sparse
from .test_csr import test_suite_all_OpenCL_CSR
from .test_blob_detection import test_suite_all_blob_detection
from .test_marchingsquares import test_suite_all_marchingsquares
from .test_io import test_suite_all_io
from .test_calibrant import test_suite_all_calibrant
from .test_split_pixel import test_suite_all_split
from .test_bispev import test_suite_all_bispev
from .test_bug_regression import test_suite_bug_regression
from .test_multi_geometry import test_suite_all_multi_geometry
from .test_watershed import test_suite_all_watershed


def test_suite_all():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_suite_all_dummy())
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
    testSuite.addTest(test_suite_all_marchingsquares())
    testSuite.addTest(test_suite_all_io())
    testSuite.addTest(test_suite_all_calibrant())
    testSuite.addTest(test_suite_all_polarization())
    testSuite.addTest(test_suite_all_split())
    testSuite.addTest(test_suite_all_bispev())
    testSuite.addTest(test_suite_bug_regression())
    testSuite.addTest(test_suite_all_watershed())
    testSuite.addTest(test_suite_all_multi_geometry())
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_all()
    runner = unittest.TextTestRunner()
    if runner.run(mysuite).wasSuccessful():
        UtilsTest.clean_up()
    else:
        sys.exit(1)
