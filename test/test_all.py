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

__doc__ = """Test suite for all pyFAI modules."""
__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "22/10/2015"

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
from . import test_utils
from .test_polarization import test_suite_all_polarization
from .test_detector import test_suite_all_detectors
from .test_convolution import test_suite_all_convolution
from .test_sparse import test_suite_all_sparse
from .test_csr import test_suite_all_OpenCL_CSR
from .test_blob_detection import test_suite_all_blob_detection
from .test_marchingsquares import test_suite_all_marchingsquares
from .test_io import test_suite_all_io
from .test_calibrant import test_suite_all_calibrant
from . import test_split_pixel
from .test_bispev import test_suite_all_bispev
from .test_bug_regression import test_suite_bug_regression
from .test_multi_geometry import test_suite_all_multi_geometry
from . import test_watershed
from .test_ocl_sort import test_suite_all_ocl_sort


def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(test_suite_all_dummy())
    testsuite.addTest(test_suite_all_Histogram())
    testsuite.addTest(test_suite_all_GeometryRefinement())
    testsuite.addTest(test_suite_all_AzimuthalIntegration())
    testsuite.addTest(test_suite_all_PeakPicking())
    testsuite.addTest(test_suite_all_Geometry())
    testsuite.addTest(test_suite_all_Mask())
    testsuite.addTest(test_suite_all_OpenCL())
    testsuite.addTest(test_suite_all_Export())
    testsuite.addTest(test_suite_all_Saxs())
    testsuite.addTest(test_suite_all_Integrate1d())
    testsuite.addTest(test_suite_all_bilinear())
    testsuite.addTest(test_suite_all_distortion())
    testsuite.addTest(test_suite_all_Flat())
    testsuite.addTest(test_utils.suite())
    testsuite.addTest(test_suite_all_detectors())
    testsuite.addTest(test_suite_all_convolution())
    testsuite.addTest(test_suite_all_sparse())
    testsuite.addTest(test_suite_all_OpenCL_CSR())
    testsuite.addTest(test_suite_all_blob_detection())
    testsuite.addTest(test_suite_all_marchingsquares())
    testsuite.addTest(test_suite_all_io())
    testsuite.addTest(test_suite_all_calibrant())
    testsuite.addTest(test_suite_all_polarization())
    testsuite.addTest(test_split_pixel.suite())
    testsuite.addTest(test_suite_all_bispev())
    testsuite.addTest(test_suite_bug_regression())
    testsuite.addTest(test_watershed.suite())
    testsuite.addTest(test_suite_all_multi_geometry())
    testsuite.addTest(test_suite_all_ocl_sort())
    return testsuite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    if runner.run(suite()).wasSuccessful():
        UtilsTest.clean_up()
    else:
        sys.exit(1)
