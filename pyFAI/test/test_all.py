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
__date__ = "29/01/2016"

import sys
import os
import unittest
from .utilstest import UtilsTest, getLogger

logger = getLogger("test_all")


from . import test_dummy
from . import test_histogram
from . import test_geometry_refinement
from . import test_azimuthal_integrator
from . import test_peak_picking
from . import test_geometry
from . import test_mask
from . import test_openCL
from . import test_export
from . import test_saxs
from . import test_integrate
from . import test_bilinear
from . import test_distortion
from . import test_flat
from . import test_utils
from . import test_detector
from . import test_convolution
from . import test_sparse
from . import test_csr
from . import test_blob_detection
from . import test_marchingsquares
from . import test_io
from . import test_calibrant
from . import test_polarization
from . import test_split_pixel
from . import test_bispev
from . import test_bug_regression
from . import test_watershed
from . import test_multi_geometry
from . import test_ocl_sort
from . import test_worker
from . import test_integrate_widget


def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(test_dummy.suite())
    testsuite.addTest(test_histogram.suite())
    testsuite.addTest(test_geometry_refinement.suite())
    testsuite.addTest(test_azimuthal_integrator.suite())
    testsuite.addTest(test_peak_picking.suite())
    testsuite.addTest(test_geometry.suite())
    testsuite.addTest(test_mask.suite())
    testsuite.addTest(test_openCL.suite())
    testsuite.addTest(test_export.suite())
    testsuite.addTest(test_saxs.suite())
    testsuite.addTest(test_integrate.suite())
    testsuite.addTest(test_bilinear.suite())
    testsuite.addTest(test_distortion.suite())
    testsuite.addTest(test_flat.suite())
    testsuite.addTest(test_utils.suite())
    testsuite.addTest(test_detector.suite())
    testsuite.addTest(test_convolution.suite())
    testsuite.addTest(test_sparse.suite())
    testsuite.addTest(test_csr.suite())
    testsuite.addTest(test_blob_detection.suite())
    testsuite.addTest(test_marchingsquares.suite())
    testsuite.addTest(test_io.suite())
    testsuite.addTest(test_calibrant.suite())
    testsuite.addTest(test_polarization.suite())
    testsuite.addTest(test_split_pixel.suite())
    testsuite.addTest(test_bispev.suite())
    testsuite.addTest(test_bug_regression.suite())
    testsuite.addTest(test_watershed.suite())
    testsuite.addTest(test_multi_geometry.suite())
    testsuite.addTest(test_ocl_sort.suite())
    testsuite.addTest(test_worker.suite())
    testsuite.addTest(test_integrate_widget.suite())
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    if runner.run(suite()).wasSuccessful():
        UtilsTest.clean_up()
    else:
        sys.exit(1)
