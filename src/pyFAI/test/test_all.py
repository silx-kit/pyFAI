#!/usr/bin/env python
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2024 European Synchrotron Radiation Facility, Grenoble, France
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

"""Test suite for all pyFAI modules."""

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "09/07/2025"

import logging
import sys
import unittest

from ..gui import test as test_gui
from ..opencl import test as test_opencl
from . import (
    test_average,
    test_azimuthal_integrator,
    test_bayes,
    test_bilinear,
    test_bispev,
    test_blob_detection,
    test_bug_regression,
    test_calibrant,
    test_containers,
    test_convolution,
    test_crystallography,
    test_csr,
    test_detector,
    test_distortion,
    test_dummy,
    test_error_model,
    test_export,
    test_fiber_integrator,
    test_flat,
    test_geometry,
    test_geometry_refinement,
    test_goniometer,
    test_histogram,
    test_integrate,
    test_integrate_app,
    test_integrate_config,
    test_invert_geometry,
    test_io,
    test_io_diffmap_config,
    test_io_image,
    test_mask,
    test_massif,
    test_medfilt_engine,
    test_method_registry,
    test_multi_geometry,
    test_parallax,
    test_peak_picking,
    test_polarization,
    test_preproc,
    test_pyfai_api,
    test_rectangle,
    test_ring_extraction,
    test_saxs,
    test_scripts,
    test_sparse,
    test_sparse_builder,
    test_spline,
    test_split_pixel,
    test_uncertainties,
    test_units,
    test_utils,
    test_utils_ellipse,
    test_utils_header,
    test_utils_mathutil,
    test_utils_shell,
    test_utils_stringutil,
    test_watershed,
    test_worker,
)
from .utilstest import UtilsTest

logger = logging.getLogger(__name__)


def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(test_gui.suite())
    testsuite.addTest(test_units.suite())
    testsuite.addTest(test_average.suite())
    testsuite.addTest(test_dummy.suite())
    testsuite.addTest(test_histogram.suite())
    testsuite.addTest(test_geometry_refinement.suite())
    testsuite.addTest(test_azimuthal_integrator.suite())
    testsuite.addTest(test_peak_picking.suite())
    testsuite.addTest(test_geometry.suite())
    testsuite.addTest(test_mask.suite())
    testsuite.addTest(test_method_registry.suite())
    testsuite.addTest(test_export.suite())
    testsuite.addTest(test_saxs.suite())
    testsuite.addTest(test_integrate.suite())
    testsuite.addTest(test_integrate_app.suite())
    testsuite.addTest(test_integrate_config.suite())
    testsuite.addTest(test_bilinear.suite())
    testsuite.addTest(test_distortion.suite())
    testsuite.addTest(test_flat.suite())
    testsuite.addTest(test_utils.suite())
    testsuite.addTest(test_detector.suite())
    testsuite.addTest(test_convolution.suite())
    testsuite.addTest(test_sparse.suite())
    testsuite.addTest(test_csr.suite())
    testsuite.addTest(test_blob_detection.suite())
    testsuite.addTest(test_io.suite())
    testsuite.addTest(test_io_image.suite())
    testsuite.addTest(test_calibrant.suite())
    testsuite.addTest(test_polarization.suite())
    testsuite.addTest(test_split_pixel.suite())
    testsuite.addTest(test_bispev.suite())
    testsuite.addTest(test_bug_regression.suite())
    testsuite.addTest(test_watershed.suite())
    testsuite.addTest(test_multi_geometry.suite())
    testsuite.addTest(test_worker.suite())
    testsuite.addTest(test_utils_shell.suite())
    testsuite.addTest(test_utils_stringutil.suite())
    testsuite.addTest(test_utils_mathutil.suite())
    testsuite.addTest(test_utils_header.suite())
    testsuite.addTest(test_utils_ellipse.suite())
    testsuite.addTest(test_preproc.suite())
    testsuite.addTest(test_bayes.suite())
    testsuite.addTest(test_scripts.suite())
    testsuite.addTest(test_spline.suite())
    testsuite.addTest(test_sparse_builder.suite())
    testsuite.addTest(test_goniometer.suite())
    testsuite.addTest(test_opencl.suite())
    testsuite.addTest(test_pyfai_api.suite())
    testsuite.addTest(test_invert_geometry.suite())
    testsuite.addTest(test_massif.suite())
    testsuite.addTest(test_rectangle.suite())
    testsuite.addTest(test_parallax.suite())
    testsuite.addTest(test_error_model.suite())
    testsuite.addTest(test_uncertainties.suite())
    testsuite.addTest(test_ring_extraction.suite())
    testsuite.addTest(test_fiber_integrator.suite())
    testsuite.addTest(test_medfilt_engine.suite())
    testsuite.addTest(test_containers.suite())
    testsuite.addTest(test_io_diffmap_config.suite())
    testsuite.addTest(test_crystallography.suite())
    return testsuite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    if runner.run(suite()).wasSuccessful():
        UtilsTest.clean_up()
    else:
        sys.exit(1)
