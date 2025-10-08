#!/usr/bin/env python3
# coding: utf-8
#
#    Project: Simple histogram in Python + OpenCL
#             https://github.com/silx-kit/pyFAI
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

"""
Simple test of histgrams within pyFAI
"""

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"

__copyright__ = "2019-2021 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "04/09/2025"

import logging
import numpy

import unittest
from .. import ocl
if ocl:
    import pyopencl.array
from ...test.utilstest import UtilsTest
from ...integrator.azimuthal import AzimuthalIntegrator
from ...containers import ErrorModel
from scipy.ndimage import gaussian_filter1d
logger = logging.getLogger(__name__)


@unittest.skipIf(UtilsTest.opencl is False, "User request to skip OpenCL tests")
@unittest.skipUnless(ocl, "PyOpenCl is missing")
class TestOclHistogram(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestOclHistogram, cls).setUpClass()
        if ocl:
            cls.ctx = ocl.create_context()
            if logger.getEffectiveLevel() <= logging.INFO:
                cls.PROFILE = True
                properties = pyopencl.command_queue_properties.PROFILING_ENABLE
                cls.queue = pyopencl.CommandQueue(cls.ctx, properties=properties)
            else:
                cls.PROFILE = False
                cls.queue = pyopencl.CommandQueue(cls.ctx)
            if "cl_khr_int64_base_atomics" in cls.ctx.devices[0].extensions:
                cls.precise = True
            else:
                cls.precise = False
            if "AMD" in cls.ctx.devices[0].platform.name:
                logger.warning("Decreasing precision on amdgpu")
                cls.precise = False

        cls.ai = AzimuthalIntegrator(detector="Pilatus100k")

    @classmethod
    def tearDownClass(cls):
        super(TestOclHistogram, cls).tearDownClass()
        logger.info("We were using device %s", cls.ctx.devices[0])
        cls.ctx = None
        cls.queue = None
        cls.ai = None

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_histogram1d(self):
        """
        tests the 1d histogram kernel
        """
        from ..azim_hist import OCL_Histogram1d
        data = numpy.ones(self.ai.detector.shape)
        tth = self.ai.array_from_unit(unit="2th_deg")
        npt = 500
        ref = self.ai.integrate1d_legacy(data, npt, unit="2th_deg", method="numpy")
        integrator = OCL_Histogram1d(tth, npt, ctx=self.ctx)
        precise = self.precise  and integrator.degraded is False
        solidangle = self.ai.solidAngleArray()
        res = integrator(data, solidangle=solidangle)

        # Start with smth easy: the position
        self.assertTrue(numpy.allclose(res[0], ref[0]), "position are the same")
        # A bit harder: the count of pixels
        delta = ref.count - res.count
        self.assertLessEqual(delta.max(), 2, "counts are almost the same")
        self.assertEqual(delta.sum(), 0, "as much + and -")

        # Intensities are not that different:
        delta = ref.intensity - res.intensity
        self.assertLessEqual(delta.max(), 1e-3, "intensity is almost the same")
        self.assertLessEqual((delta[1:-1] + delta[:-2] + delta[2:]).max(), 1e-3, "intensity is almost the same")

        # histogram of normalization
        ref = numpy.histogram(tth, npt, weights=solidangle)[0]
        sig = res.normalization.sum(axis=-1, dtype="float64")
        err = abs((sig - ref).sum())

        epsilon = 1e-5 if precise else 4e-3
        self.assertLess(err, epsilon, "normalization content is the same: %s<%s on device %s" % (err, epsilon, integrator.ctx.devices[0]))
        self.assertLess(abs(gaussian_filter1d(sig - ref, 9)).max(), 1.5, "normalization, after smoothing is flat")

        # histogram of signal
        ref = numpy.histogram(tth, npt, weights=data)[0]
        sig = res.signal.sum(axis=-1, dtype="float64")
        # print(abs((sig - ref).sum()), abs(gaussian_filter1d(sig / ref - 1, 9)).max())
        self.assertLess(abs((sig - ref).sum()), 9e-5, "signal content is the same")
        self.assertLess(abs(gaussian_filter1d(sig / ref - 1, 9)).max(), 2e-5, "signal, after smoothing is flat")

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_histogram2d(self):
        """
        tests the 2D histogram kernel
        """
        from ..azim_hist import OCL_Histogram2d
        from ...engines.histogram_engine import histogram2d_engine

        data = numpy.ones(self.ai.detector.shape)
        tth = self.ai.center_array(unit="2th_deg")
        chi = self.ai.center_array(unit="chi_deg")
        solidangle = self.ai.solidAngleArray()
        dummy = -42
        mini_rad = numpy.float32(tth.min())
        maxi_rad = numpy.float32(tth.max() * (1.0 + numpy.finfo(numpy.float32).eps))
        mini_azim = numpy.float32(chi.min())
        maxi_azim = numpy.float32(chi.max() * (1.0 + numpy.finfo(numpy.float32).eps))

        npt = (300, 36)

        ref = histogram2d_engine(tth, chi, npt,
                                 data,
                                 dark=None,
                                 flat=None,
                                 solidangle=solidangle,
                                 polarization=None,
                                 absorption=None,
                                 mask=None,
                                 dummy=dummy,
                                 delta_dummy=None,
                                 empty=dummy,
                                 normalization_factor=1.0,
                                 variance=None,
                                 dark_variance=None,
                                 error_model=ErrorModel.NO,
                                 radial_range=[mini_rad, maxi_rad],
                                 azimuth_range=[mini_azim, maxi_azim])

        # ref = self.ai._integrate2d_legacy(data, *npt, unit="2th_deg", method="numpy", dummy=dummy)
        integrator = OCL_Histogram2d(tth, chi, *npt, empty=dummy, profile=1)
        res = integrator(data, solidangle=solidangle)

        # Start with smth easy: the position
        self.assertTrue(numpy.allclose(res.radial, ref.radial), "radial position are the same")
        self.assertTrue(numpy.allclose(res.azimuthal, ref.azimuthal), "azimuthal position are the same")
        # A bit harder: the count of pixels

        delta = ref.count - res.count
        self.assertLessEqual(delta.max(), 2, "counts are almost the same")
        self.assertLessEqual(delta.sum(), 1, "as much + and -")
        lost = max(abs(delta).sum(), 1.1)
        # Intensities are not that different:
        delta = ref.intensity - res.intensity
        self.assertLessEqual(delta.max(), 1e-3, "intensity is almost the same")

        # histogram of normalization
        err = abs((res.normalization - ref.normalization).sum())
        allowed = lost * solidangle.max()
        self.assertLessEqual(err, allowed, "normalization content is the same: %s<=%s" % (err, allowed))

        # histogram of signal
        err = abs((res.signal - ref.signal).sum())
        allowed = lost * data.max()
        self.assertLessEqual(err, allowed, "signal content is the same: %s<=%s " % (err, allowed))


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testSuite = unittest.TestSuite()
    testSuite.addTest(loader(TestOclHistogram))
    return testSuite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
