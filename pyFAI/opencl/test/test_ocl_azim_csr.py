#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
Simple test of ocl_azim_csr within pyFAI
"""

from __future__ import division, print_function

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "2019 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "11/06/2019"

import logging
import numpy

import unittest
from .. import ocl, get_opencl_code
if ocl:
    import pyopencl.array
from ...test.utilstest import UtilsTest
from silx.opencl.common import _measure_workgroup_size
from ...azimuthalIntegrator import AzimuthalIntegrator
from ...method_registry import IntegrationMethod
from scipy.ndimage import gaussian_filter1d
logger = logging.getLogger(__name__)


@unittest.skipIf(UtilsTest.opencl is False, "User request to skip OpenCL tests")
@unittest.skipUnless(ocl, "PyOpenCl is missing")
class TestOclAzimCSR(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestOclAzimCSR, cls).setUpClass()
        if ocl:
            cls.ctx = ocl.create_context()
            if logger.getEffectiveLevel() <= logging.INFO:
                cls.PROFILE = True
                properties = pyopencl.command_queue_properties.PROFILING_ENABLE
                cls.queue = pyopencl.CommandQueue(cls.ctx, properties=properties)
            else:
                cls.PROFILE = False
                cls.queue = pyopencl.CommandQueue(cls.ctx)
            cls.max_valid_wg = 0
            if "cl_khr_int64_base_atomics" in cls.ctx.devices[0].extensions:
                cls.precise = True
            else:
                cls.precise = False
        cls.ai = AzimuthalIntegrator(detector="Pilatus100k")

    @classmethod
    def tearDownClass(cls):
        super(TestOclAzimCSR, cls).tearDownClass()
        print("Maximum valid workgroup size %s on device %s" % (cls.max_valid_wg, cls.ctx.devices[0]))
        cls.ctx = None
        cls.queue = None
        cls.ai = None

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_integrate_ng(self):
        """
        tests the 1d histogram kernel
        """
        from ..azim_csr import OCL_CSR_Integrator
        data = numpy.ones(self.ai.detector.shape)
        tth = self.ai.array_from_unit(unit="2th_deg")
        npt = 500

        method = IntegrationMethod.select_one_available(("no split", "histogram", "numpy"), dim=1, default=None, degradable=True)
        print(method)
        ref = self.ai._integrate1d_ng(data, npt, unit="2th_deg", method=method)
        print(ref)
        integrator = OCL_CSR_Integrator(tth, npt, devicetype="cpu")
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
        epsilon = 1e-5 if self.precise else 3e-3
        self.assertLess(err, epsilon, "normalization content is the same: %s<%s" % (err, epsilon))
        self.assertLess(abs(gaussian_filter1d(sig - ref, 9)).max(), 1.5, "normalization, after smoothing is flat")

        # histogram of signal
        ref = numpy.histogram(tth, npt, weights=data)[0]
        sig = res.signal.sum(axis=-1, dtype="float64")
        print(abs((sig - ref).sum()), abs(gaussian_filter1d(sig / ref - 1, 9)).max())
        self.assertLess(abs((sig - ref).sum()), 9e-5, "signal content is the same")
        self.assertLess(abs(gaussian_filter1d(sig / ref - 1, 9)).max(), 2e-5, "signal, after smoothing is flat")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testSuite = unittest.TestSuite()
    testSuite.addTest(loader(TestOclAzimCSR))
    return testSuite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
