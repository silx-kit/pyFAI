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
Simple test of ocl_azim_lut within pyFAI
"""

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "2019 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "15/01/2021"

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
class TestOclAzimLUT(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestOclAzimLUT, cls).setUpClass()
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
        cls.ai = AzimuthalIntegrator(detector="Pilatus100k")

    @classmethod
    def tearDownClass(cls):
        super(TestOclAzimLUT, cls).tearDownClass()
        logger.debug("Maximum valid workgroup size %s on device %s" % (cls.ctx.devices[0].max_work_group_size, cls.ctx.devices[0]))
        cls.ctx = None
        cls.queue = None
        cls.ai = None

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_integrate_ng(self):
        """
        tests the 1d histogram kernel
        """
        from ..azim_lut import OCL_LUT_Integrator
        data = numpy.ones(self.ai.detector.shape)
        npt = 500
        unit = "r_mm"
        method = IntegrationMethod.select_one_available(("no", "histogram", "python"),
                                                        dim=1, default=None, degradable=True)
        lut_method = IntegrationMethod.select_one_available(("no", "lut", "cython"),
                                                            dim=1, default=None, degradable=False)

        # Retrieve the LUT array
        cpu_integrate = self.ai._integrate1d_legacy(data, npt, unit=unit, method=lut_method)
        r_m = cpu_integrate[0]
        lut_engine = list(self.ai.engines.values())[0]
        lut = lut_engine.engine.lut
        ref = self.ai.integrate1d_ng(data, npt, unit=unit, method=method)
        integrator = OCL_LUT_Integrator(lut, data.size)
        solidangle = self.ai.solidAngleArray()
        res = integrator.integrate_ng(data, solidangle=solidangle)
        # for info, res contains: position intensity error signal variance normalization count

        # Start with smth easy: the position
        self.assertTrue(numpy.allclose(r_m, ref[0]), "position are the same")

        if "AMD" in integrator.ctx.devices[0].platform.name:
            logger.warning("This test is known to be complicated for AMD-GPU, relax the constrains for them")
        else:
            # A bit harder: the count of pixels
            delta = ref.count - res.count
            self.assertLessEqual(delta.max(), 1, "counts are almost the same")
            self.assertEqual(delta.sum(), 0, "as much + and -")

            # Intensities are not that different:
            delta = ref.intensity - res.intensity
            self.assertLessEqual(abs(delta.max()), 1e-5, "intensity is almost the same")

            # histogram of normalization
            ref = self.ai._integrate1d_ng(solidangle, npt, unit=unit, method=method).sum_signal
            sig = res.normalization
            err = abs((sig - ref).max())
            self.assertLess(err, 5e-5, "normalization content is the same: %s<5e-5" % (err))

            # histogram of signal
            ref = self.ai._integrate1d_ng(data, npt, unit=unit, method=method).sum_signal
            sig = res.signal
            self.assertLess(abs((sig - ref).sum()), 5e-5, "signal content is the same")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testSuite = unittest.TestSuite()
    testSuite.addTest(loader(TestOclAzimLUT))
    return testSuite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
