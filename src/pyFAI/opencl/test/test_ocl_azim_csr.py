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
Simple test of ocl_azim_csr within pyFAI
"""

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "2019-2021 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "06/12/2024"

import logging
import numpy

import unittest
from .. import ocl
if ocl:
    import pyopencl.array
from ...test.utilstest import UtilsTest
from ...integrator.azimuthal import AzimuthalIntegrator
from ...method_registry import IntegrationMethod
from ...containers import ErrorModel
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
            if "cl_khr_int64_base_atomics" in cls.ctx.devices[0].extensions:
                cls.precise = True
            else:
                cls.precise = False
        cls.ai = AzimuthalIntegrator(detector="Pilatus100k")

    @classmethod
    def tearDownClass(cls):
        super(TestOclAzimCSR, cls).tearDownClass()
        logger.debug("Maximum valid workgroup size %s on device %s" % (cls.ctx.devices[0].max_work_group_size, cls.ctx.devices[0]))
        cls.ctx = None
        cls.queue = None
        cls.ai = None

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def integrate_ng(self, block_size=None, method_called="integrate_ng", extra=None):
        """
        tests the 1d histogram kernel, with variable workgroup size
        """
        if extra is None:
            extra={}
        from ..azim_csr import OCL_CSR_Integrator
        data = numpy.ones(self.ai.detector.shape)
        npt = 500
        unit = "r_mm"
        method = IntegrationMethod.select_one_available(("no", "histogram", "python"),
                                                        dim=1, default=None, degradable=True)
        csr_method = IntegrationMethod.select_one_available(("no", "csr", "cython"),
                                                            dim=1, default=None, degradable=False)

        # Retrieve the CSR array
        cpu_integrate = self.ai._integrate1d_ng(data, npt, unit=unit, method=csr_method)
        r_m = cpu_integrate[0]
        csr_engine = list(self.ai.engines.values())[0]
        csr = csr_engine.engine.lut
        ref = self.ai._integrate1d_ng(data, npt, unit=unit, method=method, error_model="poisson")
        integrator = OCL_CSR_Integrator(csr, data.size, block_size=block_size, empty=-1)
        solidangle = self.ai.solidAngleArray()
        res = integrator.__getattribute__(method_called)(data, solidangle=solidangle, error_model=ErrorModel.POISSON, **extra)
        # for info, res contains: position intensity error signal variance normalization count

        # Start with smth easy: the position
        self.assertTrue(numpy.allclose(r_m, ref[0]), "position are the same")
        # A bit harder: the count of pixels
        delta = ref.count - res.count
        if "AMD" in integrator.ctx.devices[0].platform.name:
            logger.warning("This test is known to be complicated for AMD-GPU, relax the constrains for them")
        else:
            if method_called=="integrate_ng":
                self.assertLessEqual(delta.max(), 1, "counts are almost the same")
                self.assertEqual(delta.sum(), 0, "as much + and -")
            elif method_called=="medfilt":
                pix = csr[2][1:]-csr[2][:-1]
                self.assertTrue(numpy.allclose(res.count, pix), "all pixels have been counted")


            # histogram of normalization
            # print(ref.sum_normalization)
            # print(res.normalization)
            err = abs((res.normalization - ref.sum_normalization))
            # print(err)
            self.assertLess(err.max(), 5e-4, "normalization content is the same: %s<5e-5" % (err.max))

            # histogram of signal
            self.assertLess(abs((res.signal - ref.sum_signal)).max(), 5e-5, "signal content is the same")

            # histogram of variance
            self.assertLess(abs((res.variance - ref.sum_variance)).max(), 5e-5, "signal content is the same")

            # Intensities are not that different:
            delta = ref.intensity - res.intensity
            # print(delta)
            self.assertLessEqual(abs(delta).max(), 1e-5, "intensity is almost the same")


    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_integrate_ng(self):
        """
        tests the 1d histogram kernel, default block size
        """
        self.integrate_ng(block_size=None)

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_integrate_ng_single(self):
        """
        tests the 1d histogram kernel, default block size
        """
        self.integrate_ng(block_size=1)

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_sigma_clip(self):
        """
        tests the sigma-clipping kernel, default block size
        """
        self.integrate_ng(None, "sigma_clip",{"cutoff":100.0, "cycle":0,})

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_medfilt(self):
        """
        tests the median filtering kernel, default block size
        """
        self.integrate_ng(None, "medfilt", {"quant_min":0, "quant_max":1})


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testSuite = unittest.TestSuite()
    testSuite.addTest(loader(TestOclAzimCSR))
    return testSuite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
