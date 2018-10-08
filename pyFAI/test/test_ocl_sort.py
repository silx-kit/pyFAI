# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2018 European Synchrotron Radiation Facility, Grenoble, France
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

"""Test for OpenCL sorting on GPU"""

from __future__ import absolute_import, print_function, division
__license__ = "MIT"
__date__ = "08/10/2018"
__copyright__ = "2015, ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import unittest
import numpy
import logging
import warnings

from .utilstest import UtilsTest

logger = logging.getLogger(__name__)


from ..opencl import ocl
if ocl:
    from ..opencl import sort as ocl_sort

as_strided = numpy.lib.stride_tricks.as_strided


def sigma_clip(image, sigma_lo=3, sigma_hi=3, max_iter=5, axis=0):
    """Reference implementation in numpy"""
    image = image.copy()
    mask = numpy.logical_not(numpy.isfinite(image))
    dummies = mask.sum()
    image[mask] = numpy.NaN
    mean = numpy.nanmean(image, axis=axis, dtype="float64")
    std = numpy.nanstd(image, axis=axis, dtype="float64")
    for _ in range(max_iter):
        if axis == 0:
            mean2d = as_strided(mean, image.shape, (0, mean.strides[0]))
            std2d = as_strided(std, image.shape, (0, std.strides[0]))
        else:
            mean2d = as_strided(mean, image.shape, (mean.strides[0], 0))
            std2d = as_strided(std, image.shape, (std.strides[0], 0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            delta = (image - mean2d) / std2d
            mask = numpy.logical_or(delta > sigma_hi,
                                    delta < -sigma_lo)
        dummies = mask.sum()
        if dummies == 0:
            break
        image[mask] = numpy.NaN
        mean = numpy.nanmean(image, axis=axis, dtype="float64")
        std = numpy.nanstd(image, axis=axis, dtype="float64")
    return mean, std


@unittest.skipIf(UtilsTest.opencl is False, "User request to skip OpenCL tests")
@unittest.skipIf(ocl is None, "OpenCL is not available")
class TestOclSort(unittest.TestCase):
    KNOWN_INVALID_RESULTS = ["Portable Computing Language"]
    # This platform is known to process properly but giving wrong results.
    # See https://github.com/pocl/pocl/issues/617

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.shape = (128, 256)
        self.ary = numpy.random.random(self.shape).astype(numpy.float32)
        self.sorted_vert = numpy.sort(self.ary.copy(), axis=0)
        self.sorted_hor = numpy.sort(self.ary.copy(), axis=1)
        self.vector_vert = self.sorted_vert[self.shape[0] // 2]
        self.vector_hor = self.sorted_hor[:, self.shape[1] // 2]

        # Change to True to profile the code
        self.PROFILE = False

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.shape = self.ary = self.sorted_vert = self.sorted_hor = self.vector_vert = self.sorted_hor = None

    def test_sort_vert(self):
        s = ocl_sort.Separator(self.shape[0], self.shape[1], profile=self.PROFILE)
        res = s.sort_vertical(self.ary).get()
        self.assertTrue(numpy.allclose(self.sorted_vert, res), "vertical sort is OK")
        if self.PROFILE:
            s.log_profile()
            s.reset_timer()

    def test_filter_vert(self):
        s = ocl_sort.Separator(self.shape[0], self.shape[1], profile=self.PROFILE)
        res = s.filter_vertical(self.ary).get()
        self.assertTrue(numpy.allclose(self.vector_vert, res), "vertical filter is OK")
        if self.PROFILE:
            s.log_profile()
            s.reset_timer()

    def test_sort_hor(self):
        s = ocl_sort.Separator(self.shape[0], self.shape[1], profile=self.PROFILE)
        res = s.sort_horizontal(self.ary).get()
        self.assertTrue(numpy.allclose(self.sorted_hor, res), "horizontal sort is OK")
        if self.PROFILE:
            s.log_profile()
            s.reset_timer()

    def test_filter_hor(self):
        s = ocl_sort.Separator(self.shape[0], self.shape[1], profile=self.PROFILE)
        res = s.filter_horizontal(self.ary).get()
        self.assertTrue(numpy.allclose(self.vector_hor, res), "horizontal filter is OK")
        if self.PROFILE:
            s.log_profile()
            s.reset_timer()

    def test_mean_vert(self):
        s = ocl_sort.Separator(self.shape[0], self.shape[1], profile=self.PROFILE)
        res = s.mean_std_vertical(self.ary)
        m = res[0].get()
        d = res[1].get()
        self.assertTrue(numpy.allclose(self.ary.mean(axis=0, dtype="float64"), m,), "vertical mean is OK")
        self.assertTrue(numpy.allclose(self.ary.std(axis=0, dtype="float64"), d), "vertical std is OK")
        if self.PROFILE:
            s.log_profile()
            s.reset_timer()

    def test_mean_hor(self):
        s = ocl_sort.Separator(self.shape[0], self.shape[1], profile=self.PROFILE)
        res = s.mean_std_horizontal(self.ary)
        m = res[0].get()
        d = res[1].get()
        self.assertTrue(numpy.allclose(self.ary.mean(axis=1, dtype="float64"), m,), "horizontal mean is OK")
        self.assertTrue(numpy.allclose(self.ary.std(axis=1, dtype="float64"), d), "horizontal std is OK")
        if self.PROFILE:
            s.log_profile()
            s.reset_timer()

    def test_sigma_clip_vert(self):
        s = ocl_sort.Separator(self.shape[0], self.shape[1], profile=self.PROFILE)
        res = s.sigma_clip_vertical(self.ary, sigma_lo=3, sigma_hi=3, max_iter=5)
        m = res[0].get()
        d = res[1].get()
        mn, dn = sigma_clip(self.ary, sigma_lo=3, sigma_hi=3, max_iter=5, axis=0)
        platform = s.ctx.devices[0].platform.name
        if platform in self.KNOWN_INVALID_RESULTS:
            logger.warning("Broken platform: %s", platform)
        else:
            self.assertTrue(numpy.allclose(mn, m), "sigma_clipvertical mean is OK")
            self.assertTrue(numpy.allclose(dn, d), "sigma_clipvertical std is OK")
        if self.PROFILE:
            s.log_profile()
            s.reset_timer()

    def test_sigma_clip_hor(self):
        s = ocl_sort.Separator(self.shape[0], self.shape[1], profile=self.PROFILE)
        res = s.sigma_clip_horizontal(self.ary, sigma_lo=3, sigma_hi=3, max_iter=5)
        m = res[0].get()
        d = res[1].get()
        mn, dn = sigma_clip(self.ary, sigma_lo=3, sigma_hi=3, max_iter=5, axis=1)
        platform = s.ctx.devices[0].platform.name
        if platform in self.KNOWN_INVALID_RESULTS:
            logger.warning("Broken platform: %s", platform)
        else:
            self.assertTrue(numpy.allclose(mn, m,), "sigma_clip horizontal mean is OK")
            self.assertTrue(numpy.allclose(dn, d), "sigma_clip horizontal std is OK")
        if self.PROFILE:
            s.log_profile()
            s.reset_timer()


def suite():
    testsuite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite.addTest(loader(TestOclSort))
    return testsuite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
    UtilsTest.clean_up()
