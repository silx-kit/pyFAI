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

"test suite for histogramming implementations"

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "20/03/2015"

import unittest
import time
import numpy
import logging
import sys
import platform
from numpy import cos
if __name__ == '__main__':
    import pkgutil, os
    __path__ = pkgutil.extend_path([os.path.dirname(__file__)], "pyFAI.test")
from .utilstest import UtilsTest, Rwp, getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]
from pyFAI.histogram import histogram, histogram2d
from pyFAI.splitBBoxCSR import HistoBBox1d, HistoBBox2d
if logger.getEffectiveLevel() == logging.DEBUG:
    import pylab
EPS32 = (1.0 + numpy.finfo(numpy.float32).eps)


class TestHistogram1d(unittest.TestCase):
    """basic test"""
    shape = (512, 512)
    npt = 500
    size = shape[0] * shape[1]
    maxI = 1000
    epsilon = 1.0e-4
    y, x = numpy.ogrid[:shape[0], :shape[1]]
    tth = numpy.sqrt(x * x + y * y).astype("float32")
    mod = 0.5 + 0.5 * cos(tth / 12) + 0.25 * cos(tth / 6) + 0.1 * cos(tth / 4)
    data = (numpy.random.poisson(maxI, shape) * mod).astype("uint16")
    data_sum = data.sum(dtype="float64")
    t0 = time.time()
    drange = (tth.min(), tth.max() * EPS32)
    unweight_numpy, bin_edges = numpy.histogram(tth, npt, range=drange)
    t1 = time.time()
    weight_numpy, bin_edges = numpy.histogram(tth, npt, weights=data.astype("float64"), range=drange)
    t2 = time.time()
    logger.info("Timing for Numpy   raw    histogram: %.3f", t1 - t0)
    logger.info("Timing for Numpy weighted histogram: %.3f", t2 - t1)
    bins_numpy = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    I_numpy = weight_numpy / numpy.maximum(1.0, unweight_numpy)
    t3 = time.time()
    bins_cython, I_cython, weight_cython, unweight_cython = histogram(tth, data, npt, pixelSize_in_Pos=0)
    t4 = time.time()
    logger.info("Timing for Cython  both   histogram: %.3f", t4 - t3)
    t3 = time.time()
    integrator = HistoBBox1d(tth, delta_pos0=None, pos1=None, delta_pos1=None,
                             bins=npt, pos0Range=drange, allow_pos0_neg=False,
                             unit="undefined",)
    t2 = time.time()
    bins_csr, I_csr, weight_csr, unweight_csr = integrator.integrate(data)
    t4 = time.time()
    logger.info("Timing for CSR  init: %.3fs, integrate: %0.3fs, both: %.3f", (t2 - t3), (t4 - t2), (t4 - t3))

    def test_count_numpy(self):
        """
        Test that the pixel count and the total intensity is conserved
        in numpy implementation
        """
        sump = self.unweight_numpy.sum(dtype="int64")
        intensity_obt = self.weight_numpy.sum(dtype="float64")
        delta = abs(sump - self.size)
        logger.info("Numpy: Total number of points: %s (%s expected), delta = %s", sump, self.size, delta)
        v = abs(intensity_obt - self.data_sum) / self.data_sum
        logger.info("Numpy: Total Intensity: %s (%s expected), variation = %s", intensity_obt, self.data_sum, v)
        self.assert_(delta == 0, msg="check all pixels were counted")
        summed_weight_hist = self.weight_numpy.sum(dtype="float64")
        self.assert_(summed_weight_hist == self.data_sum, msg="check all intensity is counted expected %s got %s" % (self.data_sum, summed_weight_hist))
        self.assert_(v < self.epsilon, msg="checks delta is lower than %s, got %s" % (self.epsilon, v))

    def test_count_cython(self):
        """
        Test that the pixel count and the total intensity is conserved
        in cython implementation
        """
        sump = int(self.unweight_cython.sum(dtype="float64"))
        intensity_obt = self.weight_cython.sum(dtype="float64")
        delta = abs(sump - self.size)
        logger.info("Cython: Total number of points: %s (%s expected), delta = %s", sump, self.size, delta)
        v = abs(intensity_obt - self.data_sum) / self.data_sum
        logger.info("Cython: Total Intensity: %s (%s expected), variation = %s", intensity_obt, self.data_sum, v)
        self.assert_(delta == 0, msg="check all pixels were counted expected %s got %s" % (self.size, sump))
        summed_weight_hist = self.weight_cython.sum(dtype="float64")
        self.assert_(summed_weight_hist == self.data_sum, msg="check all intensity is counted expected %s got %s" % (self.data_sum, summed_weight_hist))
        self.assertTrue(v < self.epsilon, msg="checks delta is lower than %s" % self.epsilon)

    def test_count_csr(self):
        """
        Test that the pixel count and the total intensity is conserved
        in cSR sparse matrix multiplacation implementation
        """
        sump = int(self.unweight_csr.sum(dtype="float64"))
        intensity_obt = self.weight_csr.sum(dtype="float64")
        delta = abs(sump - self.size)
        logger.info("CSR: Total number of points: %s (%s expected), delta = %s", sump, self.size, delta)
        v = abs(intensity_obt - self.data_sum) / self.data_sum
        logger.info("CSR: Total Intensity: %s (%s expected), variation = %s", intensity_obt, self.data_sum, v)
        self.assert_(delta == 0, msg="check all pixels were counted expected %s got %s" % (self.size, sump))
        summed_weight_hist = self.weight_csr.sum(dtype="float64")
        self.assert_(summed_weight_hist == self.data_sum, msg="check all intensity is counted expected %s got %s" % (self.data_sum, summed_weight_hist))
        self.assertTrue(v < self.epsilon, msg="checks delta is lower than %s" % self.epsilon)

    def test_numpy_vs_cython_vs_csr_1d(self):
        """
        Compare numpy histogram with cython simple implementation ans CSR
        """
        max_delta = abs(self.bins_numpy - self.bins_cython).max()
        logger.info("Bin-center position for cython/numpy, max delta=%s", max_delta)
        self.assert_(max_delta < self.epsilon, "Bin-center position for cython/numpy, max delta=%s" % max_delta)

        max_delta = abs(self.bins_numpy - self.bins_csr).max()
        logger.info("Bin-center position for csr/numpy, max delta=%s", max_delta)
        self.assert_(max_delta < self.epsilon, "Bin-center position for csr/numpy, max delta=%s" % max_delta)

        rwp1 = Rwp((self.bins_cython, self.I_cython), (self.bins_numpy, self.I_numpy))
        logger.info("Rwp Cython/Numpy = %.3f" % rwp1)
        self.assert_(rwp1 < self.epsilon, "Rwp Cython/Numpy = %.3f" % rwp1)

        rwp2 = Rwp((self.bins_csr, self.I_csr), (self.bins_numpy, self.I_numpy))
        logger.info("Rwp CSR/Numpy = %.3f" % rwp2)
        self.assert_(rwp2 < 3, "Rwp Cython/Numpy = %.3f" % rwp2)

        if logger.getEffectiveLevel() == logging.DEBUG:
            logger.info("Plotting results")
            fig = pylab.figure()
            fig.suptitle('Numpy /Cython R=%.3f, Numpy/CSR R=%.3f' % (rwp1, rwp2))
            sp = fig.add_subplot(111)
            sp.plot(self.bins_numpy, self.I_numpy, "-b", label='numpy')
            sp.plot(self.bins_cython, self.I_cython, "-r", label="cython")
            sp.plot(self.bins_csr, self.I_csr, "-g", label="CSR")
            handles, labels = sp.get_legend_handles_labels()
            fig.legend(handles, labels)
            fig.show()
            raw_input("Press enter to quit")

        delta_max = abs(self.unweight_numpy - self.unweight_cython).max()
        logger.info("pixel count difference numpy/cython : max delta=%s", delta_max)
        self.assert_(delta_max < 1, "numpy_vs_cython_1d max delta unweight = %s" % delta_max)
        delta_max = abs(self.I_cython - self.I_numpy).max()
        logger.info("Intensity count difference numpy/cython : max delta=%s", delta_max)
        self.assert_(delta_max < self.epsilon, "Intensity count difference numpy/cython : max delta=%s" % delta_max)

        #  TODO: fix this !!!
        delta_max = abs(self.unweight_numpy - self.unweight_csr).max()
        if delta_max > 0:
            logger.warning("pixel count difference numpy/csr : max delta=%s", delta_max)
        self.assert_(delta_max < 10, "numpy_vs_csr_1d max delta unweight = %s" % delta_max)
        delta_max = abs(self.I_csr - self.I_numpy).max()
        if delta_max > self.epsilon:
            logger.warning("Intensity count difference numpy/csr : max delta=%s", delta_max)
        self.assert_(delta_max < 0.62, "Intensity count difference numpy/csr : max delta=%s" % delta_max)


class TestHistogram2d(unittest.TestCase):
    """basic test for 2D histogram"""
    shape = (512, 512)
    size = shape[0] * shape[1]
    maxI = 1000
    epsilon = 1.1e-4
    y, x = numpy.ogrid[:shape[0], :shape[1]]
    tth = numpy.sqrt(x * x + y * y).astype("float32")
    mod = 0.5 + 0.5 * cos(tth / 12) + 0.25 * cos(tth / 6) + 0.1 * cos(tth / 4)
    data = (numpy.random.poisson(maxI, shape) * mod).astype("uint16")
    data_sum = data.sum(dtype="float64")
    npt = (400, 360)
    chi = numpy.arctan2(y, x).astype("float32")
    drange = [[tth.min(), tth.max() * EPS32], [chi.min(), chi.max() * EPS32]]
    t0 = time.time()
    unweight_numpy, tth_edges, chi_edges = numpy.histogram2d(tth.flatten(), chi.flatten(), npt, range=drange)
    t1 = time.time()
    weight_numpy, tth_edges, chi_edges = numpy.histogram2d(tth.flatten(), chi.flatten(), npt, weights=data.astype("float64").flatten(), range=drange)
    t2 = time.time()
    logger.info("Timing for Numpy  raw     histogram2d: %.3f", t1 - t0)
    logger.info("Timing for Numpy weighted histogram2d: %.3f", t2 - t1)
    tth_numpy = 0.5 * (tth_edges[1:] + tth_edges[:-1])
    chi_numpy = 0.5 * (chi_edges[1:] + chi_edges[:-1])
    I_numpy = weight_numpy / numpy.maximum(1.0, unweight_numpy)
    t3 = time.time()
    I_cython, tth_cython, chi_cython, weight_cython, unweight_cython = histogram2d(tth.flatten(), chi.flatten(), npt, data.flatten(), split=0)
    t4 = time.time()
    logger.info("Timing for Cython  both   histogram2d: %.3f", t4 - t3)
    t3 = time.time()
    integrator = HistoBBox2d(tth, None, chi, delta_pos1=None,
                             bins=npt, allow_pos0_neg=False, unit="undefined")
    t2 = time.time()
    I_csr, tth_csr, chi_csr, weight_csr, unweight_csr = integrator.integrate(data)
    t4 = time.time()
    logger.info("Timing for CSR  init: %.3fs, integrate: %0.3fs, both: %.3f", (t2 - t3), (t4 - t2), (t4 - t3))
    if platform.system() == "Linux":
        err_max_cnt = 0
    else:
        # Under windows or MacOSX, up to 1 bin error has been reported...
        err_max_cnt = 1

    def test_count_numpy(self):
        """
        Test that the pixel count and the total intensity is conserved
        in numpy implementation
        """
        sump = self.unweight_numpy.sum(dtype="int64")
        intensity_obt = self.weight_numpy.sum(dtype="float64")
        delta = abs(sump - self.size)
        logger.info("Numpy: Total number of points: %s (%s expected), delta = %s", sump, self.size, delta)
        v = abs(intensity_obt - self.data_sum) / self.data_sum
        logger.info("Numpy: Total Intensity: %s (%s expected), variation = %s", intensity_obt, self.data_sum, v)
        self.assert_(delta == 0, "Numpy: Total number of points: %s (%s expected), delta = %s" % (sump, self.size, delta))
        self.assert_(v < self.epsilon, "Numpy: Total Intensity: %s (%s expected), variation = %s" % (intensity_obt, self.data_sum, v))

    def test_count_cython(self):
        """
        Test that the pixel count and the total intensity is conserved
        in cython implementation
        """
        sump = int(self.unweight_cython.sum(dtype="int64"))
        intensity_obt = self.weight_cython.sum(dtype="float64")
        delta = abs(sump - self.size)
        logger.info("Cython: Total number of points: %s (%s expected), delta = %s", sump, self.size, delta)
        v = abs(intensity_obt - self.data_sum) / self.data_sum
        logger.info("Cython: Total Intensity: %s (%s expected), variation = %s", intensity_obt, self.data_sum, v)
        self.assert_(delta == 0, msg="check all pixels were counted")
        self.assert_(v < self.epsilon, msg="checks delta is lower than %s" % self.epsilon)

    def test_count_csr(self):
        """
        Test that the pixel count and the total intensity is conserved
        in csr implementation
        """
        sump = int(self.unweight_csr.sum(dtype="int64"))
        intensity_obt = self.weight_csr.sum(dtype="float64")
        delta = abs(sump - self.size)
        logger.info("CSR: Total number of points: %s (%s expected), delta = %s", sump, self.size, delta)
        v = abs(intensity_obt - self.data_sum) / self.data_sum
        logger.info("CSR: Total Intensity: %s (%s expected), variation = %s", intensity_obt, self.data_sum, v)
        self.assert_(delta == 0, msg="check all pixels were counted")
        self.assert_(v < self.epsilon, msg="checks delta is lower than %s" % self.epsilon)

    def test_numpy_vs_cython_vs_csr_2d(self):
        """
        Compare numpy histogram with cython simple implementation
        """
        max_delta = abs(self.tth_numpy - self.tth_cython).max()
        logger.info("Bin-center position for cython/numpy tth, max delta=%s", max_delta)
        self.assert_(max_delta < self.epsilon, "Bin-center position for cython/numpy tth, max delta=%s" % max_delta)
        max_delta = abs(self.chi_numpy - self.chi_cython).max()
        logger.info("Bin-center position for cython/numpy chi, max delta=%s", max_delta)
        self.assert_(max_delta < self.epsilon, "Bin-center position for cython/numpy chi, max delta=%s" % max_delta)

        delta_max = abs(self.unweight_numpy - self.unweight_cython).max()
        logger.info("pixel count difference numpy/cython : max delta=%s", delta_max)
        if delta_max > 0:
            logger.warning("pixel count difference numpy/cython : max delta=%s", delta_max)
        self.assert_(delta_max <= self.err_max_cnt, "pixel count difference numpy/cython : max delta=%s" % delta_max)
        delta_max = abs(self.I_cython - self.I_numpy).max()
        logger.info("Intensity count difference numpy/cython : max delta=%s", delta_max)
        self.assert_(delta_max < (self.err_max_cnt + self.epsilon) * self.maxI, "Intensity count difference numpy/cython : max delta=%s>%s" % (delta_max, (self.err_max_cnt + self.epsilon) * self.maxI))

        max_delta = abs(self.tth_numpy - self.tth_csr).max()
        logger.info("Bin-center position for csr/numpy tth, max delta=%s", max_delta)
        self.assert_(max_delta < self.epsilon, "Bin-center position for csr/numpy tth, max delta=%s" % max_delta)
        max_delta = abs(self.chi_numpy - self.chi_csr).max()
        logger.info("Bin-center position for csr/numpy chi, max delta=%s", max_delta)
        self.assert_(max_delta < self.epsilon, "Bin-center position for csr/numpy chi, max delta=%s" % max_delta)

        delta_max = abs(self.unweight_numpy - self.unweight_csr.T).max()
        if delta_max > 0:
            logger.warning("pixel count difference numpy/csr : max delta=%s", delta_max)
        self.assert_(delta_max <= self.err_max_cnt + 1, "pixel count difference numpy/csr : max delta=%s" % delta_max)
        delta_max = abs(self.I_csr.T - self.I_numpy).max()
        if delta_max > self.epsilon:
            logger.warning("Intensity count difference numpy/csr : max delta=%s", delta_max)
        self.assert_(delta_max < 29, "Intensity count difference numpy/csr : max delta=%s" % delta_max)


def test_suite_all_Histogram():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestHistogram1d("test_count_numpy"))
    testSuite.addTest(TestHistogram1d("test_count_cython"))
    testSuite.addTest(TestHistogram1d("test_count_csr"))
    testSuite.addTest(TestHistogram1d("test_numpy_vs_cython_vs_csr_1d"))
    testSuite.addTest(TestHistogram2d("test_count_numpy"))
    testSuite.addTest(TestHistogram2d("test_count_cython"))
    testSuite.addTest(TestHistogram2d("test_count_csr"))
    testSuite.addTest(TestHistogram2d("test_numpy_vs_cython_vs_csr_2d"))

    return testSuite

if __name__ == '__main__':

    mysuite = test_suite_all_Histogram()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
