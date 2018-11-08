#!/usr/bin/env python
# coding: utf-8
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

from __future__ import absolute_import, division, print_function

"""Test suite for histogramming implementations"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "08/11/2018"

import unittest
import time
import numpy
import logging
from numpy import cos
logger = logging.getLogger(__name__)
from ..ext.histogram import histogram, histogram2d, histogram2d_preproc
from ..ext.splitBBoxCSR import HistoBBox1d, HistoBBox2d
from ..third_party import six
from ..utils import mathutil

if logger.getEffectiveLevel() == logging.DEBUG:
    import pylab
EPS32 = (1.0 + numpy.finfo(numpy.float32).eps)


class TestHistogram1d(unittest.TestCase):
    """basic test"""
    @classmethod
    def setUpClass(cls):
        super(TestHistogram1d, cls).setUpClass()

        # CSR logger should stop complaining about desactivated
        csr_logger = logging.getLogger("pyFAI.ext.splitBBoxCSR")
        csr_logger.setLevel(logging.ERROR)

        shape = (512, 512)
        npt = 500
        cls.size = shape[0] * shape[1]
        maxI = 1000
        cls.epsilon = 1.0e-4
        cls.epsilon_csr = 0.33
        y, x = numpy.ogrid[:shape[0], :shape[1]]
        tth = numpy.sqrt(x * x + y * y)  # .astype("float32")
        mod = 0.5 + 0.5 * numpy.cos(tth / 12) + 0.25 * numpy.cos(tth / 6) + 0.1 * numpy.cos(tth / 4)
        # data = (numpy.random.poisson(maxI, shape) * mod).astype("uint16")
        data = (numpy.ones(shape) * maxI * mod).astype("uint16")
        cls.data_sum = data.sum(dtype="float64")
        t0 = time.time()
        drange = (tth.min(), tth.max() * EPS32)  # works as tth>0
        cls.unweight_numpy, _bin_edges = numpy.histogram(tth, npt, range=drange)
        t1 = time.time()
        cls.weight_numpy, bin_edges = numpy.histogram(tth, npt, weights=data.astype("float64"), range=drange)
        t2 = time.time()
        logger.info("Timing for Numpy   raw    histogram: %.3f", t1 - t0)
        logger.info("Timing for Numpy weighted histogram: %.3f", t2 - t1)
        cls.bins_numpy = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        cls.I_numpy = cls.weight_numpy / numpy.maximum(1.0, cls.unweight_numpy)
        t3 = time.time()
        cls.bins_cython, cls.I_cython, cls.weight_cython, cls.unweight_cython = histogram(tth, data, npt, pixelSize_in_Pos=0)
        t4 = time.time()
        logger.info("Timing for Cython  both   histogram: %.3f", t4 - t3)
        t3 = time.time()
        integrator = HistoBBox1d(tth, delta_pos0=None, pos1=None, delta_pos1=None,
                                 bins=npt, allow_pos0_neg=False,
                                 unit="undefined",)
        t2 = time.time()
        cls.bins_csr, cls.I_csr, cls.weight_csr, cls.unweight_csr = integrator.integrate(data)
        t4 = time.time()
        logger.info("Timing for CSR  init: %.3fs, integrate: %0.3fs, both: %.3f", (t2 - t3), (t4 - t2), (t4 - t3))
        # Under Linux, windows or MacOSX, up to 1 bin error has been reported...
        cls.err_max_cnt = 0
        cls.err_max_cnt_csr = 8

    @classmethod
    def tearDownClass(cls):
        super(TestHistogram1d, cls).tearDownClass()
        cls.unweight_numpy = cls.bins_numpy = None
        cls.I_numpy = cls.weight_numpy = cls.bins_csr = None
        cls.data_sum = cls.size = cls.err_max_cnt = None
        cls.bins_csr = cls.I_csr = cls.weight_csr = cls.unweight_csr = None
        csr_logger = logging.getLogger("pyFAI.ext.splitBBoxCSR")
        csr_logger.setLevel(logging.WARNING)

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
        self.assertTrue(delta == 0, msg="check all pixels were counted")
        summed_weight_hist = self.weight_numpy.sum(dtype="float64")
        self.assertTrue(summed_weight_hist == self.data_sum, msg="check all intensity is counted expected %s got %s" % (self.data_sum, summed_weight_hist))
        self.assertTrue(v < self.epsilon, msg="checks delta is lower than %s, got %s" % (self.epsilon, v))

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
        self.assertTrue(delta == 0, msg="check all pixels were counted expected %s got %s" % (self.size, sump))
        summed_weight_hist = self.weight_cython.sum(dtype="float64")
        self.assertTrue(summed_weight_hist == self.data_sum, msg="check all intensity is counted expected %s got %s" % (self.data_sum, summed_weight_hist))
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
        self.assertTrue(delta == 0, msg="check all pixels were counted expected %s got %s" % (self.size, sump))
        summed_weight_hist = self.weight_csr.sum(dtype="float64")
        self.assertTrue(summed_weight_hist == self.data_sum, msg="check all intensity is counted expected %s got %s" % (self.data_sum, summed_weight_hist))
        self.assertTrue(v < self.epsilon, msg="checks delta is lower than %s" % self.epsilon)

    def test_numpy_vs_cython_vs_csr_1d(self):
        """
        Compare numpy histogram with cython simple implementation ans CSR
        """
        max_delta = abs(self.bins_numpy - self.bins_cython).max()
        logger.info("Bin-center position for cython/numpy, max delta=%s", max_delta)
        self.assertTrue(max_delta < self.epsilon, "Bin-center position for cython/numpy, max delta=%s" % max_delta)

        max_delta = abs(self.bins_numpy - self.bins_csr).max()
        logger.info("Bin-center position for csr/numpy, max delta=%s", max_delta)
        self.assertTrue(max_delta < self.epsilon, "Bin-center position for csr/numpy, max delta=%s" % max_delta)

        rwp1 = mathutil.rwp((self.bins_cython, self.I_cython), (self.bins_numpy, self.I_numpy))
        logger.info("Rwp Cython/Numpy = %.3f", rwp1)
        self.assertTrue(rwp1 < self.epsilon, "Rwp Cython/Numpy = %.3f" % rwp1)

        rwp2 = mathutil.rwp((self.bins_csr, self.I_csr), (self.bins_numpy, self.I_numpy))
        logger.info("Rwp CSR/Numpy = %.3f", rwp2)
        self.assertTrue(rwp2 < 3, "Rwp Cython/Numpy = %.3f" % rwp2)

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
            six.moves.input("Press enter to quit")

        delta_max = abs(self.unweight_numpy - self.unweight_cython).max()
        logger.info("1d pixel count difference numpy/cython : max delta=%s", delta_max)

        if delta_max > 0:
            logger.warning("1d pixel count difference numpy/cython : max delta=%s", delta_max)
        self.assertTrue(delta_max <= self.err_max_cnt, "1d pixel count difference numpy/cython : max delta=%s" % delta_max)

        delta_max = abs(self.I_cython - self.I_numpy).max()
        logger.info("Intensity count difference numpy/cython : max delta=%s", delta_max)
        self.assertTrue(delta_max < self.epsilon, "Intensity count difference numpy/cython : max delta=%s" % delta_max)

        delta_max = abs(self.unweight_numpy - self.unweight_csr).max()

        self.assertTrue(delta_max <= self.err_max_cnt_csr, "numpy_vs_csr_1d max delta unweight = %s" % delta_max)
        delta_max = abs(self.I_csr - self.I_numpy).max()
        self.assertTrue(delta_max < self.epsilon_csr, "Intensity count difference numpy/csr : max delta=%s" % delta_max)


class TestHistogram2d(unittest.TestCase):

    """basic test for 2D histogram"""

    @classmethod
    def setUpClass(cls):
        super(TestHistogram2d, cls).setUpClass()

        # CSR logger should stop complaining about desactivated
        csr_logger = logging.getLogger("pyFAI.ext.splitBBoxCSR")
        csr_logger.setLevel(logging.ERROR)

        shape = (512, 512)
        cls.size = shape[0] * shape[1]
        cls.maxI = 1000
        cls.epsilon = 1.3e-4
        cls.epsilon_csr = 8.84
        y, x = numpy.ogrid[:shape[0], :shape[1]]
        tth = numpy.sqrt(x * x + y * y).astype("float32")
        mod = 0.5 + 0.5 * cos(tth / 12) + 0.25 * cos(tth / 6) + 0.1 * cos(tth / 4)
        # _data = (numpy.random.poisson(cls.maxI, shape) * mod).astype("uint16")
        data = (numpy.ones(shape) * cls.maxI * mod).astype("uint16")
        data_prep = numpy.empty((shape + (3,)), dtype="uint16")
        data_prep[..., 0] = data
        data_prep[..., 1] = data
        data_prep[..., 2] = 1
        cls.data_sum = data.sum(dtype="float64")
        npt = (400, 360)
        chi = numpy.arctan2(y, x).astype("float32")
        drange = [[tth.min(), tth.max() * EPS32], [chi.min(), chi.max() * EPS32]]
        t0 = time.time()
        cls.unweight_numpy, _tth_edges, _chi_edges = numpy.histogram2d(tth.flatten(), chi.flatten(), npt, range=drange)
        t1 = time.time()
        cls.weight_numpy, tth_edges, chi_edges = numpy.histogram2d(tth.flatten(),
                                                                   chi.flatten(),
                                                                   npt, weights=data.astype("float64").flatten(),
                                                                   range=drange)
        t2 = time.time()
        logger.info("Timing for Numpy  raw     histogram2d: %.3f", t1 - t0)
        logger.info("Timing for Numpy weighted histogram2d: %.3f", t2 - t1)
        cls.tth_numpy = 0.5 * (tth_edges[1:] + tth_edges[:-1])
        cls.chi_numpy = 0.5 * (chi_edges[1:] + chi_edges[:-1])
        cls.I_numpy = cls.weight_numpy / numpy.maximum(1.0, cls.unweight_numpy)
        t3 = time.time()
        cls.I_cython, cls.tth_cython, cls.chi_cython, cls.weight_cython, cls.unweight_cython = histogram2d(tth.flatten(),
                                                                                                           chi.flatten(),
                                                                                                           npt,
                                                                                                           data.flatten(),
                                                                                                           split=0)
        t4 = time.time()
        logger.info("Timing for Cython  both   histogram2d: %.3f", t4 - t3)
        t3 = time.time()
        integrator = HistoBBox2d(tth, None, chi, delta_pos1=None,
                                 bins=npt, allow_pos0_neg=False, unit="undefined")
        t2 = time.time()
        cls.I_csr, cls.tth_csr, cls.chi_csr, cls.weight_csr, cls.unweight_csr = integrator.integrate(data)
        t4 = time.time()
        logger.info("Timing for CSR  init: %.3fs, integrate: %0.3fs, both: %.3f", (t2 - t3), (t4 - t2), (t4 - t3))

#         print(tth.size, chi.size, data_prep.size, data_prep.shape)
        t5 = time.time()
        cls.histo_ng_res = histogram2d_preproc(tth.ravel(),
                                               chi.ravel(),
                                               npt,
                                               data_prep,
                                               split=False)
        t6 = time.time()
        logger.info("Timing for Cython  histogram2d_preproc: %.3f", t6 - t5)
    #     if platform.system() == "Linux":
    #         err_max_cnt = 0
    #     else:
        # Under windows or MacOSX, up to 1 bin error has been reported...
        cls.err_max_cnt = 1

    @classmethod
    def tearDownClass(cls):
        super(TestHistogram2d, cls).tearDownClass()
        cls.I_numpy = cls.size = cls.err_max_cnt = None
        cls.epsilon = cls.tth_numpy = None
        cls.I_csr = cls.tth_csr = cls.chi_csr = cls.weight_csr = cls.unweight_csr = None
        cls.I_cython = cls.tth_cython = cls.chi_cython = cls.weight_cython = cls.unweight_cython
        cls.unweight_numpy = cls.weight_numpy = None
        cls.maxI = None
        cls.histo_ng_res = None

        csr_logger = logging.getLogger("pyFAI.ext.splitBBoxCSR")
        csr_logger.setLevel(logging.WARNING)

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
        self.assertTrue(delta == 0, "Numpy: Total number of points: %s (%s expected), delta = %s" % (sump, self.size, delta))
        self.assertTrue(v < self.epsilon, "Numpy: Total Intensity: %s (%s expected), variation = %s" % (intensity_obt, self.data_sum, v))

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
        self.assertTrue(delta == 0, msg="check all pixels were counted")
        self.assertTrue(v < self.epsilon, msg="checks delta is lower than %s" % self.epsilon)

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
        self.assertTrue(delta == 0, msg="check all pixels were counted")
        self.assertTrue(v < self.epsilon, msg="checks delta is lower than %s" % self.epsilon)

    def test_numpy_vs_cython_vs_csr_2d(self):
        """
        Compare numpy histogram with cython simple implementation
        """
        max_delta = abs(self.tth_numpy - self.tth_cython).max()
        logger.info("Bin-center position for cython/numpy tth, max delta=%s", max_delta)
        self.assertTrue(max_delta < self.epsilon, "Bin-center position for cython/numpy tth, max delta=%s" % max_delta)
        max_delta = abs(self.chi_numpy - self.chi_cython).max()
        logger.info("Bin-center position for cython/numpy chi, max delta=%s", max_delta)
        self.assertTrue(max_delta < self.epsilon, "Bin-center position for cython/numpy chi, max delta=%s" % max_delta)

        delta_max = abs(self.unweight_numpy - self.unweight_cython).max()
        logger.info("2d pixel count difference numpy/cython : max delta=%s", delta_max)
        if delta_max > 0:
            logger.warning("2d pixel count difference numpy/cython : max delta=%s", delta_max)
        self.assertTrue(delta_max <= self.err_max_cnt, "2d pixel count difference numpy/cython : max delta=%s" % delta_max)
        delta_max = abs(self.I_cython - self.I_numpy).max()
        logger.info("Intensity count difference numpy/cython : max delta=%s", delta_max)
        self.assertTrue(delta_max < (self.err_max_cnt + self.epsilon) * self.maxI, "Intensity count difference numpy/cython : max delta=%s>%s" % (delta_max, (self.err_max_cnt + self.epsilon) * self.maxI))

        max_delta = abs(self.tth_numpy - self.tth_csr).max()
        logger.info("Bin-center position for csr/numpy tth, max delta=%s", max_delta)
        self.assertTrue(max_delta < self.epsilon, "Bin-center position for csr/numpy tth, max delta=%s" % max_delta)
        max_delta = abs(self.chi_numpy - self.chi_csr).max()
        logger.info("Bin-center position for csr/numpy chi, max delta=%s", max_delta)
        self.assertTrue(max_delta < self.epsilon, "Bin-center position for csr/numpy chi, max delta=%s" % max_delta)

        delta_max = abs(self.unweight_numpy - self.unweight_csr.T).max()
        if delta_max > self.err_max_cnt:
            logger.warning("pixel count difference numpy/csr : max delta=%s", delta_max)
        self.assertTrue(delta_max <= self.err_max_cnt, "pixel count difference numpy/csr : max delta=%s" % delta_max)
        delta_max = abs(self.I_csr.T - self.I_numpy).max()
        if delta_max > self.epsilon_csr:
            logger.warning("Intensity count difference numpy/csr : max delta=%s", delta_max)
        self.assertTrue(delta_max <= self.epsilon_csr, "Intensity count difference numpy/csr : max delta=%s" % delta_max)

    def test_count_cython_ng(self):
        """
        Test that the pixel count and the total intensity is conserved
        in cython preprocessed implementation
        """
        prop = self.histo_ng_res[-1]
        unweighted = prop["count"]
        weighted = prop["signal"]
        sump = int(unweighted.sum(dtype="float64"))
        intensity_obt = weighted.sum(dtype="float64")
        delta = abs(sump - self.size)
        logger.info("Cython: Total number of points: %s (%s expected), delta = %s", sump, self.size, delta)
        v = abs(intensity_obt - self.data_sum) / self.data_sum
        logger.info("Cython: Total Intensity: %s (%s expected), variation = %s", intensity_obt, self.data_sum, v)
        self.assertEqual(delta, 0, msg="check all pixels were counted")
        self.assertLess(v, self.epsilon, msg="checks delta is lower than %s" % self.epsilon)
        self.assertEqual(abs(prop["signal"] - prop["variance"]).max(), 0, "variance == signal")
        self.assertEqual(abs(prop["count"] - prop["norm"]).max(), 0, "count == norm")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestHistogram1d))
    testsuite.addTest(loader(TestHistogram2d))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
