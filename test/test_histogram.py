#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://forge.epn-campus.eu/projects/azimuthal
#
#    File: "$Id$"
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
__date__ = "19/10/2011"

import unittest
import time
import os
import numpy
import logging
import sys
from utilstest import UtilsTest, Rwp, getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]
from pyFAI.histogram import histogram, histogram2d
#histogram = sys.modules["pyFAI.histogram"].histogram
#histogram2d = sys.modules["pyFAI.histogram"].histogram2d
if logger.getEffectiveLevel() == logging.DEBUG:
    import pylab
EPS32 = (1.0 + numpy.finfo(numpy.float32).eps)
class test_histogram1d(unittest.TestCase):
    """basic test"""
    shape = (2048, 2048)#(numpy.random.randint(1000, 4000), numpy.random.randint(1000, 4000))
    npt = 1500#numpy.random.randint(1000, 4000)
    size = shape[0] * shape[1]
    epsilon = 1.0e-4
    tth = (numpy.random.random(shape).astype("float64"))
    data = numpy.random.random_integers(1, 65000, size=shape).astype("uint16")
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


    def test_count_numpy(self):
        """
        Test that the pixel count and the total intensity is conserved
        in numpy implementation
        """
        sump = int(self.unweight_numpy.sum(dtype="int64"))
        intensity_obt = self.I_numpy.sum(dtype="float64") * self.size / self.npt
        intensity_exp = self.data.sum(dtype="float64")
        delta = abs(sump - self.size)
        logger.info("Numpy: Total number of points: %s (%s expected), delta = %s", sump, self.size, delta)
        v = abs(intensity_obt - intensity_exp) / intensity_exp
        logger.info("Numpy: Total Intensity: %s (%s expected), variation = %s", intensity_obt, intensity_exp, v)
        self.assertEquals(delta, 0, msg="check all pixels were counted")
        self.assertTrue(v < self.epsilon, msg="checks delta is lower than %s" % self.epsilon)


    def test_count_cython(self):
        """
        Test that the pixel count and the total intensity is conserved
        in cython implementation
        """
        sump = int(self.unweight_cython.sum(dtype="float64"))
        intensity_obt = self.I_cython.sum(dtype="float64") * self.size / self.npt
        intensity_exp = self.data.sum(dtype="float64")
        delta = abs(sump - self.size)
        logger.info("Cython: Total number of points: %s (%s expected), delta = %s", sump, self.size, delta)
        v = abs(intensity_obt - intensity_exp) / intensity_exp
        logger.info("Cython: Total Intensity: %s (%s expected), variation = %s", intensity_obt, intensity_exp, v)
        self.assertEquals(delta, 0, msg="check all pixels were counted")
        self.assertTrue(v < self.epsilon, msg="checks delta is lower than %s" % self.epsilon)


    def test_numpy_vs_cython_1d(self):
        """
        Compare numpy histogram with cython simple implementation
        """
        max_delta = abs(self.bins_numpy - self.bins_cython).max()
        logger.info("Bin-center position for cython/numpy, max delta=%s", max_delta)
        self.assert_(max_delta < self.epsilon,"Bin-center position for cython/numpy, max delta=%s"% max_delta)
        rwp = Rwp((self.bins_cython, self.I_cython), (self.bins_numpy, self.I_numpy))
        logger.info("Rwp Cython/Numpy = %.3f" % rwp)
        self.assert_(rwp < 5.0, "Rwp Cython/Numpy = %.3f" % rwp)
        if logger.getEffectiveLevel() == logging.DEBUG:
            logger.info("Plotting results")
            fig = pylab.figure()
            fig.suptitle('Numpy vs Cython XRPD R=%.3f' % rwp)
            sp = fig.add_subplot(111)
            sp.plot(self.bins_numpy, self.I_numpy, "-b", label='numpy')
            sp.plot(self.bins_cython, self.I_cython, "-r", label="cython")
            handles, labels = sp.get_legend_handles_labels()
            fig.legend(handles, labels)
            fig.show()
            raw_input("Press enter to quit")
        delta_max = abs(self.unweight_numpy - self.unweight_cython).max()
        logger.info("pixel count difference numpy/cython : max delta=%s", delta_max)
        self.assert_(delta_max < 2, "numpy_vs_cython_1d max delta unweight = %s" % delta_max)
        delta_max = abs(self.I_cython - self.I_numpy).max()
        logger.info("Intensity count difference numpy/cython : max delta=%s", delta_max)
        self.assert_(delta_max < self.epsilon, "Intensity count difference numpy/cython : max delta=%s" % delta_max)


class test_histogram2d(unittest.TestCase):
    """basic test for 2D histogram"""
    shape = (2048, 2048)#(numpy.random.randint(1000, 4000), numpy.random.randint(1000, 4000))
    npt = (400, 360)
    size = shape[0] * shape[1]
    epsilon = 3.0e-4
    tth = (numpy.random.random(shape).astype("float64"))
    chi = (numpy.random.random(shape).astype("float64"))
    data = numpy.random.random_integers(1, 65000, size=shape).astype("uint16")
    t0 = time.time()
    drange = [[tth.min(), tth.max() * EPS32], [chi.min(), chi.max() * EPS32]]
    unweight_numpy, tth_edges, chi_edges = numpy.histogram2d(tth.flatten(), chi.flatten(), npt,range=drange)
    t1 = time.time()
    weight_numpy, tth_edges, chi_edges = numpy.histogram2d(tth.flatten(), chi.flatten(), npt, weights=data.astype("float64").flatten(),range=drange)
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

    def test_count_numpy(self):
        """
        Test that the pixel count and the total intensity is conserved
        in numpy implementation
        """
        sump = int(self.unweight_numpy.sum(dtype="int64"))
        intensity_obt = self.I_numpy.sum(dtype="float64") * self.size / float(self.npt[0] * self.npt[1])
        intensity_exp = self.data.sum(dtype="float64")
        delta = abs(sump - self.size)
        logger.info("Numpy: Total number of points: %s (%s expected), delta = %s", sump, self.size, delta)
        v = abs(intensity_obt - intensity_exp) / intensity_exp
        logger.info("Numpy: Total Intensity: %s (%s expected), variation = %s", intensity_obt, intensity_exp, v)
        self.assert_( delta == 0,"Numpy: Total number of points: %s (%s expected), delta = %s"%(sump, self.size, delta))
        self.assert_(v < self.epsilon,"Numpy: Total Intensity: %s (%s expected), variation = %s"%( intensity_obt, intensity_exp, v))


    def test_count_cython(self):
        """
        Test that the pixel count and the total intensity is conserved
        in cython implementation
        """
        sump = int(self.unweight_cython.sum(dtype="int64"))
        intensity_obt = self.I_cython.sum(dtype="float64") * self.size / float(self.npt[0] * self.npt[1])
        intensity_exp = self.data.sum(dtype="float64")
        delta = abs(sump - self.size)
        logger.info("Cython: Total number of points: %s (%s expected), delta = %s", sump, self.size, delta)
        v = abs(intensity_obt - intensity_exp) / intensity_exp
        logger.info("Cython: Total Intensity: %s (%s expected), variation = %s", intensity_obt, intensity_exp, v)
        self.assertEquals(delta, 0, msg="check all pixels were counted")
        self.assertTrue(v < self.epsilon, msg="checks delta is lower than %s" % self.epsilon)


    def test_numpy_vs_cython_2d(self):
        """
        Compare numpy histogram with cython simple implementation
        """
        max_delta = abs(self.tth_numpy - self.tth_cython).max()
        logger.info("Bin-center position for cython/numpy tth, max delta=%s", max_delta)
        self.assert_(max_delta < self.epsilon, "Bin-center position for cython/numpy tth, max delta=%s" % max_delta)
        max_delta = abs(self.chi_numpy - self.chi_cython).max()
        logger.info("Bin-center position for cython/numpy chi, max delta=%s", max_delta)
        self.assert_(max_delta < self.epsilon,"Bin-center position for cython/numpy chi, max delta=%s"% max_delta)

        delta_max = abs(self.unweight_numpy - self.unweight_cython).max()
        logger.info("pixel count difference numpy/cython : max delta=%s", delta_max)
        self.assert_(delta_max < 2, "pixel count difference numpy/cython : max delta=%s" % delta_max)
        delta_max = abs(self.I_cython - self.I_numpy).max()
        logger.info("Intensity count difference numpy/cython : max delta=%s", delta_max)
        self.assert_(delta_max < self.epsilon, "Intensity count difference numpy/cython : max delta=%s" % delta_max)


def test_suite_all_Histogram():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_histogram1d("test_count_numpy"))
    testSuite.addTest(test_histogram1d("test_count_cython"))
    testSuite.addTest(test_histogram1d("test_numpy_vs_cython_1d"))
    testSuite.addTest(test_histogram2d("test_count_numpy"))
    testSuite.addTest(test_histogram2d("test_count_cython"))
    testSuite.addTest(test_histogram2d("test_numpy_vs_cython_2d"))

    return testSuite

if __name__ == '__main__':

    mysuite = test_suite_all_Histogram()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
