# coding: utf-8
#
#    Project: Fast Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  .
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  .
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.


__doc__ = """Re-implementation of numpy histograms using OpenMP"""
__author__ = "Jerome Kieffer"
__date__ = "31/05/2016"
__license__ = "MIT"
__copyright__ = "2011-2016, ESRF"
__contact__ = "jerome.kieffer@esrf.fr"

import cython
from cython.parallel cimport prange
from cython.view cimport array as cvarray
from libc.math cimport floor
from openmp cimport omp_set_num_threads, omp_get_max_threads, omp_get_thread_num
import logging
logger = logging.getLogger("pyFAI.histogram_omp")



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def histogram(numpy.ndarray pos not None, \
              numpy.ndarray weights not None, \
              int bins=100,
              bin_range=None,
              pixelSize_in_Pos=None,
              nthread=None,
              double empty=0.0,
              double normalization_factor=1.0):
    """
    Calculates histogram of pos weighted by weights

    @param pos: 2Theta array
    @param weights: array with intensities
    @param bins: number of output bins
    @param pixelSize_in_Pos: size of a pixels in 2theta: DESACTIVATED
    @param nthread: OpenMP is disabled. unused
    @param empty: value given to empty bins
    @param normalization_factor: divide the result by this value

    @return 2theta, I, weighted histogram, raw histogram
    """

    assert pos.size == weights.size
    assert bins > 1
    cdef:
        int  size = pos.size
        float[:] cpos = numpy.ascontiguousarray(pos.ravel(), dtype=numpy.float32)
        float[:] cdata = numpy.ascontiguousarray(weights.ravel(), dtype=numpy.float32)
        numpy.ndarray[numpy.float64_t, ndim = 1] out_data = numpy.zeros(bins, dtype="float64")
        numpy.ndarray[numpy.float64_t, ndim = 1] out_count = numpy.zeros(bins, dtype="float64")
        numpy.ndarray[numpy.float64_t, ndim = 1] out_merge = numpy.zeros(bins, dtype="float64")
        double delta, min0, max0
        double a = 0.0
        double d = 0.0
        double fbin = 0.0
        double tmp_count, tmp_data = 0.0
        double epsilon = 1e-10
        int bin = 0, i, idx, thread
    if pixelSize_in_Pos:
        logger.warning("No pixel splitting in histogram")

    if bin_range is not None:
        min0 = min(bin_range)
        max0 = max(bin_range) * EPS32
    else:
        min0 = pos.min()
        max0 = pos.max() * EPS32

    delta = (max0 - min0) / float(bins)

    if nthread is not None:
        if isinstance(nthread, int) and (nthread > 0):
            omp_set_num_threads(< int > nthread)
        else:
            nthread = omp_get_max_threads()
    else:
        nthread = omp_get_max_threads()
    cdef:
        double[:, :] big_count = cvarray(shape=(nthread, bins), itemsize=sizeof(double), format="d")
        double[:, :] big_data = cvarray(shape=(nthread, bins), itemsize=sizeof(double), format="d")

    big_count[:, :] = 0.0
    big_data[:, :] = 0.0

    if pixelSize_in_Pos:
        logger.warning("No pixel splitting in histogram")

    with nogil:
        for i in prange(size):
            d = cdata[i]
            a = cpos[i]
            fbin = get_bin_number(a, min0, delta)
            bin = < int > fbin
            if bin<0 or bin>= bins:
                continue
            thread = omp_get_thread_num()
            big_count[thread, bin] += 1.0
            big_data[thread, bin] += d

        for idx in prange(bins):
            tmp_count = 0.0
            tmp_data = 0.0
            for thread in range(omp_get_max_threads()):
                tmp_count = tmp_count + big_count[thread, idx]
                tmp_data = tmp_data + big_data[thread, idx]
            out_count[idx] += tmp_count
            out_data[idx] += tmp_data
            if out_count[idx] > epsilon:
                out_merge[idx] += tmp_data / tmp_count / normalization_factor
            else:
                out_merge[idx] += empty

    out_pos = numpy.linspace(min0 + (0.5 * delta), max0 - (0.5 * delta), bins)

    return out_pos, out_merge, out_data, out_count


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def histogram2d(numpy.ndarray pos0 not None,
                numpy.ndarray pos1 not None,
                bins not None,
                numpy.ndarray weights not None,
                split=False,
                nthread=None,
                double empty=0.0,
                double normalization_factor=1.0):
    """
    Calculate 2D histogram of pos0,pos1 weighted by weights

    @param pos0: 2Theta array
    @param pos1: Chi array
    @param weights: array with intensities
    @param bins: number of output bins int or 2-tuple of int
    @param split: pixel splitting is disabled in histogram
    @param nthread: maximum number of thread to use. By default: maximum available.
    @param empty: value given to empty bins
    @param normalization_factor: divide the result by this value

    @return  I, edges0, edges1, weighted histogram(2D), unweighted histogram (2D)

    One can also limit this with OMP_NUM_THREADS environment variable
    """
    assert pos0.size == pos1.size
    assert pos0.size == weights.size
    cdef:
        int  bins0, bins1, i, j, bin0, bin1
        int  size = pos0.size
    try:
        bins0, bins1 = tuple(bins)
    except:
        bins0 = bins1 = int(bins)
    if bins0 <= 0:
        bins0 = 1
    if bins1 <= 0:
        bins1 = 1
    cdef:
        float[:] cpos0 = numpy.ascontiguousarray(pos0.ravel(), dtype=numpy.float32)
        float[:] cpos1 = numpy.ascontiguousarray(pos1.ravel(), dtype=numpy.float32)
        float[:] data = numpy.ascontiguousarray(weights.ravel(), dtype=numpy.float32)
        numpy.ndarray[numpy.float64_t, ndim = 2] out_data = numpy.zeros((bins0, bins1), dtype="float64")
        numpy.ndarray[numpy.float64_t, ndim = 2] out_count = numpy.zeros((bins0, bins1), dtype="float64")
        numpy.ndarray[numpy.float64_t, ndim = 2] out_merge = numpy.zeros((bins0, bins1), dtype="float64")
        double min0 = pos0.min()
        double max0 = pos0.max() * EPS32
        double min1 = pos1.min()
        double max1 = pos1.max() * EPS32
        double delta0 = (max0 - min0) / float(bins0)
        double delta1 = (max1 - min1) / float(bins1)
        double fbin0, fbin1, p0, p1, d
        double epsilon = 1e-10

    if split:
        logger.warning("No pixel splitting in histogram")

    edges0 = numpy.linspace(min0 + (0.5 * delta0), max0 - (0.5 * delta0), bins0)
    edges1 = numpy.linspace(min1 + (0.5 * delta1), max1 - (0.5 * delta1), bins1)
    if nthread is not None:
        if isinstance(nthread, int) and (nthread > 0):
            omp_set_num_threads(< int > nthread)
    with nogil:
        for i in range(size):
            p0 = cpos0[i]
            p1 = cpos1[i]
            d = data[i]
            fbin0 = get_bin_number(p0, min0, delta0)
            fbin1 = get_bin_number(p1, min1, delta1)
            bin0 = < int > floor(fbin0)
            bin1 = < int > floor(fbin1)
            if (bin0<0) or (bin1<0) or (bin0>=bins0) or (bin1>=bins1):
                continue
            out_count[bin0, bin1] += 1.0
            out_data[bin0, bin1] += d

        for i in prange(bins0):
            for j in range(bins1):
                if out_count[i, j] > epsilon:
                    out_merge[i, j] += out_data[i, j] / out_count[i, j] / normalization_factor
                else:
                    out_merge[i, j] += empty

    return out_merge, edges0, edges1, out_data, out_count
