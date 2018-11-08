# coding: utf-8
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#
#    Copyright (C) 2012-2018 European Synchrotron Radiation Facility, Grenoble, France
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

"""Re-implementation of numpy histograms using OpenMP"""

__author__ = "Jerome Kieffer"
__date__ = "08/11/2018"
__license__ = "MIT"
__copyright__ = "2011-2018, ESRF"
__contact__ = "jerome.kieffer@esrf.fr"

import cython
from cython.parallel cimport prange
from libc.math cimport floor, sqrt
from openmp cimport omp_set_num_threads, omp_get_max_threads, omp_get_thread_num
import logging
logger = logging.getLogger(__name__ + "_omp")


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def histogram(pos not None,
              weights not None,
              int bins=100,
              bin_range=None,
              pixelSize_in_Pos=None,
              int nthread=0,
              double empty=0.0,
              double normalization_factor=1.0):
    """
    Calculates histogram of pos weighted by weights

    :param pos: 2Theta array
    :param weights: array with intensities
    :param bins: number of output bins
    :param pixelSize_in_Pos: size of a pixels in 2theta: DESACTIVATED
    :param nthread: OpenMP is disabled. unused
    :param empty: value given to empty bins
    :param normalization_factor: divide the result by this value

    :return: 2theta, I, weighted histogram, raw histogram
    """

    assert pos.size == weights.size
    assert bins > 1
    cdef:
        int  size = pos.size
        position_t[::1] cpos = numpy.ascontiguousarray(pos.ravel(), dtype=position_d)
        data_t[::1] cdata = numpy.ascontiguousarray(weights.ravel(), dtype=data_d)
        acc_t[::1] out_data = numpy.zeros(bins, dtype=acc_d)
        acc_t[::1] out_count = numpy.zeros(bins, dtype=acc_d)
        data_t[::1] out_merge = numpy.zeros(bins, dtype=data_d)
        position_t delta, min0, max0, maxin0
        position_t a = 0.0
        position_t d = 0.0
        position_t fbin = 0.0
        position_t tmp_count, tmp_data = 0.0
        position_t epsilon = 1e-10
        int bin = 0, i, idx, thread
    if pixelSize_in_Pos:
        logger.warning("No pixel splitting in histogram")

    if bin_range is not None:
        min0 = min(bin_range)
        maxin0 = max(bin_range)
    else:
        min0 = pos.min()
        maxin0 = pos.max()
    max0 = calc_upper_bound(maxin0)
    delta = (max0 - min0) / float(bins)

    if not nthread:
        nthread = omp_get_max_threads()
    omp_set_num_threads(<int> nthread)
    cdef:
        acc_t[:, ::1] big_count = numpy.zeros((nthread, bins), dtype=acc_d)
        acc_t[:, ::1] big_data = numpy.zeros((nthread, bins), dtype=acc_d)

    if pixelSize_in_Pos:
        logger.warning("No pixel splitting in histogram")

    with nogil:
        for i in prange(size):
            d = cdata[i]
            a = cpos[i]
            fbin = get_bin_number(a, min0, delta)
            bin = < int > fbin
            if bin < 0 or bin >= bins:
                continue
            thread = omp_get_thread_num()
            big_count[thread, bin] += 1.0
            big_data[thread, bin] += d

        for idx in prange(bins):
            tmp_count = 0.0
            tmp_data = 0.0
            for thread in range(nthread):
                tmp_count = tmp_count + big_count[thread, idx]
                tmp_data = tmp_data + big_data[thread, idx]
            out_count[idx] += tmp_count
            out_data[idx] += tmp_data
            if out_count[idx] > epsilon:
                out_merge[idx] += tmp_data / tmp_count / normalization_factor
            else:
                out_merge[idx] += empty

    out_pos = numpy.linspace(min0 + (0.5 * delta), max0 - (0.5 * delta), bins)

    return (out_pos, 
            numpy.asarray(out_merge), 
            numpy.asarray(out_data), 
            numpy.asarray(out_count))


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def histogram2d(pos0 not None,
                pos1 not None,
                bins not None,
                weights not None,
                split=False,
                nthread=None,
                data_t empty=0.0,
                data_t normalization_factor=1.0):
    """
    Calculate 2D histogram of pos0,pos1 weighted by weights

    :param pos0: 2Theta array
    :param pos1: Chi array
    :param weights: array with intensities
    :param bins: number of output bins int or 2-tuple of int
    :param split: pixel splitting is disabled in histogram
    :param nthread: Unused ! see below
    :param empty: value given to empty bins
    :param normalization_factor: divide the result by this value

    :return: I, bin_centers0, bin_centers1, weighted histogram(2D), unweighted histogram (2D)

    Nota: the histogram itself is not parallelized as it is slower than in serial mode 
    (cache contention)
    """
    assert pos0.size == pos1.size
    assert pos0.size == weights.size
    cdef:
        int  bins0, bins1, i, j, bin0, bin1
        int  size = pos0.size
    try:
        bins0, bins1 = tuple(bins)
    except TypeError:
        bins0 = bins1 = int(bins)
    if bins0 <= 0:
        bins0 = 1
    if bins1 <= 0:
        bins1 = 1
    cdef:
        position_t[::1] cpos0 = numpy.ascontiguousarray(pos0.ravel(), dtype=position_d)
        position_t[::1] cpos1 = numpy.ascontiguousarray(pos1.ravel(), dtype=position_d)
        data_t[::1] data = numpy.ascontiguousarray(weights.ravel(), dtype=data_d)
        acc_t[:, ::1] out_data = numpy.zeros((bins0, bins1), dtype=acc_d)
        acc_t[:, ::1] out_count = numpy.zeros((bins0, bins1), dtype=acc_d)
        data_t[:, ::1] out_merge = numpy.zeros((bins0, bins1), dtype=data_d)
        position_t min0 = pos0.min()
        position_t max0 = calc_upper_bound(<position_t> pos0.max())
        position_t min1 = pos1.min()
        position_t max1 = calc_upper_bound(<position_t> pos1.max())
        position_t delta0 = (max0 - min0) / float(bins0)
        position_t delta1 = (max1 - min1) / float(bins1)
        position_t fbin0, fbin1, p0, p1, d
        position_t epsilon = 1e-10

    if split:
        logger.warning("No pixel splitting in histogram")

    bin_centers0 = numpy.linspace(min0 + (0.5 * delta0), max0 - (0.5 * delta0), bins0)
    bin_centers1 = numpy.linspace(min1 + (0.5 * delta1), max1 - (0.5 * delta1), bins1)
    with nogil:
        for i in range(size):
            p0 = cpos0[i]
            p1 = cpos1[i]
            
            fbin0 = get_bin_number(p0, min0, delta0)
            fbin1 = get_bin_number(p1, min1, delta1)
            bin0 = < int > floor(fbin0)
            bin1 = < int > floor(fbin1)
            if (bin0 < 0) or (bin1 < 0) or (bin0 >= bins0) or (bin1 >= bins1):
                continue
            d = data[i]
            out_count[bin0, bin1] += 1.0
            out_data[bin0, bin1] += d

        for i in prange(bins0):
            for j in range(bins1):
                if out_count[i, j] > epsilon:
                    out_merge[i, j] += out_data[i, j] / out_count[i, j] / normalization_factor
                else:
                    out_merge[i, j] += empty

    return (numpy.asarray(out_merge), 
            bin_centers0, bin_centers1, 
            numpy.asarray(out_data), 
            numpy.asarray(out_count))


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def histogram2d_preproc(numpy.ndarray pos0 not None,
                        numpy.ndarray pos1 not None,
                        bins not None,
                        numpy.ndarray weights not None,
                        split=False,
                        int nthread=0,
                        double empty=0.0,
                        ):
    """
    Calculate 2D histogram of pos0,pos1 weighted by weights when the weights are 
    preprocessed and contain: signal, variance, normalization for each pixel

    :param pos0: 2Theta array
    :param pos1: Chi array
    :param weights: array with intensities
    :param bins: number of output bins int or 2-tuple of int
    :param split: pixel splitting is disabled in histogram
    :param nthread: number of thread to use 
    :param empty: value given to empty bins

    :return: named tuple with ("sig",["var"], "norm", "count") 
    """
    assert pos0.size == pos1.size, "Positions array have the same size"
    
    cdef:
        int bins0, bins1, i, j, bin0, bin1, c, thread
        int size = pos0.size
        int nchan = weights.shape[weights.ndim - 1]
    assert weights.ndim > 1, "Weights have been preprocessed" 
    assert pos0.size == (weights.size // nchan), "Weights have the right size"
    assert nchan <= 4, "Maximum of 4 chanels"
    
    try:
        bins0, bins1 = tuple(bins)
    except TypeError:
        bins0 = bins1 = int(bins)
    if bins0 <= 0:
        bins0 = 1
    if bins1 <= 0:
        bins1 = 1

    if not nthread:
        nthread = omp_get_max_threads()
    omp_set_num_threads(<int> nthread)
    
    cdef:
        position_t[::1] cpos0 = numpy.ascontiguousarray(pos0.ravel(), dtype=position_d)
        position_t[::1] cpos1 = numpy.ascontiguousarray(pos1.ravel(), dtype=position_d)
        data_t[:, ::1] data = numpy.ascontiguousarray(weights.reshape((-1, nchan)), dtype=data_d)
        acc_t[:, :, ::1] out_data = numpy.zeros((bins0, bins1, 4), dtype=acc_d)
        data_t[:, ::1] out_signal = numpy.zeros((bins0, bins1), dtype=data_d)
        data_t[:, ::1] out_error
        position_t min0 = pos0.min()
        position_t max0 = calc_upper_bound(<position_t> pos0.max())
        position_t min1 = pos1.min()
        position_t max1 = calc_upper_bound(<position_t> pos1.max())
        position_t delta0 = (max0 - min0) / float(bins0)
        position_t delta1 = (max1 - min1) / float(bins1)
        position_t fbin0, fbin1, p0, p1
        acc_t epsilon = 1e-10
        acc_t count, sig, var, norm 
        acc_t[:, :, :, ::1] big = numpy.zeros((nthread, bins0, bins1, 4), dtype=acc_d)
    
    if nchan == 3:
        out_error = numpy.zeros((bins0, bins1), dtype=data_d)
    if split:
        logger.warning("No pixel splitting in histogram")

    bin_centers0 = numpy.linspace(min0 + (0.5 * delta0), max0 - (0.5 * delta0), bins0)
    bin_centers1 = numpy.linspace(min1 + (0.5 * delta1), max1 - (0.5 * delta1), bins1)

    with nogil:
        for i in prange(size):
            p0 = cpos0[i]
            p1 = cpos1[i]
            fbin0 = get_bin_number(p0, min0, delta0)
            fbin1 = get_bin_number(p1, min1, delta1)
            bin0 = < int > floor(fbin0)
            bin1 = < int > floor(fbin1)
            if (bin0 < 0) or (bin1 < 0) or (bin0 >= bins0) or (bin1 >= bins1):
                continue
            thread = omp_get_thread_num()
            for c in range(nchan):
                big[thread, bin0, bin1, c] += data[i, c]
            if nchan < 4:
                big[thread, bin0, bin1, 3] += 1.0
        for i in prange(bins0):
            for j in range(bins1):
                count = sig = var = norm = 0.0 
                for thread in range(nthread):
                    sig = sig + big[thread, i, j, 0] 
                    var = var + big[thread, i, j, 1]
                    norm = norm + big[thread, i, j, 2]
                    count = count + big[thread, i, j, 3]
                out_data[i, j, 0] = sig 
                out_data[i, j, 1] = var
                out_data[i, j, 2] = norm
                out_data[i, j, 3] = count

                if nchan >= 3:
                    if norm > 0:
                        out_signal[i, j] = sig / norm
                        out_error[i, j] = sqrt(var) / norm
                    else:
                        out_signal[i, j] = empty
                        out_error[i, j] = empty
                elif nchan == 2:
                    sig = out_data[i, j, 0] 
                    norm = out_data[i, j, 2]
                    if norm > 0:
                        out_signal[i, j] = sig / norm
                else:
                    out_signal[i, j] = sig
    if nchan >= 3:
        result = Integrate2dWithErrorResult(numpy.asarray(out_signal),
                                            numpy.asarray(out_error),
                                            bin_centers0, 
                                            bin_centers1,
                                            numpy.recarray(shape=(bins0, bins1), dtype=prop_d, buf=out_data))
    else:
        result = Integrate2dResult(numpy.asarray(out_signal),
                                   bin_centers0, 
                                   bin_centers1,
                                   numpy.recarray(shape=(bins0, bins1), dtype=prop_d, buf=out_data))
    return result
