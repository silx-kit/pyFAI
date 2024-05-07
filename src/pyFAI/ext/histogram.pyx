# coding: utf-8
#cython: embedsignature=True, language_level=3, binding=True
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False,
## This is for developping
## cython: profile=True, warn.undeclared=True, warn.unused=True, warn.unused_result=False, warn.unused_arg=True
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#
#    Copyright (C) 2012-2024 European Synchrotron Radiation Facility, Grenoble, France
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

"""A set of histogram functions, some of which with OpenMP enabled.

Re-implementation of the numpy.histogram, optimized for azimuthal integration.

Can be replaced by ``silx.math.histogramnd``.
"""

__author__ = "Jérôme Kieffer"
__date__ = "25/04/2024"
__license__ = "MIT"
__copyright__ = "2011-2022, ESRF"
__contact__ = "jerome.kieffer@esrf.fr"


include "regrid_common.pxi"

import cython
from cython.parallel cimport prange
from libc.math cimport floor, sqrt
cimport pyFAI.ext._openmp as _openmp

import logging
logger = logging.getLogger(__name__)

from .preproc import preproc
from .splitBBox_common import calc_boundaries

_COMPILED_WITH_OPENMP = _openmp.COMPILED_WITH_OPENMP


def _histogram_omp(pos,
                   weights,
                   int bins=100,
                   bin_range=None,
                   pixelSize_in_Pos=None,
                   int nthread=0,
                   double empty=0.0,
                   double normalization_factor=1.0):
    """
    Calculates histogram of pos weighted by weights
    Multi threaded implementation

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
        int size = pos.size
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
        nthread = _openmp.omp_get_max_threads()
    _openmp.omp_set_num_threads(<int> nthread)
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
            thread = _openmp.omp_get_thread_num()
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


def _histogram_nomp(pos,
                    weights,
                    int bins=100,
                    bin_range=None,
                    pixelSize_in_Pos=None,
                    nthread=None,
                    double empty=0.0,
                    double normalization_factor=1.0):
    """
    Calculates histogram of pos weighted by weights,
    Single threaded implementation

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
        int bin = 0, i, idx
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

    with nogil:
        for i in range(size):
            a = cpos[i]
            fbin = get_bin_number(a, min0, delta)
            bin = < int > fbin
            if bin < 0 or bin >= bins:
                continue
            d = cdata[i]
            out_count[bin] += 1.0
            out_data[bin] += d

        for idx in range(bins):
            if out_count[idx] > epsilon:
                out_merge[idx] = out_data[idx] / out_count[idx] / normalization_factor
            else:
                out_merge[idx] = empty

    out_pos = numpy.linspace(min0 + (0.5 * delta), max0 - (0.5 * delta), bins)

    return (out_pos,
            numpy.asarray(out_merge),
            numpy.asarray(out_data),
            numpy.asarray(out_count))

if _COMPILED_WITH_OPENMP:
    histogram = _histogram_omp
else:
    histogram = _histogram_nomp


def histogram2d(pos0,
                pos1,
                bins,
                weights,
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


def histogram_preproc(pos,
                      weights,
                      int bins=100,
                      bin_range=None,
                      int error_model=0):
    """
    Calculates histogram of pos weighted by weights
    in the case data have been preprocessed, i.e. each datapoint contains
    (signal, normalization), (signal, variance, normalization), (signal, variance, normalization, count)

    :param pos: radial array
    :param weights: array with intensities, variance, normalization and count
    :param bins: number of output bins
    :param bin_range: 2-tuple with lower and upper bound for the valid position range.
    :param error_model: 0:no error propagation, 1:variance 2:poisson 3: azimuthal
    :return: 5 histograms concatenated, radial position (bin center)
    """
    cdef int nchan, ndim
    assert bins > 1
    ndim = weights.ndim
    nchan = weights.shape[ndim - 1]
    assert pos.size == weights.size // nchan
    cdef:
        Py_ssize_t  size = pos.size, bin = 0, i, j
        position_t[::1] cpos = numpy.ascontiguousarray(pos.ravel(), dtype=position_d)
        data_t[:, ::1] cdata = numpy.ascontiguousarray(weights, dtype=data_d).reshape(-1, nchan)
        acc_t[:, ::1] out_prop = numpy.zeros((bins, 5), dtype=acc_d)
        acc_t sum_sig, sum_var, sum_norm, sum_norm_sq, sum_count, omega_A, omega_B, omega_AB, omega2_A, omega2_B
        acc_t signal, variance, nrm, cnt, b, delta1, delta2
        position_t delta, min0, max0, maxin0
        position_t a = 0.0
        position_t fbin = 0.0
        position_t epsilon = 1e-10
        data_t tmp

    if bin_range is not None:
        min0 = min(bin_range)
        maxin0 = max(bin_range)
    else:
        with nogil:
            maxin0 = min0 = cpos[0]
            for i in range(1, size):
                a = cpos[i]
                maxin0 = max(maxin0, a)
                min0 = min(min0, a)

    with nogil:
        max0 = calc_upper_bound(maxin0)
        delta = (max0 - min0) / float(bins)
        for i in range(size):
            a = cpos[i]
            fbin = get_bin_number(a, min0, delta)
            bin = < Py_ssize_t > fbin
            if bin < 0 or bin >= bins:
                continue
            if error_model == 0:
                out_prop[bin, 0] += cdata[i, 0]
                out_prop[bin, 1] += cdata[i, 1]
                if nchan>2:
                    tmp = cdata[i, 2]
                    out_prop[bin, 2] += tmp
                    out_prop[bin, 4] += tmp*tmp
                if nchan>3:
                    out_prop[bin, 3] += cdata[i, 3]
            else:
                sum_sig = out_prop[bin, 0]
                sum_var = out_prop[bin, 1]
                sum_norm = out_prop[bin, 2]
                sum_count = out_prop[bin, 3]
                sum_norm_sq = out_prop[bin, 4]

                signal = cdata[i, 0]
                variance = cdata[i, 1]
                nrm = cdata[i, 2]
                cnt = cdata[i, 3]
                if error_model==3:
                    "Azimuthal error model"
                    if sum_norm_sq > 0.0:
                        # Inspired from https://dbs.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf
                        # Not correct, Inspired by VV_{A+b} = VV_A + ω²·(b-V_A/Ω_A)·(b-V_{A+b}/Ω_{A+b})
                        # Emprically validated against 2-pass implementation in Python/scipy-sparse
                        if nrm:
                            omega_A = sum_norm
                            omega_B = nrm
                            omega2_A = sum_norm_sq
                            omega2_B = omega_B*omega_B
                            sum_norm = omega_AB = omega_A + omega_B
                            sum_norm_sq = omega2_A + omega2_B

                            # VV_{AUb} = VV_A + ω_b^2 * (b-<A>) * (b-<AUb>)
                            b = signal / nrm
                            delta1 = sum_sig/omega_A - b
                            sum_sig += signal
                            delta2 = sum_sig / omega_AB - b
                            sum_var += omega2_B * delta1 * delta2
                    else:
                        sum_sig = signal
                        sum_norm = nrm
                        sum_norm_sq = nrm*nrm
                else:
                    sum_sig += signal
                    sum_norm += nrm
                    sum_norm_sq += nrm * nrm
                    sum_var += variance
                sum_count += cnt
                out_prop[bin, 0] = sum_sig
                out_prop[bin, 1] = sum_var
                out_prop[bin, 2] = sum_norm
                out_prop[bin, 3] = sum_count
                out_prop[bin, 4] = sum_norm_sq

    return (numpy.asarray(out_prop),
            numpy.linspace(min0 + (0.5 * delta), max0 - (0.5 * delta), bins))


def histogram1d_engine(radial, int npt,
                       raw,
                       dark=None,
                       flat=None,
                       solidangle=None,
                       polarization=None,
                       absorption=None,
                       mask=None,
                       dummy=None,
                       delta_dummy=None,
                       normalization_factor=1.0,
                       data_t empty=0.0,
                       split_result=False,
                       variance=None,
                       dark_variance=None,
                       error_model=ErrorModel.NO,
                       bint weighted_average=True,
                       radial_range=None
                       ):
    """Implementation of rebinning engine (without splitting) using pure cython histograms

    :param radial: radial position 2D array (same shape as raw)
    :param npt: number of points to integrate over
    :param raw: 2D array with the raw signal
    :param dark: array containing the value of the dark noise, to be subtracted
    :param flat: Array containing the flatfield image. It is also checked for dummies if relevant.
    :param solidangle: the value of the solid_angle. This processing may be performed during the rebinning instead. left for compatibility
    :param polarization: Correction for polarization of the incident beam
    :param absorption: Correction for absorption in the sensor volume
    :param mask: 2d array of int/bool: non-null where data should be ignored
    :param dummy: value of invalid data
    :param delta_dummy: precision for invalid data
    :param normalization_factor: final value is divided by this
    :param empty: value to be given for empty bins
    :param variance: provide an estimation of the variance
    :param dark_variance: provide an estimation of the variance of the dark_current,
    :param error_model: One of the several ErrorModel, only variance and Poisson are implemented.
    :param bool weighted_average: set to False to use an unweighted mean (similar to legacy) instead of the weighted average.


    NaN are always considered as invalid values

    if neither empty nor dummy is provided, empty pixels are left at 0.

    Nota: "azimuthal_range" has to be integrated into the
           mask prior to the call of this function

    :return: Integrate1dtpl named tuple containing:
            position, average intensity, std on intensity,
            plus the various histograms on signal, variance, normalization and count.

    """
    cdef:
        acc_t[:, ::1] res
        float32_t[:, ::1] prep
        position_t[::1] position
        data_t[::1] histo_normalization, histo_signal, histo_variance, histo_count, intensity, std, sem, histo_normalization2
        data_t norm, sig, var, cnt, norm2
        int i
        bint do_variance = error_model
        bint do_azimuthal_variance = error_model is ErrorModel.AZIMUTHAL

    prep = preproc(raw,
                   dark=dark,
                   flat=flat,
                   solidangle=solidangle,
                   polarization=polarization,
                   absorption=absorption,
                   mask=mask,
                   dummy=dummy,
                   delta_dummy=delta_dummy,
                   normalization_factor=normalization_factor,
                   split_result=4,
                   variance=variance,
                   dark_variance=dark_variance,
                   error_model=error_model,
                   apply_normalization=not weighted_average,
                   ).reshape(-1, 4)
    res, position = histogram_preproc(radial.ravel(),
                                      prep,
                                      npt,
                                      bin_range=radial_range,
                                      error_model = <int> error_model)

    histo_signal = numpy.empty(npt, dtype=data_d)
    histo_variance = numpy.empty(npt, dtype=data_d)
    histo_normalization = numpy.empty(npt, dtype=data_d)
    histo_normalization2 = numpy.empty(npt, dtype=data_d)
    histo_count = numpy.empty(npt, dtype=data_d)
    intensity = numpy.empty(npt, dtype=data_d)
    std = numpy.empty(npt, dtype=data_d)
    sem = numpy.empty(npt, dtype=data_d)
    if dummy is not None:
        empty = dummy
    with nogil:
        for i in range(npt):
            sig = histo_signal[i] = res[i, 0]
            var = histo_variance[i] = res[i, 1]
            norm = histo_normalization[i] = res[i, 2]
            cnt = histo_count[i] = res[i, 3]
            norm2 = histo_normalization2[i] = res[i, 4]
            if norm2 > 0.0:
                intensity[i] = sig / norm
                if do_variance:
                    std[i] = sqrt(var / norm2)
                    sem[i] = sqrt(var) / norm
                else:
                    std[i] = empty
                    sem[i] = empty
            else:
                intensity[i] = empty
                std[i] = empty
                sem[i] = empty
    return Integrate1dtpl(numpy.asarray(position),
                          numpy.asarray(intensity),
                          numpy.asarray(sem),
                          numpy.asarray(histo_signal),
                          numpy.asarray(histo_variance),
                          numpy.asarray(histo_normalization),
                          numpy.asarray(histo_count),
                          numpy.asarray(std),
                          numpy.asarray(sem),
                          numpy.asarray(histo_normalization2)
                          )


def histogram2d_engine(radial, azimuthal,
                       bins, #2-tuple
                       raw,
                       dark=None,
                       flat=None,
                       solidangle=None,
                       polarization=None,
                       absorption=None,
                       mask=None,
                       dummy=None,
                       delta_dummy=None,
                       double normalization_factor=1.0,
                       data_t empty=0.0,
                       variance=None,
                       dark_variance=None,
                       int error_model=ErrorModel.NO,
                       bint weighted_average=True,
                       radial_range=None,
                       azimuth_range=None,
                       bint allow_radial_neg=False,
                       bint chiDiscAtPi=1,
                       bint clip_pos1=True
                       ):
    """Implementation of 2D rebinning engine using pure numpy histograms

    :param radial: radial position 2D array (same shape as raw)
    :param azimuthal: azimuthal position 2D array (same shape as raw)
    :param bins: number of points to integrate over in (radial, azimuthal) dimensions
    :param raw: 2D array with the raw signal
    :param dark: array containing the value of the dark noise, to be subtracted
    :param flat: Array containing the flatfield image. It is also checked for dummies if relevant.
    :param solidangle: the value of the solid_angle. This processing may be performed during the rebinning instead. left for compatibility
    :param polarization: Correction for polarization of the incident beam
    :param absorption: Correction for absorption in the sensor volume
    :param mask: 2d array of int/bool: non-null where data should be ignored
    :param dummy: value of invalid data
    :param delta_dummy: precision for invalid data
    :param normalization_factor: final value is divided by this
    :param empty: value to be given for empty bins
    :param variance: provide an estimation of the variance
    :param dark_variance: provide an estimation of the variance of the dark_current,
    :param error_model: set to "poisson" for assuming the detector is poissonian and variance = raw + dark
    :param bool weighted_average: set to False to use an unweighted mean (similar to legacy) instead of the weighted average.
    :param radial_range: enforce boundaries in radial dimention, 2tuple with lower and upper bound
    :param azimuth_range: enforce boundaries in azimuthal dimention, 2tuple with lower and upper bound
    :param allow_radial_neg: clip negative radial position (can a dimention be negative ?)
    :param chiDiscAtPi: set the azimuthal discontinuity at π (True) or at 0/2π (False)
    :param clip_pos1: clip the azimuthal range to [-π π] (or [0 2π]), set to False to deactivate behavior.

    NaN are always considered as invalid values

    if neither empty nor dummy is provided, empty pixels are left at 0.

    Nota: "azimuthal_range" has to be integrated into the
           mask prior to the call of this function

    :return: Integrate1dtpl named tuple containing:
            position, average intensity, std on intensity,
            plus the various histograms on signal, variance, normalization and count.

    :return: Integrate2dtpl namedtuple: "radial azimuthal intensity error signal variance normalization count"
    """
    cdef:
        Py_ssize_t bins0, bins1, i, j, bin0, bin1, idx
        Py_ssize_t size = raw.size
    assert size == radial.size, "radial has the same size as raw"
    assert size == azimuthal.size, "azimuthal has the same size as raw"

    try:
        bins0, bins1 = tuple(bins)
    except TypeError:
        bins0 = bins1 = int(bins)
    bins0 = max(bins0, 1)
    bins1 = max(bins1, 1)

    cdef:
        # Related to data: single precision
        data_t[::1] cdata = numpy.ascontiguousarray(raw.ravel(), dtype=data_d)
        data_t[::1] cflat, cdark, cpolarization, csolidangle, cvariance, cabsorption, cdark_variance
        data_t cdummy, ddummy=0.0
        # Related to positon: double precision
        position_t[::1] cpos0 = numpy.ascontiguousarray(radial.ravel(), dtype=position_d)
        position_t[::1] cpos1 = numpy.ascontiguousarray(azimuthal.ravel(), dtype=position_d)
        #Accumulated data are also double
        acc_t[:, :, ::1] out_data = numpy.zeros((bins0, bins1, 5), dtype=acc_d)
        data_t[:, ::1] out_intensity = numpy.zeros((bins0, bins1), dtype=data_d)
        data_t[:, ::1] out_std = numpy.zeros((bins0, bins1), dtype=data_d)
        data_t[:, ::1] out_sem = numpy.zeros((bins0, bins1), dtype=data_d)
        mask_t[::1] cmask
        acc_t sig, var, norm, cnt, norm2
        position_t c0, c1
        position_t pos0_min, pos0_maxin, pos0_max,  pos1_min, pos1_maxin, pos1_max, delta0, delta1
        position_t fbin0, fbin1
        bint check_mask = False, check_dummy = False, is_valid
        bint do_dark = False, do_flat = False, do_polarization = False, do_solidangle = False, do_absorption=False
        bint do_variance=error_model, do_dark_variance=False
        preproc_t value

    if variance is not None:
        assert variance.size == size, "variance size matches"
        do_variance = True
        cvariance = numpy.ascontiguousarray(variance.ravel(), dtype=data_d)

    if mask is not None:
        assert mask.size == size, "mask size"
        check_mask = True
        cmask = numpy.ascontiguousarray(mask.ravel(), dtype=mask_d)
    else:
        cmask = None

    if (dummy is not None) and (delta_dummy is not None):
        check_dummy = True
        cdummy = float(dummy)
        ddummy = float(delta_dummy)
    elif (dummy is not None):
        cdummy = float(dummy)
        ddummy = 0.0
        check_dummy = True
    else:
        cdummy = float(empty)
        ddummy = 0.0
        check_dummy = False

    if dark is not None:
        assert dark.size == size, "dark current array size"
        do_dark = True
        cdark = numpy.ascontiguousarray(dark.ravel(), dtype=data_d)
    if dark_variance is not None:
        assert dark_variance.size == size, "dark_varance array size"
        do_dark_variance = True
        cdark_variance = numpy.ascontiguousarray(dark_variance.ravel(), dtype=data_d)
    if flat is not None:
        assert flat.size == size, "flat-field array size"
        do_flat = True
        cflat = numpy.ascontiguousarray(flat.ravel(), dtype=data_d)
    if polarization is not None:
        do_polarization = True
        assert polarization.size == size, "polarization array size"
        cpolarization = numpy.ascontiguousarray(polarization.ravel(), dtype=data_d)
    if solidangle is not None:
        do_solidangle = True
        assert solidangle.size == size, "Solid angle array size"
        csolidangle = numpy.ascontiguousarray(solidangle.ravel(), dtype=data_d)
    if absorption is not None:
        do_absorption = True
        assert absorption.size == size, "Absorption array size"
        cabsorption = numpy.ascontiguousarray(absorption.ravel(), dtype=data_d)

    pos0_min, pos0_maxin, pos1_min, pos1_maxin = calc_boundaries(cpos0, None,
                                                                 cpos1, None,
                                                                 cmask, radial_range, azimuth_range,
                                                                 allow_radial_neg, chiDiscAtPi, clip_pos1)

    pos0_max = calc_upper_bound(pos0_maxin)
    pos1_max = calc_upper_bound(pos1_maxin)

    delta0 = (pos0_max - pos0_min) / (<position_t> bins0)
    delta1 = (pos1_max - pos1_min) / (<position_t> bins1)

    with nogil:
        for idx in range(size):
            if (check_mask) and cmask[idx]:
                continue

            is_valid = preproc_value_inplace(&value,
                                             cdata[idx],
                                             variance=cvariance[idx] if error_model==1 else 0.0,
                                             dark=cdark[idx] if do_dark else 0.0,
                                             flat=cflat[idx] if do_flat else 1.0,
                                             solidangle=csolidangle[idx] if do_solidangle else 1.0,
                                             polarization=cpolarization[idx] if do_polarization else 1.0,
                                             absorption=cabsorption[idx] if do_absorption else 1.0,
                                             mask=0, #previously checked
                                             dummy=cdummy,
                                             delta_dummy=ddummy,
                                             check_dummy=check_dummy,
                                             normalization_factor=normalization_factor,
                                             dark_variance=cdark_variance[idx] if do_dark_variance else 0.0,
                                             error_model=error_model,
                                             apply_normalization = not weighted_average,)

            if not is_valid:
                continue

            c0 = cpos0[idx]
            c1 = cpos1[idx]

            fbin0 = get_bin_number(c0, pos0_min, delta0)
            fbin1 = get_bin_number(c1, pos1_min, delta1)

            bin0 = <Py_ssize_t> fbin0
            bin1 = <Py_ssize_t> fbin1

            if (bin0 < 0) or (bin0 >= bins0) or (bin1 < 0) or (bin1 >= bins1):
                    continue

            update_2d_accumulator(out_data, bin0, bin1, value, 1.0)

        for i in range(bins0):
            for j in range(bins1):
                sig = out_data[i, j, 0]
                var = out_data[i, j, 1]
                norm = out_data[i, j, 2]
                cnt = out_data[i, j, 3]
                norm2 = out_data[i, j, 4]
                if cnt > 0.0:
                    "test on count as norm could be negatve"
                    out_intensity[i, j] = sig / norm
                    if do_variance:
                        out_sem[i, j] = sqrt(var) / norm
                        out_std[i, j] = sqrt(var / norm2)
                else:
                    out_intensity[i, j] = empty
                    if do_variance:
                        out_sem[i, j] = empty
                        out_std[i, j] = empty

    bin_centers0 = numpy.linspace(pos0_min + 0.5 * delta0, pos0_max - 0.5 * delta0, bins0)
    bin_centers1 = numpy.linspace(pos1_min + 0.5 * delta1, pos1_max - 0.5 * delta1, bins1)
    return Integrate2dtpl(bin_centers0, bin_centers1,
                          numpy.asarray(out_intensity).T,
                          numpy.asarray(out_sem).T if do_variance else None,
                          numpy.asarray(out_data[...,0]).T, numpy.asarray(out_data[...,1]).T,
                          numpy.asarray(out_data[...,2]).T, numpy.asarray(out_data[...,3]).T,
                          numpy.asarray(out_std).T, numpy.asarray(out_sem).T,
                          numpy.asarray(out_data[...,4]).T)

histogram2d_engine
