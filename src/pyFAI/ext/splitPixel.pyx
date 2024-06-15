# -*- coding: utf-8 -*-
#cython: embedsignature=True, language_level=3, binding=True
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False,
## This is for developping
## cython: profile=True, warn.undeclared=True, warn.unused=True, warn.unused_result=False, warn.unused_arg=True
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2024 European Synchrotron Radiation Facility, France
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

"""Calculates histograms of pos0 (tth) weighted by Intensity

Splitting is done by full pixel splitting
Histogram (direct) implementation
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "15/06/2024"
__status__ = "stable"
__license__ = "MIT"

include "regrid_common.pxi"


cimport cython
import numpy
import logging
logger = logging.getLogger(__name__)

from .splitpixel_common import calc_boundaries


def fullSplit1D(pos,
                weights,
                Py_ssize_t bins=100,
                pos0_range=None,
                pos1_range=None,
                dummy=None,
                delta_dummy=None,
                mask=None,
                dark=None,
                flat=None,
                solidangle=None,
                polarization=None,
                float empty=0.0,
                double normalization_factor=1.0,
                Py_ssize_t coef_power=1,
                bint allow_pos0_neg=False,
                ):
    """
    Calculates histogram of pos weighted by weights

    Splitting is done on the pixel's bounding box like fit2D.
    No compromise for speed has been made here.


    :param pos: 3D array with pos0; Corner A,B,C,D; tth or chi
    :param weights: array with intensities
    :param bins: number of output bins
    :param pos0_range: minimum and maximum  of the 2th range
    :param pos1_range: minimum and maximum  of the chi range
    :param dummy: value for bins without pixels
    :param delta_dummy: precision of dummy value
    :param mask: array (of int8) with masked pixels with 1 (0=not masked)
    :param dark: array (of float64) with dark noise to be subtracted (or None)
    :param flat: array (of float64) with flat image
    :param polarization: array (of float64) with polarization correction
    :param solidangle: array (of float64) with flat image
    :param empty: value of output bins without any contribution when dummy is None
    :param normalization_factor: divide the valid result by this value
    :param coef_power: set to 2 for variance propagation, leave to 1 for mean calculation
    :param allow_pos0_neg: allow radial dimention to be negative (useful in log-scale!)
    :return: 2theta, I, weighted histogram, unweighted histogram
    """
    cdef Py_ssize_t  size = weights.size
    if pos.ndim > 3:  # create a view
        pos = pos.reshape((-1, 4, 2))
    assert pos.shape[0] == size, "pos.shape[0] == size"
    assert pos.shape[1] == 4, "pos.shape[1] == 4"
    assert pos.shape[2] == 2, "pos.shape[2] == 2"
    assert pos.ndim == 3, "pos.ndim == 3"
    assert bins > 1, "at lease one bin"
    cdef:
        position_t[:, :, ::1] cpos = numpy.ascontiguousarray(pos, dtype=position_d)
        data_t[::1] cdata = numpy.ascontiguousarray(weights.ravel(), dtype=data_d)
        acc_t[::1] sum_data = numpy.zeros(bins, dtype=acc_d)
        acc_t[::1] sum_count = numpy.zeros(bins, dtype=acc_d)
        data_t[::1] merged = numpy.zeros(bins, dtype=data_d)
        mask_t[::1] cmask = None
        data_t[::1] cflat, cdark, cpolarization, csolidangle
        buffer_t[::1] buffer = numpy.zeros(bins, dtype=buffer_d)

        data_t cdummy = 0, cddummy = 0, data = 0
        position_t inv_area = 0
        position_t pos0_min = 0, pos0_max = 0, pos0_maxin = 0, pos1_min = 0, pos1_max = 0, pos1_maxin = 0
        position_t area_pixel = 0, sum_area = 0, sub_area = 0, dpos = 0
        position_t a0 = 0, b0 = 0, c0 = 0, d0 = 0, max0 = 0, min0 = 0, a1 = 0, b1 = 0, c1 = 0, d1 = 0, max1 = 0, min1 = 0
        double epsilon = 1e-10

        bint check_pos1=False, check_mask=False, check_dummy=False, do_dark=False, do_flat=False, do_polarization=False, do_solidangle=False
        Py_ssize_t i = 0, idx = 0, bin0_max = 0, bin0_min = 0

    if mask is not None:
        check_mask = True
        assert mask.size == size, "mask size"
        cmask = numpy.ascontiguousarray(mask.ravel(), dtype=mask_d)

    pos0_min, pos0_maxin, pos1_min, pos1_maxin = calc_boundaries(cpos, cmask, pos0_range, pos1_range,
                                                                 allow_pos0_neg=allow_pos0_neg)
    if (not allow_pos0_neg):
        pos0_min = max(pos0_min, 0.0)
        pos1_maxin = max(pos1_maxin, 0.0)
    pos0_max = calc_upper_bound(pos0_maxin)
    pos1_max = calc_upper_bound(pos1_maxin)
    dpos = (pos0_max - pos0_min) / (<position_t> (bins))

    if (dummy is not None) and (delta_dummy is not None):
        check_dummy = True
        cdummy = float(dummy)
        cddummy = float(delta_dummy)
    elif (dummy is not None):
        check_dummy = True
        cdummy = float(dummy)
        cddummy = 0.0
    else:
        check_dummy = False
        cdummy = empty
        cddummy = 0.0

    if dark is not None:
        do_dark = True
        assert dark.size == size, "dark current array size"
        cdark = numpy.ascontiguousarray(dark.ravel(), dtype=data_d)
    if flat is not None:
        do_flat = True
        assert flat.size == size, "flat-field array size"
        cflat = numpy.ascontiguousarray(flat.ravel(), dtype=data_d)
    if polarization is not None:
        do_polarization = True
        assert polarization.size == size, "polarization array size"
        cpolarization = numpy.ascontiguousarray(polarization.ravel(), dtype=data_d)
    if solidangle is not None:
        do_solidangle = True
        assert solidangle.size == size, "Solid angle array size"
        csolidangle = numpy.ascontiguousarray(solidangle.ravel(), dtype=data_d)

    with nogil:
        for idx in range(size):

            if (check_mask) and (cmask[idx]):
                continue

            data = cdata[idx]
            if check_dummy and ((cddummy == 0.0 and data == cdummy) or (cddummy != 0.0 and fabs(data - cdummy) <= cddummy)):
                continue

            # a0, b0, c0 and d0 are in bin number (2theta, q or r)
            # a1, b1, c1 and d1 are in Chi angle in radians ...
            a0 = get_bin_number(cpos[idx, 0, 0], pos0_min, dpos)
            a1 = <  position_t > cpos[idx, 0, 1]
            b0 = get_bin_number(cpos[idx, 1, 0], pos0_min, dpos)
            b1 = <  position_t > cpos[idx, 1, 1]
            c0 = get_bin_number(cpos[idx, 2, 0], pos0_min, dpos)
            c1 = <  position_t > cpos[idx, 2, 1]
            d0 = get_bin_number(cpos[idx, 3, 0], pos0_min, dpos)
            d1 = <  position_t > cpos[idx, 3, 1]
            min0 = min(a0, b0, c0, d0)
            max0 = max(a0, b0, c0, d0)

            if (max0 < 0) or (min0 >= bins):
                continue
            if check_pos1:
                min1 = min(a1, b1, c1, d1)
                max1 = max(a1, b1, c1, d1)
                if (max1 < pos1_min) or (min1 > pos1_maxin):
                    continue

            if do_dark:
                data -= cdark[idx]
            if do_flat:
                data /= cflat[idx]
            if do_polarization:
                data /= cpolarization[idx]
            if do_solidangle:
                data /= csolidangle[idx]

            bin0_min = < Py_ssize_t > floor(min0)
            bin0_max = < Py_ssize_t > floor(max0)

            if bin0_min == bin0_max:
                # All pixel is within a single bin
                sum_count[bin0_min] += 1.0
                sum_data[bin0_min] += data

            # else we have pixel splitting.
            else:
                bin0_min = max(0, bin0_min)
                bin0_max = min(bins, bin0_max + 1)

                _integrate1d(buffer, a0, a1, b0, b1)  # A-B
                _integrate1d(buffer, b0, b1, c0, c1)  # B-C
                _integrate1d(buffer, c0, c1, d0, d1)  # C-D
                _integrate1d(buffer, d0, d1, a0, a1)  # D-A

                # Distribute pixel area
                sum_area = 0.0
                for i in range(bin0_min, bin0_max):
                    sum_area += buffer[i]
                inv_area = 1.0 / sum_area
                for i in range(bin0_min, bin0_max):
                    sub_area = buffer[i] * inv_area
                    sum_count[i] += sub_area
                    sum_data[i] += pown(sub_area, coef_power) * data

                buffer[bin0_min:bin0_max] = 0
        for i in range(bins):
            if sum_count[i] > epsilon:
                merged[i] = sum_data[i] / sum_count[i] / normalization_factor
            else:
                merged[i] = cdummy
    bin_centers = numpy.linspace(pos0_min + 0.5 * dpos,
                                 pos0_max - 0.5 * dpos,
                                 bins)

    return (bin_centers, numpy.asarray(merged), numpy.asarray(sum_data), numpy.asarray(sum_count))


def fullSplit1D_engine(pos not None,
                       weights not None,
                       Py_ssize_t bins=100,
                       pos0_range=None,
                       pos1_range=None,
                       dummy=None,
                       delta_dummy=None,
                       mask=None,
                       variance=None,
                       dark_variance=None,
                       int error_model=ErrorModel.NO,
                       dark=None,
                       flat=None,
                       solidangle=None,
                       polarization=None,
                       absorption=None,
                       data_t empty=0.0,
                       double normalization_factor=1.0,
                       bint weighted_average=True,
                       bint allow_pos0_neg=True,
                       bint chiDiscAtPi=True
                       ):
    """
    Calculates histogram of pos weighted by weights

    Splitting is done on the pixel's bounding box like fit2D.
    New implementation with variance propagation


    :param pos: 3D array with pos0; Corner A,B,C,D; tth or chi
    :param weights: array with intensities
    :param bins: number of output bins
    :param pos0_range: minimum and maximum  of the 2th range
    :param pos1_range: minimum and maximum  of the chi range
    :param dummy: value for bins without pixels
    :param delta_dummy: precision of dummy value
    :param mask: array (of int8) with masked pixels with 1 (0=not masked)
    :param dark: array (of float64) with dark noise to be subtracted (or None)
    :param flat: array (of float64) with flat image
    :param polarization: array (of float64) with polarization correction
    :param solidangle: array (of float64) with flat image
    :param absorption: array (of float64) with absorption correction
    :param empty: value of output bins without any contribution when dummy is None
    :param normalization_factor: divide the valid result by this value
    :param bool weighted_average: set to False to use an unweigted mean (similar to legacy) instead of the weigted average.
    :param allow_pos0_neg: allow radial dimention to be negative (useful in log-scale!)
    :param chiDiscAtPi: tell if azimuthal discontinuity is at 0° or 180°
    :return: namedtuple with "position intensity error signal variance normalization count"
    """
    cdef Py_ssize_t  size = weights.size
    if pos.ndim > 3:  # create a view
        pos = pos.reshape((-1, 4, 2))
    assert pos.shape[0] == size, "pos.shape[0] == size"
    assert pos.shape[1] == 4, "pos.shape[1] == 4"
    assert pos.shape[2] == 2, "pos.shape[2] == 2"
    assert pos.ndim == 3, "pos.ndim == 3"
    assert bins > 1, "at lease one bin"
    cdef:
        position_t[:, :, ::1] cpos = numpy.ascontiguousarray(pos, dtype=position_d)
        position_t[:, ::1] v8 = numpy.empty((4,2), dtype=position_d)
        data_t[::1] cdata = numpy.ascontiguousarray(weights.ravel(), dtype=data_d)
        data_t[::1] cflat, cdark, cpolarization, csolidangle, cvariance, cabsorption
        acc_t[:, ::1] out_data = numpy.zeros((bins, 5), dtype=acc_d)
        data_t[::1] out_intensity = numpy.zeros(bins, dtype=data_d)
        data_t[::1] std, sem
        mask_t[::1] cmask = None
        buffer_t[::1] buffer = numpy.zeros(bins, dtype=buffer_d)
        acc_t sig, var, nrm, cnt, nrm2
        data_t cdummy = 0.0, ddummy = 0.0
        position_t inv_area = 0
        position_t pos0_min = 0, pos0_max = 0, pos0_maxin = 0, pos1_min = 0, pos1_max = 0, pos1_maxin = 0
        position_t area_pixel = 0, sum_area = 0, sub_area = 0, dpos = 0
        position_t a0 = 0, b0 = 0, c0 = 0, d0 = 0, max0 = 0, min0 = 0, a1 = 0, b1 = 0, c1 = 0, d1 = 0, max1 = 0, min1 = 0
        double epsilon = 1e-10
        bint check_pos1=pos1_range is not None, check_mask=False, check_dummy=False, do_dark=False
        bint do_flat=False, do_polarization=False, do_solidangle=False, do_absorption=False
        Py_ssize_t i = 0, idx = 0, bin0_max = 0, bin0_min = 0
        preproc_t value

    if variance is not None:
        assert variance.size == size, "variance size"
        cvariance = numpy.ascontiguousarray(variance.ravel(), dtype=data_d)
        error_model = max(error_model, 1)
    if error_model:
        std = numpy.zeros(bins, dtype=data_d)
        sem = numpy.zeros(bins, dtype=data_d)

    if mask is not None:
        check_mask = True
        assert mask.size == size, "mask size"
        cmask = numpy.ascontiguousarray(mask.ravel(), dtype=mask_d)

    pos0_min, pos0_maxin, pos1_min, pos1_maxin = calc_boundaries(cpos, cmask, pos0_range, pos1_range,
                                                                 allow_pos0_neg=allow_pos0_neg,
                                                                 chiDiscAtPi=chiDiscAtPi)
    if not allow_pos0_neg:
        pos0_min = max(0.0, pos0_min)
        pos0_maxin = max(0.0, pos0_maxin)
    pos0_max = calc_upper_bound(pos0_maxin)
    pos1_max = calc_upper_bound(pos1_maxin)

    dpos = (pos0_max - pos0_min) / (<position_t> (bins))

    if (dummy is not None) and (delta_dummy is not None):
        check_dummy = True
        cdummy = float(dummy)
        ddummy = float(delta_dummy)
    elif (dummy is not None):
        check_dummy = True
        cdummy = float(dummy)
        ddummy = 0.0
    else:
        check_dummy = False
        cdummy = empty
        ddummy = 0.0

    if dark is not None:
        do_dark = True
        assert dark.size == size, "dark current array size"
        cdark = numpy.ascontiguousarray(dark.ravel(), dtype=data_d)
    if flat is not None:
        do_flat = True
        assert flat.size == size, "flat-field array size"
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
        assert absorption.size == size, "absorption array size"
        cabsorption = numpy.ascontiguousarray(absorption.ravel(), dtype=data_d)

    with nogil:
        for idx in range(size):

            if (check_mask) and (cmask[idx]):
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
                                             dark_variance=0.0,
                                             error_model=error_model,
                                             apply_normalization=not weighted_average)

            # Play with coordinates ...
            v8[:, :] = cpos[idx, :, :]
            area_pixel = _recenter(v8, chiDiscAtPi) # this area is only approximate
            a0 = get_bin_number(v8[0, 0], pos0_min, dpos)
            a1 = v8[0, 1]
            b0 = get_bin_number(v8[1, 0], pos0_min, dpos)
            b1 = v8[1, 1]
            c0 = get_bin_number(v8[2, 0], pos0_min, dpos)
            c1 = v8[2, 1]
            d0 = get_bin_number(v8[3, 0], pos0_min, dpos)
            d1 = v8[3, 1]

            min0 = min(a0, b0, c0, d0)
            max0 = max(a0, b0, c0, d0)

            if (max0 < 0) or (min0 >= bins):
                continue
            if check_pos1:
                min1 = min(a1, b1, c1, d1)
                max1 = max(a1, b1, c1, d1)
                if (max1 < pos1_min) or (min1 > pos1_maxin):
                    continue

            bin0_min = < Py_ssize_t > floor(min0)
            bin0_max = < Py_ssize_t > floor(max0)

            if bin0_min == bin0_max:
                # All pixel is within a single bin
                update_1d_accumulator(out_data, bin0_min, value, 1.0, error_model)

            # else we have pixel splitting.
            else:
                bin0_min = max(0, bin0_min)
                bin0_max = min(bins, bin0_max + 1)

                _integrate1d(buffer, a0, a1, b0, b1)  # A-B
                _integrate1d(buffer, b0, b1, c0, c1)  # B-C
                _integrate1d(buffer, c0, c1, d0, d1)  # C-D
                _integrate1d(buffer, d0, d1, a0, a1)  # D-A

                # Distribute pixel area
                sum_area = 0.0
                for bin in range(bin0_min, bin0_max):
                    sum_area += buffer[bin]
                if sum_area != 0.0:
                    inv_area = 1.0 / sum_area
                    for bin in range(bin0_min, bin0_max):
                        update_1d_accumulator(out_data, bin, value, buffer[bin]*inv_area, error_model)
                buffer[bin0_min:bin0_max] = 0.0
        for i in range(bins):
            sig = out_data[i, 0]
            var = out_data[i, 1]
            nrm = out_data[i, 2]
            cnt = out_data[i, 3]
            nrm2 = out_data[i, 4]
            if cnt:
                "test on count as norm can be negative "
                out_intensity[i] = sig / nrm
                if error_model:
                    sem[i] = sqrt(var) / nrm
                    std[i] = sqrt(var/nrm2)
            else:
                out_intensity[i] = empty
                if error_model:
                    sem[i] = empty
                    std[i] = empty

    bin_centers = numpy.linspace(pos0_min + 0.5 * dpos,
                                 pos0_max - 0.5 * dpos,
                                 bins)

    return Integrate1dtpl(bin_centers, numpy.asarray(out_intensity),
                          numpy.asarray(sem) if error_model else None,
                          numpy.asarray(out_data[:, 0]),
                          numpy.asarray(out_data[:, 1]) if error_model else None,
                          numpy.asarray(out_data[:, 2]),
                          numpy.asarray(out_data[:, 3]),
                          std if error_model else None,
                          sem if error_model else None,
                          numpy.asarray(out_data[:, 4]) if error_model else None)

fullSplit1D_ng = fullSplit1D_engine


def fullSplit2D(pos,
                weights,
                bins not None,
                pos0_range=None,
                pos1_range=None,
                dummy=None,
                delta_dummy=None,
                mask=None,
                dark=None,
                flat=None,
                solidangle=None,
                polarization=None,
                bint allow_pos0_neg=True,
                bint chiDiscAtPi=1,
                float empty=0.0,
                double normalization_factor=1.0,
                Py_ssize_t coef_power=1
                ):
    """
    Calculate 2D histogram of pos weighted by weights

    Splitting is done on the pixel's bounding box like fit2D


    :param pos: 3D array with pos0; Corner A,B,C,D; tth or chi
    :param weights: array with intensities
    :param bins: number of output bins int or 2-tuple of int
    :param pos0_range: minimum and maximum  of the 2th range
    :param pos1_range: minimum and maximum  of the chi range
    :param dummy: value for bins without pixels
    :param delta_dummy: precision of dummy value
    :param mask: array (of int8) with masked pixels with 1 (0=not masked)
    :param dark: array (of float64) with dark noise to be subtracted (or None)
    :param flat: array (of float64) with flat-field image
    :param polarization: array (of float64) with polarization correction
    :param solidangle: array (of float64)with solid angle corrections
    :param allow_pos0_neg: allow radial dimention to be negative (useful in log-scale!)
    :param chiDiscAtPi: boolean; by default the chi_range is in the range ]-pi,pi[ set to 0 to have the range ]0,2pi[
    :param empty: value of output bins without any contribution when dummy is None
    :param normalization_factor: divide the valid result by this value
    :param coef_power: set to 2 for variance propagation, leave to 1 for mean calculation
    :return: I, edges0, edges1, weighted histogram(2D), unweighted histogram (2D)
    """

    cdef Py_ssize_t bins0 = 0, bins1 = 0, size = weights.size
    if pos.ndim > 3:  # create a view
        pos = pos.reshape((-1, 4, 2))

    assert pos.shape[0] == size, "pos.shape[0] == size"
    assert pos.shape[1] == 4, "pos.shape[1] == 4: pos has 4 corners"
    assert pos.shape[2] == 2, "pos.shape[2] == 2: tth and chi"
    assert pos.ndim == 3, "pos.ndim == 3"
    try:
        bins0, bins1 = tuple(bins)
    except TypeError:
        bins0 = bins1 = < Py_ssize_t > bins
    if bins0 <= 0:
        bins0 = 1
    if bins1 <= 0:
        bins1 = 1
    cdef:
        position_t[:, :, ::1] cpos = numpy.ascontiguousarray(pos, dtype=position_d)
        data_t[::1] cdata = numpy.ascontiguousarray(weights.ravel(), dtype=data_d)
        acc_t[:, ::1] sum_data = numpy.zeros((bins0, bins1), dtype=acc_d)
        acc_t[:, ::1] sum_count = numpy.zeros((bins0, bins1), dtype=acc_d)
        data_t[:, ::1] merged = numpy.zeros((bins0, bins1), dtype=data_d)
        mask_t[:] cmask = None
        data_t[:] cflat, cdark, cpolarization, csolidangle
        bint check_mask = False, check_dummy = False, do_dark = False, do_flat = False, do_polarization = False, do_solidangle = False
        data_t cdummy = 0, cddummy = 0, data = 0
        position_t min0 = 0, max0 = 0, min1 = 0, max1 = 0, delta_right = 0, delta_left = 0, delta_up = 0, delta_down = 0, inv_area = 0
        position_t pos0_min = 0, pos0_max = 0, pos1_min = 0, pos1_max = 0, pos0_maxin = 0, pos1_maxin = 0
        position_t area_pixel = 0, fbin0_min = 0, fbin0_max = 0, fbin1_min = 0, fbin1_max = 0
        position_t a0 = 0, a1 = 0, b0 = 0, b1 = 0, c0 = 0, c1 = 0, d0 = 0, d1 = 0
        position_t epsilon = 1e-10
        position_t delta0, delta1
        Py_ssize_t bin0_max = 0, bin0_min = 0, bin1_max = 0, bin1_min = 0, i = 0, j = 0, idx = 0

    if mask is not None:
        check_mask = True
        assert mask.size == size, "mask size"
        cmask = numpy.ascontiguousarray(mask.ravel(), dtype=mask_d)

    pos0_min, pos0_maxin, pos1_min, pos1_maxin = calc_boundaries(cpos, cmask, pos0_range, pos1_range,
                                                                 allow_pos0_neg=allow_pos0_neg,
                                                                 chiDiscAtPi=chiDiscAtPi)
    if (not allow_pos0_neg):
        pos0_min = max(0.0, pos0_min)
        pos0_maxin = max(0.0, pos0_maxin)
    if pos0_range:
        pos0_min = min(pos0_range)
        pos0_maxin = max(pos0_range)
    if pos1_range:
        pos1_min = min(pos1_range)
        pos1_maxin = max(pos1_range)

    pos0_max = calc_upper_bound(pos0_maxin)
    pos1_max = calc_upper_bound(pos1_maxin)

    delta0 = (pos0_max - pos0_min) / (<position_t> (bins0))
    delta1 = (pos1_max - pos1_min) / (<position_t> (bins1))

    if (dummy is not None) and (delta_dummy is not None):
        check_dummy = True
        cdummy = <data_t> float(dummy)
        cddummy = <data_t> float(delta_dummy)
    elif (dummy is not None):
        check_dummy = True
        cdummy = <data_t> float(dummy)
        cddummy = 0.0
    else:
        check_dummy = False
        cdummy = <data_t> float(empty)
        cddummy = 0.0

    if dark is not None:
        do_dark = True
        assert dark.size == size, "dark current array size"
        cdark = numpy.ascontiguousarray(dark.ravel(), dtype=data_d)
    if flat is not None:
        do_flat = True
        assert flat.size == size, "flat-field array size"
        cflat = numpy.ascontiguousarray(flat.ravel(), dtype=data_d)
    if polarization is not None:
        do_polarization = True
        assert polarization.size == size, "polarization array size"
        cpolarization = numpy.ascontiguousarray(polarization.ravel(), dtype=data_d)
    if solidangle is not None:
        do_solidangle = True
        assert solidangle.size == size, "Solid angle array size"
        csolidangle = numpy.ascontiguousarray(solidangle.ravel(), dtype=data_d)

    with nogil:
        for idx in range(size):

            if (check_mask) and (cmask[idx]):
                continue

            data = cdata[idx]
            if check_dummy and ((cddummy == 0.0 and data == cdummy) or (cddummy != 0.0 and fabs(data - cdummy) <= cddummy)):
                continue

            a0 = cpos[idx, 0, 0]
            a1 = cpos[idx, 0, 1]
            b0 = cpos[idx, 1, 0]
            b1 = cpos[idx, 1, 1]
            c0 = cpos[idx, 2, 0]
            c1 = cpos[idx, 2, 1]
            d0 = cpos[idx, 3, 0]
            d1 = cpos[idx, 3, 1]

            min0 = min(a0, b0, c0, d0)
            max0 = max(a0, b0, c0, d0)
            min1 = min(a1, b1, c1, d1)
            max1 = max(a1, b1, c1, d1)

            if (max0 < pos0_min) or (min0 > pos0_maxin) or (max1 < pos1_min) or (min1 > pos1_maxin):
                    continue

            if not allow_pos0_neg:
                min0 = max(min0, 0.0)
                max0 = max(max0, 0.0)

            if max1 > (2 - chiDiscAtPi) * pi:
                max1 = (2 - chiDiscAtPi) * pi
            if min1 < (-chiDiscAtPi) * pi:
                min1 = (-chiDiscAtPi) * pi

            if do_dark:
                data -= cdark[idx]
            if do_flat:
                data /= cflat[idx]
            if do_polarization:
                data /= cpolarization[idx]
            if do_solidangle:
                data /= csolidangle[idx]

            if min0 < pos0_min:
                data = data * (pos0_min - min0) / (max0 - min0)
                min0 = pos0_min
            if min1 < pos1_min:
                data = data * (pos1_min - min1) / (max1 - min1)
                min1 = pos1_min
            if max0 > pos0_maxin:
                data = data * (max0 - pos0_maxin) / (max0 - min0)
                max0 = pos0_maxin
            if max1 > pos1_maxin:
                data = data * (max1 - pos1_maxin) / (max1 - min1)
                max1 = pos1_maxin

            # Treat data for pixel on chi discontinuity
            if ((max1 - min1) / delta1) > (0.5 * bins1):
                if pos1_maxin - max1 > min1 - pos1_min:
                    min1 = max1
                    max1 = pos1_maxin
                else:
                    max1 = min1
                    min1 = pos1_min

            fbin0_min = get_bin_number(min0, pos0_min, delta0)
            fbin0_max = get_bin_number(max0, pos0_min, delta0)
            fbin1_min = get_bin_number(min1, pos1_min, delta1)
            fbin1_max = get_bin_number(max1, pos1_min, delta1)

            bin0_min = < Py_ssize_t > fbin0_min
            bin0_max = < Py_ssize_t > fbin0_max
            bin1_min = < Py_ssize_t > fbin1_min
            bin1_max = < Py_ssize_t > fbin1_max

            if bin0_min == bin0_max:
                if bin1_min == bin1_max:
                    # All pixel is within a single bin
                    sum_count[bin0_min, bin1_min] += 1.0
                    sum_data[bin0_min, bin1_min] += data
                else:
                    # spread on more than 2 bins
                    area_pixel = fbin1_max - fbin1_min
                    delta_down = (<acc_t> (bin1_min + 1)) - fbin1_min
                    delta_up = fbin1_max - (<acc_t> bin1_max)
                    inv_area = 1.0 / area_pixel

                    sum_count[bin0_min, bin1_min] += inv_area * delta_down
                    sum_data[bin0_min, bin1_min] += data * pown(inv_area * delta_down, coef_power)

                    sum_count[bin0_min, bin1_max] += inv_area * delta_up
                    sum_data[bin0_min, bin1_max] += data * pown(inv_area * delta_up, coef_power)
                    # if bin1_min + 1 < bin1_max:
                    for j in range(bin1_min + 1, bin1_max):
                            sum_count[bin0_min, j] += inv_area
                            sum_data[bin0_min, j] += data * pown(inv_area, coef_power)

            else:
                # spread on more than 2 bins in dim 0
                if bin1_min == bin1_max:
                    # All pixel fall on 1 bins in dim 1
                    area_pixel = fbin0_max - fbin0_min
                    delta_left = (<acc_t> (bin0_min + 1)) - fbin0_min
                    inv_area = delta_left / area_pixel
                    sum_count[bin0_min, bin1_min] += inv_area
                    sum_data[bin0_min, bin1_min] += data * pown(inv_area, coef_power)
                    delta_right = fbin0_max - (<acc_t> bin0_max)
                    inv_area = delta_right / area_pixel
                    sum_count[bin0_max, bin1_min] += inv_area
                    sum_data[bin0_max, bin1_min] += data * pown(inv_area, coef_power)
                    inv_area = 1.0 / area_pixel
                    for i in range(bin0_min + 1, bin0_max):
                            sum_count[i, bin1_min] += inv_area
                            sum_data[i, bin1_min] += data * pown(inv_area, coef_power)
                else:
                    # spread on n pix in dim0 and m pixel in dim1:
                    area_pixel = (fbin0_max - fbin0_min) * (fbin1_max - fbin1_min)
                    delta_left = (<acc_t> (bin0_min + 1.0)) - fbin0_min
                    delta_right = fbin0_max - (<position_t> bin0_max)
                    delta_down = (<acc_t> (bin1_min + 1.0)) - fbin1_min
                    delta_up = fbin1_max - (<acc_t> bin1_max)
                    inv_area = 1.0 / area_pixel

                    sum_count[bin0_min, bin1_min] += inv_area * delta_left * delta_down
                    sum_data[bin0_min, bin1_min] += data * pown(inv_area * delta_left * delta_down, coef_power)

                    sum_count[bin0_min, bin1_max] += inv_area * delta_left * delta_up
                    sum_data[bin0_min, bin1_max] += data * pown(inv_area * delta_left * delta_up, coef_power)

                    sum_count[bin0_max, bin1_min] += inv_area * delta_right * delta_down
                    sum_data[bin0_max, bin1_min] += data * pown(inv_area * delta_right * delta_down, coef_power)

                    sum_count[bin0_max, bin1_max] += inv_area * delta_right * delta_up
                    sum_data[bin0_max, bin1_max] += data * pown(inv_area * delta_right * delta_up, coef_power)
                    for i in range(bin0_min + 1, bin0_max):
                            sum_count[i, bin1_min] += inv_area * delta_down
                            sum_data[i, bin1_min] += data * pown(inv_area * delta_down, coef_power)
                            for j in range(bin1_min + 1, bin1_max):
                                sum_count[i, j] += inv_area
                                sum_data[i, j] += data * pown(inv_area, coef_power)
                            sum_count[i, bin1_max] += inv_area * delta_up
                            sum_data[i, bin1_max] += data * pown(inv_area * delta_up, coef_power)
                    for j in range(bin1_min + 1, bin1_max):
                            sum_count[bin0_min, j] += inv_area * delta_left
                            sum_data[bin0_min, j] += data * pown(inv_area * delta_left, coef_power)

                            sum_count[bin0_max, j] += inv_area * delta_right
                            sum_data[bin0_max, j] += data * pown(inv_area * delta_right, coef_power)

    # with nogil:
        for i in range(bins0):
            for j in range(bins1):
                if sum_count[i, j] > epsilon:
                    merged[i, j] = sum_data[i, j] / sum_count[i, j] / normalization_factor
                else:
                    merged[i, j] = cdummy

    bin_centers0 = numpy.linspace(pos0_min + 0.5 * delta0,
                                  pos0_max - 0.5 * delta0,
                                  bins0)
    bin_centers1 = numpy.linspace(pos1_min + 0.5 * delta1,
                                  pos1_max - 0.5 * delta1,
                                  bins1)

    return (numpy.asarray(merged).T,
            bin_centers0,
            bin_centers1,
            numpy.asarray(sum_data).T,
            numpy.asarray(sum_count).T)


def pseudoSplit2D_engine(pos not None,
                         weights not None,
                         bins not None,
                         pos0_range=None,
                         pos1_range=None,
                         dummy=None,
                         delta_dummy=None,
                         mask=None,
                         variance=None,
                         dark_variance=None,
                         int error_model=ErrorModel.NO,
                         dark=None,
                         flat=None,
                         solidangle=None,
                         polarization=None,
                         absorption=None,
                         bint allow_pos0_neg=0,
                         bint chiDiscAtPi=1,
                         float empty=0.0,
                         double normalization_factor=1.0,
                         bint weighted_average=True,
                         ):
    """
    Calculate 2D histogram of pos weighted by weights

    Splitting is done on the pixel's bounding box, similar to fit2D
    New implementation with variance propagation



    :param pos: 3D array with pos0; Corner A,B,C,D; tth or chi
    :param weights: array with intensities
    :param bins: number of output bins int or 2-tuple of int
    :param pos0_range: minimum and maximum  of the 2th range
    :param pos1_range: minimum and maximum  of the chi range
    :param dummy: value for bins without pixels
    :param delta_dummy: precision of dummy value
    :param mask: array (of int8) with masked pixels with 1 (0=not masked)
    :param variance: variance associated with the weights
    :param dark: array (of float64) with dark noise to be subtracted (or None)
    :param flat: array (of float64) with flat-field image
    :param polarization: array (of float64) with polarization correction
    :param solidangle: array (of float64)with solid angle corrections
    :param absorption: array with absorption correction
    :param allow_pos0_neg: set to true to allow negative radial values.
    :param chiDiscAtPi: boolean; by default the chi_range is in the range ]-pi,pi[ set to 0 to have the range ]0,2pi[
    :param empty: value of output bins without any contribution when dummy is None
    :param normalization_factor: divide the valid result by this value
    :param bool weighted_average: set to False to use an unweigted mean (similar to legacy) instead of the weigted average.
    :return: Integrate2dtpl namedtuple: "radial azimuthal intensity error signal variance normalization count"
    """

    cdef Py_ssize_t bins0 = 0, bins1 = 0, size = weights.size
    if pos.ndim > 3:  # create a view
        pos = pos.reshape((-1, 4, 2))

    assert pos.shape[0] == size, "pos.shape[0] == size"
    assert pos.shape[1] == 4, "pos.shape[1] == 4: pos has 4 corners"
    assert pos.shape[2] == 2, "pos.shape[2] == 2: tth and chi"
    assert pos.ndim == 3, "pos.ndim == 3"
    try:
        bins0, bins1 = tuple(bins)
    except TypeError:
        bins0 = bins1 = < Py_ssize_t > bins
    if bins0 <= 0:
        bins0 = 1
    if bins1 <= 0:
        bins1 = 1
    cdef:
        position_t[:, :, ::1] cpos = numpy.ascontiguousarray(pos, dtype=position_d)
        data_t[::1] cdata = numpy.ascontiguousarray(weights.ravel(), dtype=data_d)
        acc_t[:, :, ::1] out_data = numpy.zeros((bins0, bins1, 5), dtype=acc_d)
        data_t[:, ::1] out_intensity = numpy.empty((bins0, bins1), dtype=data_d)
        data_t[:, ::1] std, sem
        mask_t[:] cmask = None
        data_t[:] cflat, cdark, cpolarization, csolidangle, cvariance, cabsorption
        bint check_mask = False, check_dummy = False, do_dark = False, do_absorption=False
        bint do_flat = False, do_polarization = False, do_solidangle = False,
        bint is_valid
        data_t cdummy = 0, cddummy = 0, scale = 1
        position_t min0 = 0, max0 = 0, min1 = 0, max1 = 0, delta_right = 0, delta_left = 0, delta_up = 0, delta_down = 0, inv_area = 0
        position_t pos0_min = 0, pos0_max = 0, pos1_min = 0, pos1_max = 0, pos0_maxin = 0, pos1_maxin = 0
        position_t fbin0_min = 0, fbin0_max = 0, fbin1_min = 0, fbin1_max = 0
        position_t a0 = 0, a1 = 0, b0 = 0, b1 = 0, c0 = 0, c1 = 0, d0 = 0, d1 = 0
        position_t center0 = 0.0, center1 = 0.0, area, width, height,
        position_t delta0, delta1, new_width, new_height, new_min0, new_max0, new_min1, new_max1
        Py_ssize_t bin0_max = 0, bin0_min = 0, bin1_max = 0, bin1_min = 0, i = 0, j = 0, idx = 0
        acc_t sig, var, norm, cnt, norm2
        preproc_t value

    if mask is not None:
        check_mask = True
        assert mask.size == size, "mask size"
        cmask = numpy.ascontiguousarray(mask.ravel(), dtype=mask_d)

    pos0_min, pos0_maxin, pos1_min, pos1_maxin = calc_boundaries(cpos, cmask, pos0_range, pos1_range,
                                                                 allow_pos0_neg=allow_pos0_neg,
                                                                 chiDiscAtPi=chiDiscAtPi)
    if (not allow_pos0_neg):
        pos0_min = max(0.0, pos0_min)
        pos0_maxin = max(pos0_maxin, 0.0)
    pos0_max = calc_upper_bound(pos0_maxin)
    pos1_max = calc_upper_bound(pos1_maxin)

    delta0 = (pos0_max - pos0_min) / (<position_t> (bins0))
    delta1 = (pos1_max - pos1_min) / (<position_t> (bins1))

    if (dummy is not None) and (delta_dummy is not None):
        check_dummy = True
        cdummy = <data_t> float(dummy)
        cddummy = <data_t> float(delta_dummy)
    elif (dummy is not None):
        check_dummy = True
        cdummy = <data_t> float(dummy)
        cddummy = 0.0
    else:
        check_dummy = False
        cdummy = <data_t> float(empty)
        cddummy = 0.0

    if variance is not None:
        assert variance.size == size, "variance size"
        cvariance = numpy.ascontiguousarray(variance.ravel(), dtype=data_d)
        error_model = max(error_model, 1)
    if error_model:
        std = numpy.zeros((bins0, bins1), dtype=data_d)
        sem = numpy.zeros((bins0, bins1), dtype=data_d)

    if dark is not None:
        do_dark = True
        assert dark.size == size, "dark current array size"
        cdark = numpy.ascontiguousarray(dark.ravel(), dtype=data_d)
    if flat is not None:
        do_flat = True
        assert flat.size == size, "flat-field array size"
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
        assert absorption.size == size, "absorption array size"
        cabsorption = numpy.ascontiguousarray(absorption.ravel(), dtype=data_d)

    with nogil:
        for idx in range(size):

            if (check_mask) and (cmask[idx]):
                continue

            a0 = cpos[idx, 0, 0]
            a1 = cpos[idx, 0, 1]
            b0 = cpos[idx, 1, 0]
            b1 = cpos[idx, 1, 1]
            c0 = cpos[idx, 2, 0]
            c1 = cpos[idx, 2, 1]
            d0 = cpos[idx, 3, 0]
            d1 = cpos[idx, 3, 1]

            min0 = min(a0, b0, c0, d0)
            max0 = max(a0, b0, c0, d0)
            min1 = min(a1, b1, c1, d1)
            max1 = max(a1, b1, c1, d1)

            if (max0 < pos0_min) or (min0 > pos0_maxin) or (max1 < pos1_min) or (min1 > pos1_maxin):
                    continue

            # Recalculate here min0, max0, min1, max1 based on the actual area of ABCD and the width/height ratio
            center0 = (a0 + b0 + c0 + d0) / 4.0
            center1 = (a1 + b1 + c1 + d1) / 4.0
            area = fabs(area4(a0, a1, b0, b1, c0, c1, d0, d1))
            width = (max1 - min1)
            height = (max0 - min0)
            if (width != 0) and (height != 0):
                new_height = sqrt(area * height / width)
                new_width = new_height * width / height

                new_min0 = center0 - new_width / 2.0
                new_max0 = center0 + new_width / 2.0
                new_min1 = center1 - new_height / 2.0
                new_max1 = center1 + new_height / 2.0
            if (new_min0 < min0) or (new_max0 > max0) or (new_min1 < min1) or (new_max1 > max1):
                # This is a pathological pixel laying on the Chi discontinuity

                with gil:

                    logger.debug("%s -> %s; %s -> %s; %s -> %s; %s -> %s",
                                 min0, new_min0, max0, new_max0, min1, new_min1,
                                 max1, new_max1)
            else:
                min0 = new_min0
                max0 = new_max0
                min1 = new_min1
                max1 = new_max1

            if not allow_pos0_neg:
                min0 = max(0.0, min0)
                max0 = max(0.0, max0)

            if max1 > (2 - chiDiscAtPi) * pi:
                max1 = (2 - chiDiscAtPi) * pi
            if min1 < (-chiDiscAtPi) * pi:
                min1 = (-chiDiscAtPi) * pi

            is_valid = preproc_value_inplace(&value,
                                             cdata[idx],
                                             variance=cvariance[idx] if error_model==1 else 0.0,
                                             dark=cdark[idx] if do_dark else 0.0,
                                             flat=cflat[idx] if do_flat else 1.0,
                                             solidangle=csolidangle[idx] if do_solidangle else 1.0,
                                             polarization=cpolarization[idx] if do_polarization else 1.0,
                                             absorption=cabsorption[idx] if do_absorption else 1.0,
                                             mask=cmask[idx] if check_mask else 0,
                                             dummy=cdummy,
                                             delta_dummy=cddummy,
                                             check_dummy=check_dummy,
                                             normalization_factor=normalization_factor,
                                             dark_variance=0.0,
                                             error_model=error_model,
                                             apply_normalization=not weighted_average)
            if not is_valid:
                continue

            scale = 1.0
            if min0 < pos0_min:
                scale = scale * (pos0_min - min0) / (max0 - min0)
                min0 = pos0_min
            if min1 < pos1_min:
                scale = scale * (pos1_min - min1) / (max1 - min1)
                min1 = pos1_min
            if max0 > pos0_maxin:
                scale = scale * (max0 - pos0_maxin) / (max0 - min0)
                max0 = pos0_maxin
            if max1 > pos1_maxin:
                scale = scale * (max1 - pos1_maxin) / (max1 - min1)
                max1 = pos1_maxin

            if scale != 1.0:
                value.signal *= scale
                value.norm *= scale
                value.variance *= scale * scale
                value.count *= scale

            # Treat data for pixel on chi discontinuity
            if ((max1 - min1) / delta1) > (bins1 / 2.0):
                if pos1_maxin - max1 > min1 - pos1_min:
                    min1 = max1
                    max1 = pos1_maxin
                else:
                    max1 = min1
                    min1 = pos1_min

            fbin0_min = get_bin_number(min0, pos0_min, delta0)
            fbin0_max = get_bin_number(max0, pos0_min, delta0)
            fbin1_min = get_bin_number(min1, pos1_min, delta1)
            fbin1_max = get_bin_number(max1, pos1_min, delta1)

            bin0_min = < Py_ssize_t > fbin0_min
            bin0_max = < Py_ssize_t > fbin0_max
            bin1_min = < Py_ssize_t > fbin1_min
            bin1_max = < Py_ssize_t > fbin1_max

            fbin0_min = get_bin_number(min0, pos0_min, delta0)
            fbin0_max = get_bin_number(max0, pos0_min, delta0)
            fbin1_min = get_bin_number(min1, pos1_min, delta1)
            fbin1_max = get_bin_number(max1, pos1_min, delta1)

            bin0_min = <Py_ssize_t> fbin0_min
            bin0_max = <Py_ssize_t> fbin0_max
            bin1_min = <Py_ssize_t> fbin1_min
            bin1_max = <Py_ssize_t> fbin1_max

            if bin0_min == bin0_max:
                # No spread along dim0
                if bin1_min == bin1_max:
                    # All pixel is within a single bin
                    update_2d_accumulator(out_data, bin0_min, bin1_min, value, 1.0)
                else:
                    # spread on 2 or more bins in dim1
                    delta_down = (<acc_t> (bin1_min + 1)) - fbin1_min
                    delta_up = fbin1_max - (bin1_max)
                    inv_area = 1.0 / (fbin1_max - fbin1_min)

                    update_2d_accumulator(out_data, bin0_min, bin1_min, value, inv_area * delta_down)
                    update_2d_accumulator(out_data, bin0_min, bin1_max, value, inv_area * delta_up)
                    for j in range(bin1_min + 1, bin1_max):
                        update_2d_accumulator(out_data, bin0_min, j, value, inv_area)

            else:
                # spread on 2 or more bins in dim 0
                if bin1_min == bin1_max:
                    # All pixel fall inside the same bins in dim 1
                    inv_area = 1.0 / (fbin0_max - fbin0_min)

                    delta_left = (<acc_t> (bin0_min + 1)) - fbin0_min
                    update_2d_accumulator(out_data, bin0_min, bin1_min, value, inv_area * delta_left)

                    delta_right = fbin0_max - (<acc_t> bin0_max)
                    update_2d_accumulator(out_data, bin0_max, bin1_min, value, inv_area * delta_right)
                    for i in range(bin0_min + 1, bin0_max):
                            update_2d_accumulator(out_data, i, bin1_min, value, inv_area)
                else:
                    # spread on n pix in dim0 and m pixel in dim1:
                    inv_area = 1.0 / (fbin0_max - fbin0_min) * (fbin1_max - fbin1_min)

                    delta_left = (<acc_t> (bin0_min + 1)) - fbin0_min
                    delta_right = fbin0_max - (<acc_t> bin0_max)
                    delta_down = (<acc_t> (bin1_min + 1)) - fbin1_min
                    delta_up = fbin1_max - (<acc_t> bin1_max)

                    update_2d_accumulator(out_data, bin0_min, bin1_min, value, inv_area * delta_left * delta_down)
                    update_2d_accumulator(out_data, bin0_min, bin1_max, value, inv_area * delta_left * delta_up)
                    update_2d_accumulator(out_data, bin0_max, bin1_min, value, inv_area * delta_right * delta_down)
                    update_2d_accumulator(out_data, bin0_max, bin1_max, value, inv_area * delta_right * delta_up)
                    for i in range(bin0_min + 1, bin0_max):
                        update_2d_accumulator(out_data, i, bin1_min, value, inv_area * delta_down)
                        for j in range(bin1_min + 1, bin1_max):
                            update_2d_accumulator(out_data, i, j, value, inv_area)
                        update_2d_accumulator(out_data, i, bin1_max, value, inv_area * delta_up)
                    for j in range(bin1_min + 1, bin1_max):
                        update_2d_accumulator(out_data, bin0_min, j, value, inv_area * delta_left)
                        update_2d_accumulator(out_data, bin0_max, j, value, inv_area * delta_right)

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
                    if error_model:
                        sem[i, j] = sqrt(var) / norm
                        std[i, j] = sqrt(var / norm2)
                else:
                    out_intensity[i, j] = empty
                    if error_model:
                        sem[i, j] = empty
                        std[i, j] = empty

    bin_centers0 = numpy.linspace(pos0_min + 0.5 * delta0, pos0_max - 0.5 * delta0, bins0)
    bin_centers1 = numpy.linspace(pos1_min + 0.5 * delta1, pos1_max - 0.5 * delta1, bins1)
    return Integrate2dtpl(bin_centers0, bin_centers1,
                          numpy.asarray(out_intensity).T,
                          numpy.asarray(sem).T if error_model else None,
                          numpy.asarray(out_data[...,0]).T, numpy.asarray(out_data[...,1]).T,
                          numpy.asarray(out_data[...,2]).T, numpy.asarray(out_data[...,3]).T,
                          numpy.asarray(std).T if error_model else None,
                          numpy.asarray(sem).T if error_model else None,
                          numpy.asarray(out_data[...,4]).T if error_model else None)

pseudoSplit2D_ng = pseudoSplit2D_engine


def fullSplit2D_engine(pos not None,
                         weights not None,
                         bins not None,
                         pos0_range=None,
                         pos1_range=None,
                         dummy=None,
                         delta_dummy=None,
                         mask=None,
                         variance=None,
                         dark_variance=None,
                         int error_model=ErrorModel.NO,
                         dark=None,
                         flat=None,
                         solidangle=None,
                         polarization=None,
                         absorption=None,
                         bint allow_pos0_neg=0,
                         bint chiDiscAtPi=1,
                         float empty=0.0,
                         double normalization_factor=1.0,
                         bint weighted_average=True,
                         ):
    """
    Calculate 2D histogram of pos weighted by weights

    Splitting is done on the pixel's boundary (straight segments)
    New implementation with variance propagation

    :param pos: 3D array with pos0; Corner A,B,C,D; tth or chi
    :param weights: array with intensities
    :param bins: number of output bins int or 2-tuple of int
    :param pos0_range: minimum and maximum  of the 2th range
    :param pos1_range: minimum and maximum  of the chi range
    :param dummy: value for bins without pixels
    :param delta_dummy: precision of dummy value
    :param mask: array (of int8) with masked pixels with 1 (0=not masked)
    :param variance: variance associated with the weights
    :param dark: array (of float64) with dark noise to be subtracted (or None)
    :param flat: array (of float64) with flat-field image
    :param polarization: array (of float64) with polarization correction
    :param solidangle: array (of float64)with solid angle corrections
    :param absorption: array with absorption correction
    :param allow_pos0_neg: set to true to allow negative radial values.
    :param chiDiscAtPi: boolean; by default the chi_range is in the range ]-pi,pi[ set to 0 to have the range ]0,2pi[
    :param empty: value of output bins without any contribution when dummy is None
    :param normalization_factor: divide the valid result by this value
    :param bool weighted_average: set to False to use an unweigted mean (similar to legacy) instead of the weigted average.
    :return: Integrate2dtpl namedtuple: "radial azimuthal intensity error signal variance normalization count"
    """

    cdef Py_ssize_t bins0 = 0, bins1 = 0, size = weights.size
    if pos.ndim > 3:  # create a view
        pos = pos.reshape((-1, 4, 2))

    assert pos.shape[0] == size, "pos.shape[0] == size"
    assert pos.shape[1] == 4, "pos.shape[1] == 4: pos has 4 corners"
    assert pos.shape[2] == 2, "pos.shape[2] == 2: tth and chi"
    assert pos.ndim == 3, "pos.ndim == 3"
    try:
        bins0, bins1 = tuple(bins)
    except TypeError:
        bins0 = bins1 = < Py_ssize_t > bins
    bins0 = max(bins0, 1)
    bins1 = max(bins1, 1)

    cdef:
        position_t[:, :, ::1] cpos = numpy.ascontiguousarray(pos, dtype=position_d)
        position_t[:, ::1] v8 = numpy.empty((4,2), dtype=position_d)
        data_t[::1] cdata = numpy.ascontiguousarray(weights.ravel(), dtype=data_d)
        acc_t[:, :, ::1] out_data = numpy.zeros((bins0, bins1, 5), dtype=acc_d)
        data_t[:, ::1] out_intensity = numpy.empty((bins0, bins1), dtype=data_d)
        data_t[:, ::1] std, sem
        mask_t[:] cmask = None
        data_t[:] cflat, cdark, cpolarization, csolidangle, cvariance, cabsorption
        bint check_mask=False, check_dummy=False, do_dark=False, do_absorption=False
        bint do_flat=False, do_polarization=False, do_solidangle=False,
        bint is_valid
        data_t cdummy = 0, cddummy = 0
        position_t min0 = 0, max0 = 0, min1 = 0, max1 = 0, inv_area = 0
        position_t pos0_min = 0, pos0_max = 0, pos1_min = 0, pos1_max = 0, pos0_maxin = 0, pos1_maxin = 0
        position_t fbin0_min = 0, fbin0_max = 0, fbin1_min = 0, fbin1_max = 0
        position_t a0 = 0, a1 = 0, b0 = 0, b1 = 0, c0 = 0, c1 = 0, d0 = 0, d1 = 0
        position_t center0 = 0.0, center1 = 0.0, area, width, height,
        position_t delta0, delta1, new_width, new_height, new_min0, new_max0, new_min1, new_max1
        Py_ssize_t bin0_max = 0, bin0_min = 0, bin1_max = 0, bin1_min = 0, i = 0, j = 0, idx = 0
        acc_t sig, var, norm, cnt, norm2
        preproc_t value
        Py_ssize_t ioffset0, ioffset1, w0, w1, bw0=15, bw1=15
        buffer_t[::1] linbuffer = numpy.empty(256, dtype=buffer_d)
        buffer_t[:, ::1] buffer = numpy.asarray(linbuffer[:(bw0+1)*(bw1+1)]).reshape((bw0+1,bw1+1))
        double foffset0, foffset1, sum_area, loc_area

    pos0_min, pos0_maxin, pos1_min, pos1_maxin = calc_boundaries(cpos, cmask, pos0_range, pos1_range,
                                                                 allow_pos0_neg=allow_pos0_neg,
                                                                 chiDiscAtPi=chiDiscAtPi)
    if not allow_pos0_neg:
        pos0_min = max(0.0, pos0_min)
        pos0_maxin = max(0.0, pos0_maxin)

    pos0_max = calc_upper_bound(pos0_maxin)
    pos1_max = calc_upper_bound(pos1_maxin)

    delta0 = (pos0_max - pos0_min) / (<position_t> (bins0))
    delta1 = (pos1_max - pos1_min) / (<position_t> (bins1))
    if (dummy is not None) and (delta_dummy is not None):
        check_dummy = True
        cdummy = <data_t> float(dummy)
        cddummy = <data_t> float(delta_dummy)
    elif (dummy is not None):
        check_dummy = True
        cdummy = <data_t> float(dummy)
        cddummy = 0.0
    else:
        check_dummy = False
        cdummy = <data_t> float(empty)
        cddummy = 0.0

    if variance is not None:
        assert variance.size == size, "variance size"
        cvariance = numpy.ascontiguousarray(variance.ravel(), dtype=data_d)
        error_model = max(error_model, 1)
    if error_model:
        std = numpy.zeros((bins0, bins1), dtype=data_d)
        sem = numpy.zeros((bins0, bins1), dtype=data_d)

    if mask is not None:
        check_mask = True
        assert mask.size == size, "mask size"
        cmask = numpy.ascontiguousarray(mask.ravel(), dtype=mask_d)
    if dark is not None:
        do_dark = True
        assert dark.size == size, "dark current array size"
        cdark = numpy.ascontiguousarray(dark.ravel(), dtype=data_d)
    if flat is not None:
        do_flat = True
        assert flat.size == size, "flat-field array size"
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
        assert absorption.size == size, "absorption array size"
        cabsorption = numpy.ascontiguousarray(absorption.ravel(), dtype=data_d)

    with nogil:
        for idx in range(size):

            if (check_mask) and (cmask[idx]):
                continue

            is_valid = preproc_value_inplace(&value,
                                             cdata[idx],
                                             variance=cvariance[idx] if error_model==1 else 0.0,
                                             dark=cdark[idx] if do_dark else 0.0,
                                             flat=cflat[idx] if do_flat else 1.0,
                                             solidangle=csolidangle[idx] if do_solidangle else 1.0,
                                             polarization=cpolarization[idx] if do_polarization else 1.0,
                                             absorption=cabsorption[idx] if do_absorption else 1.0,
                                             mask=cmask[idx] if check_mask else 0,
                                             dummy=cdummy,
                                             delta_dummy=cddummy,
                                             check_dummy=check_dummy,
                                             normalization_factor=normalization_factor,
                                             dark_variance=0.0,
                                             error_model=error_model,
                                             apply_normalization=not weighted_average)
            if not is_valid:
                continue

            # Play with coordinates ...
            v8[:, :] = cpos[idx, :, :]
            area = _recenter(v8, chiDiscAtPi)
            a0 = v8[0, 0]
            a1 = v8[0, 1]
            b0 = v8[1, 0]
            b1 = v8[1, 1]
            c0 = v8[2, 0]
            c1 = v8[2, 1]
            d0 = v8[3, 0]
            d1 = v8[3, 1]

            min0 = min(a0, b0, c0, d0)
            max0 = max(a0, b0, c0, d0)
            min1 = min(a1, b1, c1, d1)
            max1 = max(a1, b1, c1, d1)

            if (max0 < pos0_min) or (min0 > pos0_maxin) or (max1 < pos1_min) or (min1 > pos1_maxin):
                    continue

            # Swith to bin space.
            a0 = get_bin_number(_clip(a0, pos0_min, pos0_maxin), pos0_min, delta0)
            a1 = get_bin_number(_clip(a1, pos1_min, pos1_maxin), pos1_min, delta1)
            b0 = get_bin_number(_clip(b0, pos0_min, pos0_maxin), pos0_min, delta0)
            b1 = get_bin_number(_clip(b1, pos1_min, pos1_maxin), pos1_min, delta1)
            c0 = get_bin_number(_clip(c0, pos0_min, pos0_maxin), pos0_min, delta0)
            c1 = get_bin_number(_clip(c1, pos1_min, pos1_maxin), pos1_min, delta1)
            d0 = get_bin_number(_clip(d0, pos0_min, pos0_maxin), pos0_min, delta0)
            d1 = get_bin_number(_clip(d1, pos1_min, pos1_maxin), pos1_min, delta1)

            # Recalculate here min0, max0, min1, max1 based on the actual area of ABCD and the width/height ratio
            min0 = min(a0, b0, c0, d0)
            max0 = max(a0, b0, c0, d0)
            min1 = min(a1, b1, c1, d1)
            max1 = max(a1, b1, c1, d1)
            foffset0 = floor(min0)
            foffset1 = floor(min1)
            ioffset0 = <Py_ssize_t> foffset0
            ioffset1 = <Py_ssize_t> foffset1
            w0 = <Py_ssize_t>(ceil(max0) - foffset0)
            w1 = <Py_ssize_t>(ceil(max1) - foffset1)
            if (w0>bw0) or (w1>bw1):
                if (w0+1)*(w1+1)>linbuffer.shape[0]:
                    with gil:
                        linbuffer = numpy.empty((w0+1)*(w1+1), dtype=buffer_d)
                        buffer = numpy.asarray(linbuffer).reshape((w0+1,w1+1))
                        logger.debug("malloc  %s->%s and %s->%s", w0, bw0, w1, bw1)
                else:
                    with gil:
                        buffer = numpy.asarray(linbuffer[:(w0+1)*(w1+1)]).reshape((w0+1,w1+1))
                        logger.debug("reshape %s->%s and %s->%s", w0, bw0, w1, bw1)
                bw0 = w0
                bw1 = w1
            buffer[:, :] = 0.0

            a0 -= foffset0
            a1 -= foffset1
            b0 -= foffset0
            b1 -= foffset1
            c0 -= foffset0
            c1 -= foffset1
            d0 -= foffset0
            d1 -= foffset1

            # ABCD is anti-trigonometric order: order input position accordingly
            _integrate2d(buffer, a0, a1, b0, b1)
            _integrate2d(buffer, b0, b1, c0, c1)
            _integrate2d(buffer, c0, c1, d0, d1)
            _integrate2d(buffer, d0, d1, a0, a1)


            sum_area = 0.0
            for i in range(w0):
                for j in range(w1):
                    sum_area += buffer[i, j]
            inv_area = 1.0 / sum_area
            for i in range(w0):
                for j in range(w1):
                    update_2d_accumulator(out_data,
                                          ioffset0 + i,
                                          ioffset1 + j,
                                          value,
                                          weight=buffer[i, j] * inv_area)

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
                    if error_model:
                        sem[i, j] = sqrt(var) / norm
                        std[i, j] = sqrt(var / norm2)
                else:
                    out_intensity[i, j] = empty
                    if error_model:
                        sem[i, j] = empty
                        std[i, j] = empty

    bin_centers0 = numpy.linspace(pos0_min + 0.5 * delta0, pos0_max - 0.5 * delta0, bins0)
    bin_centers1 = numpy.linspace(pos1_min + 0.5 * delta1, pos1_max - 0.5 * delta1, bins1)
    return Integrate2dtpl(bin_centers0, bin_centers1,
                          numpy.asarray(out_intensity).T,
                          numpy.asarray(sem).T if error_model else None,
                          numpy.asarray(out_data[...,0]).T, numpy.asarray(out_data[...,1]).T,
                          numpy.asarray(out_data[...,2]).T, numpy.asarray(out_data[...,3]).T,
                          numpy.asarray(std).T if error_model else None,
                          numpy.asarray(sem).T if error_model else None,
                          numpy.asarray(out_data[...,4]).T if error_model else None)
