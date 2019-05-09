# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2018 European Synchrotron Radiation Facility, France
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

Splitting is done on the pixel's bounding box similar to fit2D
"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "09/05/2019"
__status__ = "stable"
__license__ = "MIT"

include "regrid_common.pxi"
import logging
logger = logging.getLogger(__name__)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def histoBBox1d(weights,
                pos0,
                delta_pos0,
                pos1=None,
                delta_pos1=None,
                size_t bins=100,
                pos0Range=None,
                pos1Range=None,
                dummy=None,
                delta_dummy=None,
                mask=None,
                dark=None,
                flat=None,
                solidangle=None,
                polarization=None,
                empty=None,
                double normalization_factor=1.0,
                int coef_power=1):

    """
    Calculates histogram of pos0 (tth) weighted by weights

    Splitting is done on the pixel's bounding box like fit2D

    :param weights: array with intensities
    :param pos0: 1D array with pos0: tth or q_vect
    :param delta_pos0: 1D array with delta pos0: max center-corner distance
    :param pos1: 1D array with pos1: chi
    :param delta_pos1: 1D array with max pos1: max center-corner distance, unused !
    :param bins: number of output bins
    :param pos0Range: minimum and maximum  of the 2th range
    :param pos1Range: minimum and maximum  of the chi range
    :param dummy: value for bins without pixels & value of "no good" pixels
    :param delta_dummy: precision of dummy value
    :param mask: array (of int8) with masked pixels with 1 (0=not masked)
    :param dark: array (of float32) with dark noise to be subtracted (or None)
    :param flat: array (of float32) with flat-field image
    :param solidangle: array (of float32) with solid angle corrections
    :param polarization: array (of float32) with polarization corrections
    :param empty: value of output bins without any contribution when dummy is None
    :param normalization_factor: divide the result by this value
    :param coef_power: set to 2 for variance propagation, leave to 1 for mean calculation

    :return: 2theta, I, weighted histogram, unweighted histogram
    """
    cdef size_t  size = weights.size
    assert pos0.size == size, "pos0.size == size"
    assert delta_pos0.size == size, "delta_pos0.size == size"
    assert bins > 1, "at lease one bin"
    cdef:
        ssize_t  idx, bin0_max, bin0_min
        data_t data, cdummy = 0.0, ddummy = 0.0
        acc_t epsilon = 1e-10,
        position_t pos0_min = 0.0, pos1_min = 0.0, pos0_max = 0.0, pos1_max = 0.0
        position_t pos0_maxin = 0.0, pos1_maxin = 0.0, min0 = 0.0, max0 = 0.0, fbin0_min = 0.0, fbin0_max = 0.0
        bint check_pos1 = False, check_mask = False, check_dummy = False
        bint do_dark = False, do_flat = False, do_polarization = False, do_solidangle = False
        double delta

        data_t[::1] cdata, cflat, cdark, cpolarization, csolidangle
        position_t[::1] cpos0, dpos0, cpos1, dpos1, cpos0_lower, cpos0_upper
        mask_t[::1] cmask
        acc_t inv_area, delta_right, delta_left

    cdata = numpy.ascontiguousarray(weights.ravel(), dtype=data_d)
    cpos0 = numpy.ascontiguousarray(pos0.ravel(), dtype=position_d)
    dpos0 = numpy.ascontiguousarray(delta_pos0.ravel(), dtype=position_d)
    cdef:
        acc_t[::1] sum_data = numpy.zeros(bins, dtype=acc_d)
        acc_t[::1] sum_count = numpy.zeros(bins, dtype=acc_d)
        data_t[::1] out_merge = numpy.zeros(bins, dtype=data_d)

    if mask is not None:
        assert mask.size == size, "mask size"
        check_mask = True
        cmask = numpy.ascontiguousarray(mask.ravel(), dtype=mask_d)

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
        cdummy = empty or 0.0
        ddummy = 0.0
    if dark is not None:
        assert dark.size == size, "dark current array size"
        do_dark = True
        cdark = numpy.ascontiguousarray(dark.ravel(), dtype=data_d)
    if flat is not None:
        assert flat.size == size, "flat-field array size"
        do_flat = True
        cflat = numpy.ascontiguousarray(flat.ravel(), dtype=data_d)
    if polarization is not None:
        do_polarization = True
        assert polarization.size == size, "polarization array size"
        cpolarization = numpy.ascontiguousarray(polarization.ravel(), dtype=numpy.float32)
    if solidangle is not None:
        do_solidangle = True
        assert solidangle.size == size, "Solid angle array size"
        csolidangle = numpy.ascontiguousarray(solidangle.ravel(), dtype=numpy.float32)

    cpos0_lower = numpy.zeros(size, dtype=position_d)
    cpos0_upper = numpy.zeros(size, dtype=position_d)
    pos0_min = cpos0[0]
    pos0_max = cpos0[0]
    with nogil:
        for idx in range(size):
            if (check_mask) and (cmask[idx]):
                continue
            min0 = cpos0[idx] - dpos0[idx]
            max0 = cpos0[idx] + dpos0[idx]
            cpos0_upper[idx] = max0
            cpos0_lower[idx] = min0
            if max0 > pos0_max:
                pos0_max = max0
            if min0 < pos0_min:
                pos0_min = min0

    if pos0Range is not None and len(pos0Range) > 1:
        pos0_min = min(pos0Range)
        pos0_maxin = max(pos0Range)
    else:
        pos0_maxin = pos0_max
    if pos0_min < 0:
        pos0_min = 0
    pos0_max = calc_upper_bound(pos0_maxin)

    if pos1Range is not None and len(pos1Range) > 1:
        assert pos1.size == size, "pos1.size == size"
        assert delta_pos1.size == size, "delta_pos1.size == size"
        check_pos1 = 1
        cpos1 = numpy.ascontiguousarray(pos1.ravel(), dtype=position_d)
        dpos1 = numpy.ascontiguousarray(delta_pos1.ravel(), dtype=position_d)
        pos1_min = min(pos1Range)
        pos1_maxin = max(pos1Range)
        pos1_max = calc_upper_bound(pos1_maxin)

    delta = (pos0_max - pos0_min) / (<position_t> (bins))

    with nogil:
        for idx in range(size):
            if (check_mask) and (cmask[idx]):
                continue

            data = cdata[idx]
            if check_dummy and (fabs(data - cdummy) <= ddummy):
                continue

            min0 = cpos0_lower[idx]
            max0 = cpos0_upper[idx]

            if check_pos1 and (((cpos1[idx] + dpos1[idx]) < pos1_min) or ((cpos1[idx] - dpos1[idx]) > pos1_max)):
                    continue

            fbin0_min = get_bin_number(min0, pos0_min, delta)
            fbin0_max = get_bin_number(max0, pos0_min, delta)
            if (fbin0_max < 0) or (fbin0_min >= bins):
                continue
            if fbin0_max >= bins:
                bin0_max = bins - 1
            else:
                bin0_max = < ssize_t > fbin0_max
            if fbin0_min < 0:
                bin0_min = 0
            else:
                bin0_min = < ssize_t > fbin0_min

            if do_dark:
                data -= cdark[idx]
            if do_flat:
                data /= cflat[idx]
            if do_polarization:
                data /= cpolarization[idx]
            if do_solidangle:
                data /= csolidangle[idx]

            if bin0_min == bin0_max:
                # All pixel is within a single bin
                sum_count[bin0_min] += 1.0
                sum_data[bin0_min] += data

            else:
                # we have pixel splitting.
                inv_area = 1.0 / (fbin0_max - fbin0_min)

                delta_left = < float > (bin0_min + 1) - fbin0_min
                delta_right = fbin0_max - (<float> bin0_max)

                sum_count[bin0_min] += (inv_area * delta_left)
                sum_data[bin0_min] += (data * (inv_area * delta_left) ** coef_power)

                sum_count[bin0_max] += (inv_area * delta_right)
                sum_data[bin0_max] += (data * (inv_area * delta_right) ** coef_power)

                if bin0_min + 1 < bin0_max:
                    for idx in range(bin0_min + 1, bin0_max):
                        sum_count[idx] += inv_area
                        sum_data[idx] += data * (inv_area ** coef_power)

        for idx in range(bins):
                if sum_count[idx] > epsilon:
                    out_merge[idx] = sum_data[idx] / sum_count[idx] / normalization_factor
                else:
                    out_merge[idx] = cdummy

    bin_centers = numpy.linspace(pos0_min + 0.5 * delta, pos0_max - 0.5 * delta, bins)

    return bin_centers, numpy.asarray(out_merge), numpy.asarray(sum_data), numpy.asarray(sum_count)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def histoBBox2d(weights,
                pos0,
                delta_pos0,
                pos1,
                delta_pos1,
                bins=(100, 36),
                pos0Range=None,
                pos1Range=None,
                dummy=None,
                delta_dummy=None,
                mask=None,
                dark=None,
                flat=None,
                solidangle=None,
                polarization=None,
                bint allow_pos0_neg=0,
                bint chiDiscAtPi=1,
                empty=0.0,
                double normalization_factor=1.0,
                int coef_power=1,
                bint clip_pos1=1):
    """
    Calculate 2D histogram of pos0(tth),pos1(chi) weighted by weights

    Splitting is done on the pixel's bounding box like fit2D


    :param weights: array with intensities
    :param pos0: 1D array with pos0: tth or q_vect
    :param delta_pos0: 1D array with delta pos0: max center-corner distance
    :param pos1: 1D array with pos1: chi
    :param delta_pos1: 1D array with max pos1: max center-corner distance, unused !
    :param bins: number of output bins (tth=100, chi=36 by default)
    :param pos0Range: minimum and maximum  of the 2th range
    :param pos1Range: minimum and maximum  of the chi range
    :param dummy: value for bins without pixels & value of "no good" pixels
    :param delta_dummy: precision of dummy value
    :param mask: array (of int8) with masked pixels with 1 (0=not masked)
    :param dark: array (of float32) with dark noise to be subtracted (or None)
    :param flat: array (of float32) with flat-field image
    :param solidangle: array (of float32) with solid angle corrections
    :param polarization: array (of float32) with polarization corrections
    :param chiDiscAtPi: boolean; by default the chi_range is in the range ]-pi,pi[ set to 0 to have the range ]0,2pi[
    :param empty: value of output bins without any contribution when dummy is None
    :param normalization_factor: divide the result by this value
    :param coef_power: set to 2 for variance propagation, leave to 1 for mean calculation
    :param clip_pos1: clip the azimuthal range to -pi/pi (or 0-2pi), set to False to deactivate behavior


    :return: I, bin_centers0, bin_centers1, weighted histogram(2D), unweighted histogram (2D)
    """

    cdef ssize_t bins0, bins1, i, j, idx
    cdef size_t size = weights.size
    assert pos0.size == size, "pos0.size == size"
    assert pos1.size == size, "pos1.size == size"
    assert delta_pos0.size == size, "delta_pos0.size == size"
    assert delta_pos1.size == size, "delta_pos1.size == size"
    try:
        bins0, bins1 = tuple(bins)
    except TypeError:
        bins0 = bins1 = bins
    if bins0 <= 0:
        bins0 = 1
    if bins1 <= 0:
        bins1 = 1
    cdef:
        #Related to data: single precision
        data_t[::1] cdata = numpy.ascontiguousarray(weights.ravel(), dtype=data_d)
        data_t[::1] cflat, cdark, cpolarization, csolidangle
        data_t cdummy, ddummy

        #related to positions: double precision
        position_t[::1] cpos0 = numpy.ascontiguousarray(pos0.ravel(), dtype=position_d)
        position_t[::1] dpos0 = numpy.ascontiguousarray(delta_pos0.ravel(), dtype=position_d)
        position_t[::1] cpos1 = numpy.ascontiguousarray(pos1.ravel(), dtype=position_d)
        position_t[::1] dpos1 = numpy.ascontiguousarray(delta_pos1.ravel(), dtype=position_d)
        position_t[::1] cpos0_upper = numpy.empty(size, dtype=position_d)
        position_t[::1] cpos0_lower = numpy.empty(size, dtype=position_d)
        position_t[::1] cpos1_upper = numpy.empty(size, dtype=position_d)
        position_t[::1] cpos1_lower = numpy.empty(size, dtype=position_d)
        acc_t[:, ::1] sum_data = numpy.zeros((bins0, bins1), dtype=acc_d)
        acc_t[:, ::1] sum_count = numpy.zeros((bins0, bins1), dtype=acc_d)
        data_t[:, ::1] out_merge = numpy.zeros((bins0, bins1), dtype=data_d)
        mask_t[::1] cmask

        position_t c0, c1, d0, d1
        position_t min0, max0, min1, max1, delta0, delta1
        position_t pos0_min, pos0_max, pos1_min, pos1_max, pos0_maxin, pos1_maxin
        position_t fbin0_min, fbin0_max, fbin1_min, fbin1_max,
        acc_t data, epsilon = 1e-10
        acc_t  inv_area, delta_up, delta_down, delta_right, delta_left
        ssize_t  bin0_max, bin0_min, bin1_max, bin1_min
        bint check_mask = False, check_dummy = False
        bint do_dark = False, do_flat = False, do_polarization = False, do_solidangle = False

    if mask is not None:
        assert mask.size == size, "mask size"
        check_mask = True
        cmask = numpy.ascontiguousarray(mask.ravel(), dtype=mask_d)

    if (dummy is not None) and delta_dummy is not None:
        check_dummy = True
        cdummy = float(dummy)
        ddummy = float(delta_dummy)
    elif (dummy is not None):
        cdummy = float(dummy)
    else:
        cdummy = float(empty)

    if dark is not None:
        assert dark.size == size, "dark current array size"
        do_dark = True
        cdark = numpy.ascontiguousarray(dark.ravel(), dtype=numpy.float32)
    if flat is not None:
        assert flat.size == size, "flat-field array size"
        do_flat = True
        cflat = numpy.ascontiguousarray(flat.ravel(), dtype=numpy.float32)
    if polarization is not None:
        do_polarization = True
        assert polarization.size == size, "polarization array size"
        cpolarization = numpy.ascontiguousarray(polarization.ravel(), dtype=numpy.float32)
    if solidangle is not None:
        do_solidangle = True
        assert solidangle.size == size, "Solid angle array size"
        csolidangle = numpy.ascontiguousarray(solidangle.ravel(), dtype=numpy.float32)

    pos0_min = cpos0[0]
    pos0_max = cpos0[0]
    pos1_min = cpos1[0]
    pos1_max = cpos1[0]

    with nogil:
        for idx in range(size):
            if (check_mask and cmask[idx]):
                continue
            c0 = cpos0[idx]
            d0 = dpos0[idx]
            min0 = c0 - d0
            max0 = c0 + d0
            c1 = cpos1[idx]
            d1 = dpos1[idx]
            min1 = c1 - d1
            max1 = c1 + d1
            if not allow_pos0_neg:
                if min0 < 0.0:
                    min0 = 0.0
                if max0 < 0.0:
                    max0 = 0.0
            if clip_pos1:
                if max1 > (2 - chiDiscAtPi) * pi:
                    max1 = (2 - chiDiscAtPi) * pi
                if min1 < (-chiDiscAtPi) * pi:
                    min1 = (-chiDiscAtPi) * pi
            cpos0_upper[idx] = max0
            cpos0_lower[idx] = min0
            cpos1_upper[idx] = max1
            cpos1_lower[idx] = min1
            if max0 > pos0_max:
                pos0_max = max0
            if min0 < pos0_min:
                pos0_min = min0
            if max1 > pos1_max:
                pos1_max = max1
            if min1 < pos1_min:
                pos1_min = min1

    if pos0Range is not None and len(pos0Range) > 1:
        pos0_min = min(pos0Range)
        pos0_maxin = max(pos0Range)
    else:
        pos0_maxin = pos0_max

    if (pos1Range is not None) and (len(pos1Range) > 1):
        pos1_min = min(pos1Range)
        pos1_maxin = max(pos1Range)
    else:
        pos1_maxin = pos1_max

    if (not allow_pos0_neg) and pos0_min < 0:
        pos0_min = 0

    pos0_max = calc_upper_bound(pos0_maxin)
    pos1_max = calc_upper_bound(pos1_maxin)

    delta0 = (pos0_max - pos0_min) / (<position_t> bins0)
    delta1 = (pos1_max - pos1_min) / (<position_t> bins1)

    with nogil:
        for idx in range(size):
            if (check_mask) and cmask[idx]:
                continue

            data = cdata[idx]
            if (check_dummy) and (fabs(data - cdummy) <= ddummy):
                continue

            if do_dark:
                data -= cdark[idx]
            if do_flat:
                data /= cflat[idx]
            if do_polarization:
                data /= cpolarization[idx]
            if do_solidangle:
                data /= csolidangle[idx]

            min0 = cpos0_lower[idx]
            max0 = cpos0_upper[idx]
            min1 = cpos1_lower[idx]
            max1 = cpos1_upper[idx]

            if (max0 < pos0_min) or (max1 < pos1_min) or (min0 > pos0_maxin) or (min1 > pos1_maxin):
                continue

            if min0 < pos0_min:
                min0 = pos0_min
            if min1 < pos1_min:
                min1 = pos1_min
            if max0 > pos0_maxin:
                max0 = pos0_maxin
            if max1 > pos1_maxin:
                max1 = pos1_maxin

            fbin0_min = get_bin_number(min0, pos0_min, delta0)
            fbin0_max = get_bin_number(max0, pos0_min, delta0)
            fbin1_min = get_bin_number(min1, pos1_min, delta1)
            fbin1_max = get_bin_number(max1, pos1_min, delta1)

            bin0_min = <ssize_t> fbin0_min
            bin0_max = <ssize_t> fbin0_max
            bin1_min = <ssize_t> fbin1_min
            bin1_max = <ssize_t> fbin1_max

            if bin0_min == bin0_max:
                # No spread along dim0
                if bin1_min == bin1_max:
                    # All pixel is within a single bin
                    sum_count[bin0_min, bin1_min] += 1.0
                    sum_data[bin0_min, bin1_min] += data
                else:
                    # spread on 2 or more bins in dim1
                    delta_down = (<acc_t> (bin1_min + 1)) - fbin1_min
                    delta_up = fbin1_max - (bin1_max)
                    inv_area = 1.0 / (fbin1_max - fbin1_min)

                    sum_count[bin0_min, bin1_min] += inv_area * delta_down
                    sum_data[bin0_min, bin1_min] += data * (inv_area * delta_down) ** coef_power

                    sum_count[bin0_min, bin1_max] += inv_area * delta_up
                    sum_data[bin0_min, bin1_max] += data * (inv_area * delta_up) ** coef_power
                    for j in range(bin1_min + 1, bin1_max):
                        sum_count[bin0_min, j] += inv_area
                        sum_data[bin0_min, j] += data * (inv_area) ** coef_power

            else:
                # spread on 2 or more bins in dim 0
                if bin1_min == bin1_max:
                    # All pixel fall inside the same bins in dim 1
                    inv_area = 1.0 / (fbin0_max - fbin0_min)

                    delta_left = (<acc_t> (bin0_min + 1)) - fbin0_min
                    sum_count[bin0_min, bin1_min] += inv_area * delta_left
                    sum_data[bin0_min, bin1_min] += data * (inv_area * delta_left) ** coef_power
                    delta_right = fbin0_max - (<acc_t> bin0_max)
                    sum_count[bin0_max, bin1_min] += inv_area * delta_right
                    sum_data[bin0_max, bin1_min] += data * (inv_area * delta_right) ** coef_power
                    for i in range(bin0_min + 1, bin0_max):
                            sum_count[i, bin1_min] += inv_area
                            sum_data[i, bin1_min] += data * (inv_area) ** coef_power
                else:
                    # spread on n pix in dim0 and m pixel in dim1:
                    inv_area = 1.0 / (fbin0_max - fbin0_min) * (fbin1_max - fbin1_min)

                    delta_left = (<acc_t> (bin0_min + 1)) - fbin0_min
                    delta_right = fbin0_max - (<acc_t> bin0_max)
                    delta_down = (<acc_t> (bin1_min + 1)) - fbin1_min
                    delta_up = fbin1_max - (<acc_t> bin1_max)

                    sum_count[bin0_min, bin1_min] += inv_area * delta_left * delta_down
                    sum_data[bin0_min, bin1_min] += data * (inv_area * delta_left * delta_down) ** coef_power

                    sum_count[bin0_min, bin1_max] += inv_area * delta_left * delta_up
                    sum_data[bin0_min, bin1_max] += data * (inv_area * delta_left * delta_up) ** coef_power

                    sum_count[bin0_max, bin1_min] += inv_area * delta_right * delta_down
                    sum_data[bin0_max, bin1_min] += data * (inv_area * delta_right * delta_down) ** coef_power

                    sum_count[bin0_max, bin1_max] += inv_area * delta_right * delta_up
                    sum_data[bin0_max, bin1_max] += data * (inv_area * delta_right * delta_up) ** coef_power
                    for i in range(bin0_min + 1, bin0_max):
                            sum_count[i, bin1_min] += inv_area * delta_down
                            sum_data[i, bin1_min] += data * (inv_area * delta_down) ** coef_power
                            for j in range(bin1_min + 1, bin1_max):
                                sum_count[i, j] += inv_area
                                sum_data[i, j] += data * (inv_area) ** coef_power
                            sum_count[i, bin1_max] += inv_area * delta_up
                            sum_data[i, bin1_max] += data * (inv_area * delta_up) ** coef_power
                    for j in range(bin1_min + 1, bin1_max):
                            sum_count[bin0_min, j] += inv_area * delta_left
                            sum_data[bin0_min, j] += data * (inv_area * delta_left) ** coef_power

                            sum_count[bin0_max, j] += inv_area * delta_right
                            sum_data[bin0_max, j] += data * (inv_area * delta_right) ** coef_power

        for i in range(bins0):
            for j in range(bins1):
                if sum_count[i, j] > epsilon:
                    out_merge[i, j] = sum_data[i, j] / sum_count[i, j] / normalization_factor
                else:
                    out_merge[i, j] = cdummy

    bin_centers0 = numpy.linspace(pos0_min + 0.5 * delta0, pos0_max - 0.5 * delta0, bins0)
    bin_centers1 = numpy.linspace(pos1_min + 0.5 * delta1, pos1_max - 0.5 * delta1, bins1)
    return (numpy.asarray(out_merge).T,
            bin_centers0,
            bin_centers1,
            numpy.asarray(sum_data).T,
            numpy.asarray(sum_count).T)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def histoBBox2d_ng(weights,
                   pos0,
                   delta_pos0,
                   pos1,
                   delta_pos1,
                   bins=(100, 36),
                   pos0Range=None,
                   pos1Range=None,
                   dummy=None,
                   delta_dummy=None,
                   mask=None,
                   variance=None,
                   dark=None,
                   flat=None,
                   solidangle=None,
                   polarization=None,
                   bint allow_pos0_neg=0,
                   bint chiDiscAtPi=1,
                   float empty=0.0,
                   double normalization_factor=1.0,
                   bint clip_pos1=1
                   ):
    """
    Calculate 2D histogram of pos0(tth),pos1(chi) weighted by weights

    Splitting is done on the pixel's bounding box, similar to fit2D
    New implementation with variance propagation

    :param weights: array with intensities
    :param pos0: 1D array with pos0: tth or q_vect
    :param delta_pos0: 1D array with delta pos0: max center-corner distance
    :param pos1: 1D array with pos1: chi
    :param delta_pos1: 1D array with max pos1: max center-corner distance, unused !
    :param bins: number of output bins (tth=100, chi=36 by default)
    :param pos0Range: minimum and maximum  of the 2th range
    :param pos1Range: minimum and maximum  of the chi range
    :param dummy: value for bins without pixels & value of "no good" pixels
    :param delta_dummy: precision of dummy value
    :param mask: array (of int8) with masked pixels with 1 (0=not masked)
    :param variance: variance associated with the weights
    :param dark: array (of float32) with dark noise to be subtracted (or None)
    :param flat: array (of float32) with flat-field image
    :param solidangle: array (of float32) with solid angle corrections
    :param polarization: array (of float32) with polarization corrections
    :param chiDiscAtPi: boolean; by default the chi_range is in the range ]-pi,pi[ set to 0 to have the range ]0,2pi[
    :param empty: value of output bins without any contribution when dummy is None
    :param normalization_factor: divide the result by this value
    :param clip_pos1: clip the azimuthal range to -pi/pi (or 0-2pi), set to False to deactivate behavior

    :return: named tuple with "signal", ["error"], "bins0", "bins1", "propagated"
    """

    cdef ssize_t bins0, bins1, i, j, idx
    cdef size_t size = weights.size
    assert pos0.size == size, "pos0.size == size"
    assert pos1.size == size, "pos1.size == size"
    assert delta_pos0.size == size, "delta_pos0.size == size"
    assert delta_pos1.size == size, "delta_pos1.size == size"
    try:
        bins0, bins1 = tuple(bins)
    except TypeError:
        bins0 = bins1 = bins
    if bins0 <= 0:
        bins0 = 1
    if bins1 <= 0:
        bins1 = 1
    cdef:
        # Related to data: single precision
        data_t[::1] cdata = numpy.ascontiguousarray(weights.ravel(), dtype=data_d)
        data_t[::1] cflat, cdark, cpolarization, csolidangle, cvariance
        data_t cdummy, ddummy=0.0

        # Related to positions: double precision
        position_t[::1] cpos0 = numpy.ascontiguousarray(pos0.ravel(), dtype=position_d)
        position_t[::1] dpos0 = numpy.ascontiguousarray(delta_pos0.ravel(), dtype=position_d)
        position_t[::1] cpos1 = numpy.ascontiguousarray(pos1.ravel(), dtype=position_d)
        position_t[::1] dpos1 = numpy.ascontiguousarray(delta_pos1.ravel(), dtype=position_d)
        position_t[::1] cpos0_upper = numpy.empty(size, dtype=position_d)
        position_t[::1] cpos0_lower = numpy.empty(size, dtype=position_d)
        position_t[::1] cpos1_upper = numpy.empty(size, dtype=position_d)
        position_t[::1] cpos1_lower = numpy.empty(size, dtype=position_d)
        acc_t[:, :, ::1] out_data = numpy.zeros((bins0, bins1, 4), dtype=acc_d)
        data_t[:, ::1] out_intensity = numpy.zeros((bins0, bins1), dtype=data_d)
        data_t[:, ::1] out_error
        mask_t[::1] cmask

        position_t c0, c1, d0, d1
        position_t min0, max0, min1, max1, delta0, delta1
        position_t pos0_min, pos0_max, pos1_min, pos1_max, pos0_maxin, pos1_maxin
        position_t fbin0_min, fbin0_max, fbin1_min, fbin1_max,
        acc_t norm
        acc_t  inv_area, delta_up, delta_down, delta_right, delta_left
        ssize_t  bin0_max, bin0_min, bin1_max, bin1_min
        bint check_mask = False, check_dummy = False, do_variance = False, is_valid
        bint do_dark = False, do_flat = False, do_polarization = False, do_solidangle = False
        preproc_t value

    if variance is not None:
        assert variance.size == size, "variance size"
        do_variance = True
        cvariance = numpy.ascontiguousarray(variance.ravel(), dtype=data_d)
        out_error = numpy.zeros((bins0, bins1), dtype=data_d)

    if mask is not None:
        assert mask.size == size, "mask size"
        check_mask = True
        cmask = numpy.ascontiguousarray(mask.ravel(), dtype=mask_d)

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
        cdark = numpy.ascontiguousarray(dark.ravel(), dtype=numpy.float32)
    if flat is not None:
        assert flat.size == size, "flat-field array size"
        do_flat = True
        cflat = numpy.ascontiguousarray(flat.ravel(), dtype=numpy.float32)
    if polarization is not None:
        do_polarization = True
        assert polarization.size == size, "polarization array size"
        cpolarization = numpy.ascontiguousarray(polarization.ravel(), dtype=numpy.float32)
    if solidangle is not None:
        do_solidangle = True
        assert solidangle.size == size, "Solid angle array size"
        csolidangle = numpy.ascontiguousarray(solidangle.ravel(), dtype=numpy.float32)

    pos0_min = cpos0[0]
    pos0_max = cpos0[0]
    pos1_min = cpos1[0]
    pos1_max = cpos1[0]
    with nogil:
        for idx in range(size):
            if (check_mask and cmask[idx]):
                continue
            c0 = cpos0[idx]
            d0 = dpos0[idx]
            min0 = c0 - d0
            max0 = c0 + d0
            c1 = cpos1[idx]
            d1 = dpos1[idx]
            min1 = c1 - d1
            max1 = c1 + d1
            if not allow_pos0_neg:
                if min0 < 0.0:
                    min0 = 0.0
                if max0 < 0.0:
                    max0 = 0.0
            if clip_pos1:
                if max1 > (2 - chiDiscAtPi) * pi:
                    max1 = (2 - chiDiscAtPi) * pi
                if min1 < (-chiDiscAtPi) * pi:
                    min1 = (-chiDiscAtPi) * pi
            cpos0_upper[idx] = max0
            cpos0_lower[idx] = min0
            cpos1_upper[idx] = max1
            cpos1_lower[idx] = min1
            if max0 > pos0_max:
                pos0_max = max0
            if min0 < pos0_min:
                pos0_min = min0
            if max1 > pos1_max:
                pos1_max = max1
            if min1 < pos1_min:
                pos1_min = min1

    if pos0Range is not None and len(pos0Range) > 1:
        pos0_min = min(pos0Range)
        pos0_maxin = max(pos0Range)
    else:
        pos0_maxin = pos0_max

    if (pos1Range is not None) and (len(pos1Range) > 1):
        pos1_min = min(pos1Range)
        pos1_maxin = max(pos1Range)
    else:
        pos1_maxin = pos1_max

    if (not allow_pos0_neg) and pos0_min < 0:
        pos0_min = 0

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
                                             variance=cvariance[idx] if do_variance else 0.0,
                                             dark=cdark[idx] if do_dark else 0.0,
                                             flat=cflat[idx] if do_flat else 1.0,
                                             solidangle=csolidangle[idx] if do_solidangle else 1.0,
                                             polarization=cpolarization[idx] if do_polarization else 1.0,
                                             absorption=1.0,
                                             mask=0, #previously checked
                                             dummy=cdummy,
                                             delta_dummy=ddummy,
                                             check_dummy=check_dummy,
                                             normalization_factor=normalization_factor,
                                             dark_variance=0.0)

            if not is_valid:
                continue

            min0 = cpos0_lower[idx]
            max0 = cpos0_upper[idx]
            min1 = cpos1_lower[idx]
            max1 = cpos1_upper[idx]

            if (max0 < pos0_min) or (max1 < pos1_min) or (min0 > pos0_maxin) or (min1 > pos1_maxin):
                continue

            if min0 < pos0_min:
                min0 = pos0_min
            if min1 < pos1_min:
                min1 = pos1_min
            if max0 > pos0_maxin:
                max0 = pos0_maxin
            if max1 > pos1_maxin:
                max1 = pos1_maxin

            fbin0_min = get_bin_number(min0, pos0_min, delta0)
            fbin0_max = get_bin_number(max0, pos0_min, delta0)
            fbin1_min = get_bin_number(min1, pos1_min, delta1)
            fbin1_max = get_bin_number(max1, pos1_min, delta1)

            bin0_min = <ssize_t> fbin0_min
            bin0_max = <ssize_t> fbin0_max
            bin1_min = <ssize_t> fbin1_min
            bin1_max = <ssize_t> fbin1_max

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
                norm = out_data[i, j, 2]
                if norm > 0.0:
                    out_intensity[i, j] = out_data[i, j, 0] / norm
                    if do_variance:
                        out_error[i, j] = sqrt(out_data[i, j, 1]) / norm
                else:
                    out_intensity[i, j] = empty
                    if do_variance:
                        out_error[i, j] = empty

    bin_centers0 = numpy.linspace(pos0_min + 0.5 * delta0, pos0_max - 0.5 * delta0, bins0)
    bin_centers1 = numpy.linspace(pos1_min + 0.5 * delta1, pos1_max - 0.5 * delta1, bins1)
    # TODO ... remove Integrate2dWithErrorResult namedtuple
    if do_variance:
        result = Integrate2dWithErrorResult(numpy.asarray(out_intensity).T,
                                            numpy.asarray(out_error).T,
                                            bin_centers0,
                                            bin_centers1,
                                            numpy.asarray(out_data).view(dtype=prop_d).reshape((bins0, bins1)).T)
    else:
        result = Integrate2dResult(numpy.asarray(out_intensity).T,
                                   bin_centers0,
                                   bin_centers1,
                                   numpy.asarray(out_data).view(dtype=prop_d).reshape((bins0, bins1)).T)

    return result
