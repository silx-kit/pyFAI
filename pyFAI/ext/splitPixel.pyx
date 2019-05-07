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

Splitting is done by full pixel splitting
Histogram (direct) implementation
"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "02/05/2019"
__status__ = "stable"
__license__ = "MIT"

include "regrid_common.pxi"


cimport cython
from cython cimport floating
import numpy
from libc.math cimport fabs, ceil, floor
import logging
logger = logging.getLogger(__name__)


cdef inline floating area4(floating a0,
                           floating a1,
                           floating b0,
                           floating b1,
                           floating c0,
                           floating c1,
                           floating d0,
                           floating d1) nogil:
    """
    Calculate the area of the ABCD polygon with 4 with corners:
    A(a0,a1)
    B(b0,b1)
    C(c0,c1)
    D(d0,d1)
    :return: area, i.e. 1/2 * (AC ^ BD)
    """
    return 0.5 * fabs(((c0 - a0) * (d1 - b1)) - ((c1 - a1) * (d0 - b0)))


cdef inline floating calc_area(floating I1, floating I2, floating slope, floating intercept) nogil:
    "Calculate the area between I1 and I2 of a line with a given slope & intercept"
    return 0.5 * ((I2 - I1) * (slope * (I2 + I1) + 2 * intercept))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void integrate(acc_t[::1] buffer, int buffer_size, position_t start0, position_t start1, position_t stop0, position_t stop1) nogil:
    "Integrate in a box a line between start and stop"

    if stop0 == start0:
        # slope is infinite, area is null: no change to the buffer
        return
    cdef position_t slope, intercept
    cdef int i, istart0 = <int> floor(start0), istop0 = <int> floor(stop0)
    slope = (stop1 - start1) / (stop0 - start0)
    intercept = start1 - slope * start0
    if buffer_size > istop0 == istart0 >= 0:
        buffer[istart0] += calc_area(start0, stop0, slope, intercept)
    else:
        if stop0 > start0:
                if 0 <= start0 < buffer_size:
                    buffer[istart0] += calc_area(start0, floor(start0 + 1), slope, intercept)
                for i in range(max(istart0 + 1, 0), min(istop0, buffer_size)):
                    buffer[i] += calc_area(i, i + 1, slope, intercept)
                if buffer_size > stop0 >= 0:
                    buffer[istop0] += calc_area(istop0, stop0, slope, intercept)
        else:
            if 0 <= start0 < buffer_size:
                buffer[istart0] += calc_area(start0, istart0, slope, intercept)
            for i in range(min(istart0, buffer_size) - 1, max(<int> floor(stop0), -1), -1):
                buffer[i] += calc_area(i + 1, i, slope, intercept)
            if buffer_size > stop0 >= 0:
                buffer[istop0] += calc_area(floor(stop0 + 1), stop0, slope, intercept)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def fullSplit1D(cnumpy.ndarray pos not None,
                cnumpy.ndarray weights not None,
                int bins=100,
                pos0Range=None,
                pos1Range=None,
                dummy=None,
                delta_dummy=None,
                mask=None,
                dark=None,
                flat=None,
                solidangle=None,
                polarization=None,
                float empty=0.0,
                double normalization_factor=1.0,
                int coef_power=1
                ):
    """
    Calculates histogram of pos weighted by weights

    Splitting is done on the pixel's bounding box like fit2D.
    No compromise for speed has been made here.


    :param pos: 3D array with pos0; Corner A,B,C,D; tth or chi
    :param weights: array with intensities
    :param bins: number of output bins
    :param pos0Range: minimum and maximum  of the 2th range
    :param pos1Range: minimum and maximum  of the chi range
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

    :return: 2theta, I, weighted histogram, unweighted histogram
    """
    cdef int  size = weights.size
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
        mask_t[::1] cmask
        data_t[::1] cflat, cdark, cpolarization, csolidangle
        acc_t[::1] buffer = numpy.zeros(bins, dtype=acc_d)

        data_t cdummy = 0, cddummy = 0, data = 0
        position_t inv_area = 0
        position_t pos0_min = 0, pos0_max = 0, pos0_maxin = 0, pos1_min = 0, pos1_max = 0, pos1_maxin = 0
        position_t area_pixel = 0, sum_area = 0, sub_area = 0, dpos = 0
        position_t a0 = 0, b0 = 0, c0 = 0, d0 = 0, max0 = 0, min0 = 0, a1 = 0, b1 = 0, c1 = 0, d1 = 0, max1 = 0, min1 = 0
        double epsilon = 1e-10

        bint check_pos1=False, check_mask=False, check_dummy=False, do_dark=False, do_flat=False, do_polarization=False, do_solidangle=False
        int i = 0, idx = 0, bin0_max = 0, bin0_min = 0

    if mask is not None:
        check_mask = True
        assert mask.size == size, "mask size"
        cmask = numpy.ascontiguousarray(mask.ravel(), dtype=mask_d)

    if pos0Range is not None and len(pos0Range) > 1:
        pos0_min = min(pos0Range)
        pos0_maxin = max(pos0Range)
    else:
        with nogil:
            for idx in range(size):
                if (check_mask) and (cmask[idx]):
                    pos0_max = pos0_min = cpos[idx, 0, 0]
                    pos1_max = pos1_min = cpos[idx, 0, 1]
                    break
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
                if max0 > pos0_max:
                    pos0_max = max0
                if min0 < pos0_min:
                    pos0_min = min0
                min1 = min(a1, b1, c1, d1)
                max1 = max(a1, b1, c1, d1)
                if max1 > pos1_max:
                    pos1_max = max1
                if min1 < pos1_min:
                    pos1_min = min1

            pos0_maxin = pos0_max
    if pos0_min < 0:
        pos0_min = 0
    pos0_max = calc_upper_bound(pos0_maxin)

    if pos1Range is not None and len(pos1Range) > 1:
        pos1_min = min(pos1Range)
        pos1_maxin = max(pos1Range)
        check_pos1 = True
    else:
        if min1 == max1 == 0:
            pos1_min = pos[:, :, 1].min()
            pos1_maxin = pos[:, :, 1].max()
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
            a1 = <  double > cpos[idx, 0, 1]
            b0 = get_bin_number(cpos[idx, 1, 0], pos0_min, dpos)
            b1 = <  double > cpos[idx, 1, 1]
            c0 = get_bin_number(cpos[idx, 2, 0], pos0_min, dpos)
            c1 = <  double > cpos[idx, 2, 1]
            d0 = get_bin_number(cpos[idx, 3, 0], pos0_min, dpos)
            d1 = <  double > cpos[idx, 3, 1]
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

            bin0_min = < int > floor(min0)
            bin0_max = < int > floor(max0)

            if bin0_min == bin0_max:
                # All pixel is within a single bin
                sum_count[bin0_min] += 1.0
                sum_data[bin0_min] += data

            # else we have pixel splitting.
            else:
                bin0_min = max(0, bin0_min)
                bin0_max = min(bins, bin0_max + 1)
                area_pixel = area4(a0, a1, b0, b1, c0, c1, d0, d1)
                inv_area = 1.0 / area_pixel

                integrate(buffer, bins, a0, a1, b0, b1)  # A-B
                integrate(buffer, bins, b0, b1, c0, c1)  # B-C
                integrate(buffer, bins, c0, c1, d0, d1)  # C-D
                integrate(buffer, bins, d0, d1, a0, a1)  # D-A

                # Distribute pixel area
                sum_area = 0.0
                for i in range(bin0_min, bin0_max):
                    sub_area = fabs(buffer[i])
                    sum_area += sub_area
                    sub_area = sub_area * inv_area
                    sum_count[i] += sub_area
                    sum_data[i] += (sub_area ** coef_power) * data

                # Check the total area:
                if fabs(sum_area - area_pixel) / area_pixel > 1e-6 and bin0_min != 0 and bin0_max != bins:
                    with gil:
                        print("area_pixel=%s area_sum=%s, Error= %s" % (area_pixel, sum_area, (area_pixel - sum_area) / area_pixel))
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


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def fullSplit2D(cnumpy.ndarray pos not None,
                cnumpy.ndarray weights not None,
                bins not None,
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
                float empty=0.0,
                double normalization_factor=1.0,
                int coef_power=1
                ):
    """
    Calculate 2D histogram of pos weighted by weights

    Splitting is done on the pixel's bounding box like fit2D


    :param pos: 3D array with pos0; Corner A,B,C,D; tth or chi
    :param weights: array with intensities
    :param bins: number of output bins int or 2-tuple of int
    :param pos0Range: minimum and maximum  of the 2th range
    :param pos1Range: minimum and maximum  of the chi range
    :param dummy: value for bins without pixels
    :param delta_dummy: precision of dummy value
    :param mask: array (of int8) with masked pixels with 1 (0=not masked)
    :param dark: array (of float64) with dark noise to be subtracted (or None)
    :param flat: array (of float64) with flat-field image
    :param polarization: array (of float64) with polarization correction
    :param solidangle: array (of float64)with solid angle corrections
    :param allow_pos0_neg: set to true to allow negative radial values.
    :param chiDiscAtPi: boolean; by default the chi_range is in the range ]-pi,pi[ set to 0 to have the range ]0,2pi[
    :param empty: value of output bins without any contribution when dummy is None
    :param normalization_factor: divide the valid result by this value
    :param coef_power: set to 2 for variance propagation, leave to 1 for mean calculation

    :return: I, edges0, edges1, weighted histogram(2D), unweighted histogram (2D)
    """

    cdef int bins0 = 0, bins1 = 0, size = weights.size
    if pos.ndim > 3:  # create a view
        pos = pos.reshape((-1, 4, 2))

    assert pos.shape[0] == size, "pos.shape[0] == size"
    assert pos.shape[1] == 4, "pos.shape[1] == 4: pos has 4 corners"
    assert pos.shape[2] == 2, "pos.shape[2] == 2: tth and chi"
    assert pos.ndim == 3, "pos.ndim == 3"
    try:
        bins0, bins1 = tuple(bins)
    except TypeError:
        bins0 = bins1 = < int > bins
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
        mask_t[:] cmask
        data_t[:] cflat, cdark, cpolarization, csolidangle
        bint check_mask = False, check_dummy = False, do_dark = False, do_flat = False, do_polarization = False, do_solidangle = False
        data_t cdummy = 0, cddummy = 0, data = 0
        position_t min0 = 0, max0 = 0, min1 = 0, max1 = 0, delta_right = 0, delta_left = 0, delta_up = 0, delta_down = 0, inv_area = 0
        position_t pos0_min = 0, pos0_max = 0, pos1_min = 0, pos1_max = 0, pos0_maxin = 0, pos1_maxin = 0
        position_t area_pixel = 0, fbin0_min = 0, fbin0_max = 0, fbin1_min = 0, fbin1_max = 0
        position_t a0 = 0, a1 = 0, b0 = 0, b1 = 0, c0 = 0, c1 = 0, d0 = 0, d1 = 0
        position_t epsilon = 1e-10
        position_t delta0, delta1
        int bin0_max = 0, bin0_min = 0, bin1_max = 0, bin1_min = 0, i = 0, j = 0, idx = 0

    if pos0Range is not None and len(pos0Range) == 2:
        pos0_min = min(pos0Range)
        pos0_maxin = max(pos0Range)
    else:
        pos0_min = pos[:, :, 0].min()
        pos0_maxin = pos[:, :, 0].max()
    pos0_max = calc_upper_bound(pos0_maxin)

    if pos1Range is not None and len(pos1Range) > 1:
        pos1_min = min(pos1Range)
        pos1_maxin = max(pos1Range)
    else:
        pos1_min = pos[:, :, 1].min()
        pos1_maxin = pos[:, :, 1].max()
    pos1_max = calc_upper_bound(pos1_maxin)

    delta0 = (pos0_max - pos0_min) / (<acc_t> (bins0))
    delta1 = (pos1_max - pos1_min) / (<acc_t> (bins1))

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
                if min0 < 0.0:
                    min0 = 0.0
                if max0 < 0.0:
                    max0 = 0.0

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

            bin0_min = < int > fbin0_min
            bin0_max = < int > fbin0_max
            bin1_min = < int > fbin1_min
            bin1_max = < int > fbin1_max

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
                    sum_data[bin0_min, bin1_min] += data * (inv_area * delta_down) ** coef_power

                    sum_count[bin0_min, bin1_max] += inv_area * delta_up
                    sum_data[bin0_min, bin1_max] += data * (inv_area * delta_up) ** coef_power
                    # if bin1_min + 1 < bin1_max:
                    for j in range(bin1_min + 1, bin1_max):
                            sum_count[bin0_min, j] += inv_area
                            sum_data[bin0_min, j] += data * inv_area ** coef_power

            else:
                # spread on more than 2 bins in dim 0
                if bin1_min == bin1_max:
                    # All pixel fall on 1 bins in dim 1
                    area_pixel = fbin0_max - fbin0_min
                    delta_left = (<acc_t> (bin0_min + 1)) - fbin0_min
                    inv_area = delta_left / area_pixel
                    sum_count[bin0_min, bin1_min] += inv_area
                    sum_data[bin0_min, bin1_min] += data * inv_area ** coef_power
                    delta_right = fbin0_max - (<acc_t> bin0_max)
                    inv_area = delta_right / area_pixel
                    sum_count[bin0_max, bin1_min] += inv_area
                    sum_data[bin0_max, bin1_min] += data * inv_area ** coef_power
                    inv_area = 1.0 / area_pixel
                    for i in range(bin0_min + 1, bin0_max):
                            sum_count[i, bin1_min] += inv_area
                            sum_data[i, bin1_min] += data * inv_area ** coef_power
                else:
                    # spread on n pix in dim0 and m pixel in dim1:
                    area_pixel = (fbin0_max - fbin0_min) * (fbin1_max - fbin1_min)
                    delta_left = (<acc_t> (bin0_min + 1.0)) - fbin0_min
                    delta_right = fbin0_max - (<double> bin0_max)
                    delta_down = (<acc_t> (bin1_min + 1.0)) - fbin1_min
                    delta_up = fbin1_max - (<acc_t> bin1_max)
                    inv_area = 1.0 / area_pixel

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
                                sum_data[i, j] += data * inv_area ** coef_power
                            sum_count[i, bin1_max] += inv_area * delta_up
                            sum_data[i, bin1_max] += data * (inv_area * delta_up) ** coef_power
                    for j in range(bin1_min + 1, bin1_max):
                            sum_count[bin0_min, j] += inv_area * delta_left
                            sum_data[bin0_min, j] += data * (inv_area * delta_left) ** coef_power

                            sum_count[bin0_max, j] += inv_area * delta_right
                            sum_data[bin0_max, j] += data * (inv_area * delta_right) ** coef_power

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


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def pseudoSplit2D_ng(pos not None,
                     weights not None,
                     bins not None,
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
                     double normalization_factor=1.0
                     ):
    """
    Calculate 2D histogram of pos weighted by weights

    Splitting is done on the pixel's bounding box, similar to fit2D
    New implementation with variance propagation



    :param pos: 3D array with pos0; Corner A,B,C,D; tth or chi
    :param weights: array with intensities
    :param bins: number of output bins int or 2-tuple of int
    :param pos0Range: minimum and maximum  of the 2th range
    :param pos1Range: minimum and maximum  of the chi range
    :param dummy: value for bins without pixels
    :param delta_dummy: precision of dummy value
    :param mask: array (of int8) with masked pixels with 1 (0=not masked)
    :param variance: variance associated with the weights
    :param dark: array (of float64) with dark noise to be subtracted (or None)
    :param flat: array (of float64) with flat-field image
    :param polarization: array (of float64) with polarization correction
    :param solidangle: array (of float64)with solid angle corrections
    :param allow_pos0_neg: set to true to allow negative radial values.
    :param chiDiscAtPi: boolean; by default the chi_range is in the range ]-pi,pi[ set to 0 to have the range ]0,2pi[
    :param empty: value of output bins without any contribution when dummy is None
    :param normalization_factor: divide the valid result by this value
    :param coef_power: set to 2 for variance propagation, leave to 1 for mean calculation

    :return: I, edges0, edges1, weighted histogram(2D), unweighted histogram (2D)
    """

    cdef int bins0 = 0, bins1 = 0, size = weights.size
    if pos.ndim > 3:  # create a view
        pos = pos.reshape((-1, 4, 2))

    assert pos.shape[0] == size, "pos.shape[0] == size"
    assert pos.shape[1] == 4, "pos.shape[1] == 4: pos has 4 corners"
    assert pos.shape[2] == 2, "pos.shape[2] == 2: tth and chi"
    assert pos.ndim == 3, "pos.ndim == 3"
    try:
        bins0, bins1 = tuple(bins)
    except TypeError:
        bins0 = bins1 = < int > bins
    if bins0 <= 0:
        bins0 = 1
    if bins1 <= 0:
        bins1 = 1
    cdef:
        position_t[:, :, ::1] cpos = numpy.ascontiguousarray(pos, dtype=position_d)
        data_t[::1] cdata = numpy.ascontiguousarray(weights.ravel(), dtype=data_d)
        acc_t[:, :, ::1] out_data = numpy.zeros((bins0, bins1, 4), dtype=acc_d)
        data_t[:, ::1] out_error, out_intensity = numpy.zeros((bins0, bins1), dtype=data_d)
        mask_t[:] cmask
        data_t[:] cflat, cdark, cpolarization, csolidangle, cvariance
        bint check_mask = False, check_dummy = False, do_dark = False, 
        bint do_flat = False, do_polarization = False, do_solidangle = False, 
        bint do_variance = False, is_valid
        data_t cdummy = 0, cddummy = 0, scale = 1
        position_t min0 = 0, max0 = 0, min1 = 0, max1 = 0, delta_right = 0, delta_left = 0, delta_up = 0, delta_down = 0, inv_area = 0
        position_t pos0_min = 0, pos0_max = 0, pos1_min = 0, pos1_max = 0, pos0_maxin = 0, pos1_maxin = 0
        position_t fbin0_min = 0, fbin0_max = 0, fbin1_min = 0, fbin1_max = 0
        position_t a0 = 0, a1 = 0, b0 = 0, b1 = 0, c0 = 0, c1 = 0, d0 = 0, d1 = 0
        position_t center0 = 0.0, center1 = 0.0, area, width, height,   
        position_t delta0, delta1, new_width, new_height, new_min0, new_max0, new_min1, new_max1
        int bin0_max = 0, bin0_min = 0, bin1_max = 0, bin1_min = 0, i = 0, j = 0, idx = 0
        acc_t norm
        preproc_t value

    if pos0Range is not None and len(pos0Range) == 2:
        pos0_min = min(pos0Range)
        pos0_maxin = max(pos0Range)
    else:
        pos0_min = pos[:, :, 0].min()
        pos0_maxin = pos[:, :, 0].max()
    pos0_max = calc_upper_bound(pos0_maxin)

    if pos1Range is not None and len(pos1Range) > 1:
        pos1_min = min(pos1Range)
        pos1_maxin = max(pos1Range)
    else:
        pos1_min = pos[:, :, 1].min()
        pos1_maxin = pos[:, :, 1].max()
    pos1_max = calc_upper_bound(pos1_maxin)

    delta0 = (pos0_max - pos0_min) / (<acc_t> (bins0))
    delta1 = (pos1_max - pos1_min) / (<acc_t> (bins1))

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
        do_variance = True
        cvariance = numpy.ascontiguousarray(variance.ravel(), dtype=data_d)
        out_error = numpy.zeros((bins0, bins1), dtype=data_d)

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
                if min0 < 0.0:
                    min0 = 0.0
                if max0 < 0.0:
                    max0 = 0.0

            if max1 > (2 - chiDiscAtPi) * pi:
                max1 = (2 - chiDiscAtPi) * pi
            if min1 < (-chiDiscAtPi) * pi:
                min1 = (-chiDiscAtPi) * pi

            is_valid = preproc_value_inplace(&value,
                                             cdata[idx],
                                             variance=cvariance[idx] if do_variance else 0.0,
                                             dark=cdark[idx] if do_dark else 0.0,
                                             flat=cflat[idx] if do_flat else 1.0,
                                             solidangle=csolidangle[idx] if do_solidangle else 1.0,
                                             polarization=cpolarization[idx] if do_polarization else 1.0,
                                             absorption=1.0,
                                             mask=cmask[idx] if check_mask else 0,
                                             dummy=cdummy,
                                             delta_dummy=cddummy,
                                             check_dummy=check_dummy,
                                             normalization_factor=normalization_factor,
                                             dark_variance=0.0)
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

            bin0_min = < int > fbin0_min
            bin0_max = < int > fbin0_max
            bin1_min = < int > fbin1_min
            bin1_max = < int > fbin1_max

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
