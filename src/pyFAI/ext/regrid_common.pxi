# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2023 European Synchrotron Radiation Facility, Grenoble, France
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

"""Common cdef constants and functions for preprocessing

Some are defined in the associated header file .pxd
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "12/11/2024"
__status__ = "stable"
__license__ = "MIT"


# Imports at the Python level
import cython
import numpy
import sys
from libc.math cimport ceil, floor, copysign, pow, fabs, M_PI, sqrt, log, NAN

include "math_common.pxi"

# Work around for issue similar to : https://github.com/pandas-dev/pandas/issues/16358

_numpy_1_12_py2_bug = ((sys.version_info.major == 2) and
                       ([1, 12] >= [int(i) for i in numpy.version.version.split(".", 2)[:2]]))

# Imports at the C level
from .isnan cimport isnan

# How position are stored
ctypedef float64_t position_t
position_d = numpy.float64

# How weights or data are stored
ctypedef float32_t data_t
data_d = numpy.float32

# how data are accumulated
ctypedef float64_t acc_t
acc_d = numpy.float64

# type of the mask:
ctypedef int8_t mask_t
mask_d = numpy.int8

# type of the indexes:
ctypedef int32_t index_t
index_d = numpy.int32

#type of the buffer:
ctypedef float32_t buffer_t
buffer_d = numpy.float32


cdef struct lut_t:
    index_t idx
    data_t coef

LUT_ITEMSIZE = int(sizeof(lut_t))

# Work around for issue similar to : https://github.com/pandas-dev/pandas/issues/16358
if _numpy_1_12_py2_bug:
    lut_d = numpy.dtype([(b"idx", index_d), (b"coef", data_d)])
else:
    lut_d = numpy.dtype([("idx", index_d), ("coef", numpy.float32)])

# Type used for propagating variance
if _numpy_1_12_py2_bug:
    prop_d = numpy.dtype([(b'signal', acc_d),
                          (b'variance', acc_d),
                          (b'norm', acc_d),
                          (b'count', acc_d)])
else:
    prop_d = numpy.dtype([('signal', acc_d),
                          ('variance', acc_d),
                          ('norm', acc_d),
                         ('count', acc_d)])

cdef:
    struct preproc_t:
        data_t signal
        data_t variance
        data_t norm
        data_t count

    float pi = <float> M_PI
    double twopi = 2.0 * M_PI
    float piover2 = <float> (pi * 0.5)
    float onef = <float> 1.0
    float zerof = <float> 1.0
    double EPS32 = (1.0 + numpy.finfo(numpy.float32).eps)


from collections import namedtuple
from ..containers import Integrate1dtpl, Integrate2dtpl, ErrorModel

Boundaries = namedtuple("Boundaries", "min0 max0 min1 max1")

@cython.cdivision(True)
cdef inline floating get_bin_number(floating x0, floating pos0_min, floating delta) noexcept nogil:
    """
    calculate the bin number for any point (as floating)

    :param x0: current position
    :param pos0_min: position minimum
    :param delta: bin width
    :return: bin number as floating point.
    """
    return (x0 - pos0_min) / delta


@cython.cdivision(True)
cdef inline floating calc_upper_bound(floating maximum_value) noexcept nogil:
    """Calculate the upper_bound for an histogram,
    given the maximum value of all the data.

    :param maximum_value: maximum value over all elements
    :return: the smallest 32 bit float greater than the maximum
    """
    return maximum_value * EPS32 if maximum_value > 0 else maximum_value / EPS32


cdef inline bint preproc_value_inplace(preproc_t* result,
                                       floating data,
                                       floating variance=0.0,
                                       floating dark=0.0,
                                       floating flat=1.0,
                                       floating solidangle=1.0,
                                       floating polarization=1.0,
                                       floating absorption=1.0,
                                       mask_t mask=0,
                                       floating dummy=0.0,
                                       floating delta_dummy=0.0,
                                       bint check_dummy=False,
                                       floating normalization_factor=1.0,
                                       floating dark_variance=0.0,
                                       int error_model=1,
                                       bint apply_normalization=False) noexcept nogil:
    """This is a Function in the C-space that performs the preprocessing
    for one data point


    :param result: the container for the result, i.e. output which contains (signal, variance, normalisation, count)
    :param data and variance: the raw value and the associated variance
    :param dark and dark_variance: the dark-noise and the associated variance to be subtracted (signal) or added (variance)
    :param flat, solidangle, polarization, absorption, normalization_factor: all normalization to be multiplied togeather
    :param dummy, delta_dummy, mask,check_dummy: controls the masking of the pixel
    :param normalization_factor: multiply normalization with this value
    :param error_model: 0 to diable, 1 for variance, 2 for Poisson model, ...
    :param apply_normalization: False by default, set to True to perform an unweighted average later on. WIP
    :return: isvalid, i.e. True if the pixel is worth further processing

    where the result is calculated this way:
    * signal = data-dark
    * variance = variance + dark_variance
    * norm = prod(all normalization)

    unless data are invalid (mask, nan, ...) where the result is all null.
    """
    cdef:
        floating signal, norm, count
        bint is_valid

    is_valid = (not isnan(data)) and (mask == 0)
    if is_valid and check_dummy:
        if delta_dummy == 0.0:
            is_valid = (data != dummy)
        else:
            is_valid = fabs(data - dummy) > delta_dummy

    if is_valid:
        if delta_dummy == 0.0:
            is_valid = (flat != dummy)
        else:
            is_valid = fabs(flat - dummy) > delta_dummy

    if is_valid:
        # Do not use "/=" as they mean reduction for cython
        if error_model==2: # Poisson error-model:
            variance = max(1.0, data)

        if dark:
            signal = data - dark
            if dark_variance:
                variance = variance + dark_variance
        else:
            signal = data
        norm = normalization_factor * flat * polarization * solidangle * absorption

        if (isnan(signal) or isnan(norm) or isnan(variance) or (norm == 0)):
            signal = 0.0
            variance = 0.0
            norm = 0.0
            count = 0.0
            is_valid = False
        else:
            count = 1.0
            if apply_normalization:
                signal /= norm
                variance /= norm*norm
                norm = 1.0
    else:
        signal = 0.0
        variance = 0.0
        norm = 0.0
        count = 0.0
    result.signal = signal
    result.variance = variance
    result.norm = norm
    result.count = count
    return is_valid


cdef inline void update_1d_accumulator(acc_t[:, ::1] out_data,
                                       int bin,
                                       preproc_t value,
                                       double weight=1.0,
                                       int error_model=0) noexcept nogil:
    """Update a 1D array at given position with the proper values

    :param out_data: output 1D+(,4) accumulator
    :param bin: in which bin assign this data
    :param value: 4-uplet with (signal, variance, nomalisation, count)
    :param weight: weight associated with this value
    :param error_model: 0:disable, 1:variance, 2: poisson, 3:azimuhal
    :return: Nothing
    """
    cdef:
        double weight2 = weight * weight
        acc_t omega_A, omega_B, omega2_A, omega2_B, w, w2, omega_AB, b, delta1, delta2
        acc_t sum_sig = out_data[bin, 0]
        acc_t sum_var = out_data[bin, 1]
        acc_t sum_nrm = out_data[bin, 2]
        acc_t sum_cnt = out_data[bin, 3]
        acc_t sum_nrm2 = out_data[bin, 4]

    w = weight * value.norm
    w2 = w * w
    if error_model == 3:
        if sum_nrm2 > 0.0:
            # Inspired from https://dbs.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf
            # Not correct, Inspired by VV_{A+b} = VV_A + ω²·(b-V_A/Ω_A)·(b-V_{A+b}/Ω_{A+b})
            # Emprically validated against 2-pass implementation in Python/scipy-sparse
            if value.norm:
                omega_A = sum_nrm
                omega2_A = sum_nrm2
                omega2_B = w2
                sum_nrm += w
                sum_nrm2 += w2

                # VV_{AUb} = VV_A + ω_b^2 * (b-<A>) * (b-<AUb>)
                b = value.signal / value.norm
                delta1 = sum_sig/omega_A - b
                sum_sig += weight * value.signal
                delta2 = sum_sig / sum_nrm - b
                sum_var += omega2_B * delta1 * delta2


        else:
            sum_sig = weight * value.signal
            sum_nrm = w
            sum_nrm2 = w2
    else:
        sum_sig += value.signal * weight
        sum_var += value.variance * weight2
        sum_nrm += w
        # if out_data.shape[1] == 5: #Σ c²·ω²
        sum_nrm2 += w2
    sum_cnt += value.count * weight

    out_data[bin, 0] = sum_sig
    out_data[bin, 1] = sum_var
    out_data[bin, 2] = sum_nrm
    out_data[bin, 3] = sum_cnt
    out_data[bin, 4] = sum_nrm2


@cython.boundscheck(False)
cdef inline void update_2d_accumulator(acc_t[:, :, ::1] out_data,
                                       int bin0,
                                       int bin1,
                                       preproc_t value,
                                       double weight=1.0) noexcept nogil:
    """Update a 2D array at given position with the proper values

    :param out_data: 2D+1 accumulator
    :param bin0, bin1: where to assign data
    :return: Nothing
    """
    cdef double w2 = weight * weight
    out_data[bin0, bin1, 0] += value.signal * weight
    out_data[bin0, bin1, 1] += value.variance * w2  # Important for variance propagation
    out_data[bin0, bin1, 2] += value.norm * weight
    out_data[bin0, bin1, 3] += value.count * weight
    # if out_data.shape[2] == 5: #Σ c²·ω²
    out_data[bin0, bin1, 4] += value.norm * value.norm * w2 # used to calculate the standard deviation


cdef inline floating area4p(floating a0,
                            floating a1,
                            floating b0,
                            floating b1,
                            floating c0,
                            floating c1,
                            floating d0,
                            floating d1) noexcept nogil:
    """
    Calculate the _APPROXIMATE_ area of the ABCD considering the parallelogram formula with 4 with corners:
    A(a0,a1)
    B(b0,b1)
    C(c0,c1)
    D(d0,d1)

    :return: approximative area 1/2 * (AC ^ BD)
    """
    return 0.5 * ((c0 - a0) * (d1 - b1)) - ((c1 - a1) * (d0 - b0))

cdef inline floating area4(floating a0,
                           floating a1,
                           floating b0,
                           floating b1,
                           floating c0,
                           floating c1,
                           floating d0,
                           floating d1) noexcept nogil:
    """
    Calculate the area of the ABCD polygon with 4 with corners:
    A(a0,a1)
    B(b0,b1)
    C(c0,c1)
    D(d0,d1)

    see http://villemin.gerard.free.fr/GeomLAV/Carre/QuadAire.htm

    :return: area, i.e. a complicated formule, not 1/2 * (AC ^ BD)
    """
    cdef floating a, b, c, d, p, q

    a = sqrt((b1-a1)**2 + (b0-a0)**2) # AB
    b = sqrt((b1-c1)**2 + (b0-c0)**2) # BC
    c = sqrt((c1-d1)**2 + (c0-d0)**2) # CD
    d = sqrt((d1-a1)**2 + (d0-a0)**2) # DA
    p = sqrt((c1-a1)**2 + (c0-a0)**2) # AC
    q = sqrt((b1-d1)**2 + (b0-d0)**2) # BD
    return 0.25 * sqrt(4*p*p*q*q - (b*b+d*d-a*a-c*c)**2)

def _sp_area4(floating a0, floating a1, floating b0, floating b1, floating c0, floating c1, floating d0, floating d1):
    return area4(a0, a1, b0, b1, c0, c1, d0, d1)

cdef inline position_t _recenter_helper(position_t azim, bint chiDiscAtPi) noexcept nogil:
    """Helper function
    """
    if (chiDiscAtPi and azim<0) or (not chiDiscAtPi and azim<pi):
        return azim + twopi
    else:
        return azim


cdef inline position_t _recenter(position_t[:, ::1] pixel, bint chiDiscAtPi) noexcept nogil:
    cdef position_t a0, a1, b0, b1, c0, c1, d0, d1, center1, area, hi
    a0 = pixel[0, 0]
    a1 = pixel[0, 1]
    b0 = pixel[1, 0]
    b1 = pixel[1, 1]
    c0 = pixel[2, 0]
    c1 = pixel[2, 1]
    d0 = pixel[3, 0]
    d1 = pixel[3, 1]
    area = area4p(a0, a1, b0, b1, c0, c1, d0, d1) # check if the quad is crossed
    if area>0:
        # area are expected to be negative except for pixel on the boundary
        a1 = _recenter_helper(a1, chiDiscAtPi)
        b1 = _recenter_helper(b1, chiDiscAtPi)
        c1 = _recenter_helper(c1, chiDiscAtPi)
        d1 = _recenter_helper(d1, chiDiscAtPi)
        center1 = 0.25 * (a1 + b1 + c1 + d1)
        hi = pi if chiDiscAtPi else twopi
        if center1>hi:
            a1 -= twopi
            b1 -= twopi
            c1 -= twopi
            d1 -= twopi
        pixel[0, 1] = a1
        pixel[1, 1] = b1
        pixel[2, 1] = c1
        pixel[3, 1] = d1
        area = area4p(a0, a1, b0, b1, c0, c1, d0, d1)
    return area

def recenter(position_t[:, ::1] pixel, bint chiDiscAtPi=1):
    """This function checks the pixel to be on the azimuthal discontinuity
    via the sign of its algebric area and recenters the corner coordinates in a
    consistent manner to have all azimuthal coordinate in

    Nota: the returned area is negative since the positive area indicate the pixel is on the discontinuity.

    :param pixel: 4x2 array with radius, azimuth for the 4 corners. MODIFIED IN PLACE !!!
    :param chiDiscAtPi: set to 0 to indicate the range goes from 0-2π instead of the default -π:π
    :return: signed area (approximate & negative)
    """
    return _recenter(pixel, chiDiscAtPi)


cdef inline any_t _clip(any_t value, any_t min_val, any_t max_val) noexcept nogil:
    "Limits the value to bounds"
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    else:
        return value

def clip(value,  min_val, int max_val):
    """Limits the value to bounds

    :param value: the value to clip
    :param min_value: the lower bound
    :param max_value: the upper bound
    :return: clipped value in the requested range

    """
    return _clip(<double>value, <double> min_val, <double> max_val)


cdef inline floating _calc_area(floating I1, floating I2, floating slope, floating intercept) noexcept nogil:
    #return 0.5 * (I2 - I1) * (slope * (I2 + I1) + 2 * intercept)
    return (I2 - I1) * (0.5 * slope * (I2 + I1) + intercept)

def calc_area(I1, I2, slope, intercept):
    "Calculate the area between I1 and I2 of a line with a given slope & intercept"
    return _calc_area(<position_t> I1, <position_t> I2, <position_t> slope, <position_t> intercept)


cdef inline void _integrate1d(buffer_t[::1] buffer,
                              floating start0, floating start1,
                              floating stop0, floating stop1) noexcept nogil:
    """"Integrate in a box a segment between start and stop

    :param buffer: Buffer which is modified in place
    :param start0: position of the start in dim0
    :param start1: position of the start in dim1
    :param stop0: position of the end of segment in dim0
    :param stop1: position of the end of segment in dim1
    """
    cdef:
        floating slope, intercept
        Py_ssize_t i, istart0, istop0

    if stop0 == start0:
        # slope is infinite, area is null: no change to the buffer
        return

    buffer_size = buffer.shape[0]
    istart0 = <Py_ssize_t> floor(start0)
    istop0 = <Py_ssize_t> floor(stop0)

    slope = (stop1 - start1) / (stop0 - start0)
    intercept = start1 - slope * start0

    if buffer_size > istop0 == istart0 >= 0:
        buffer[istart0] += _calc_area(start0, stop0, slope, intercept)
    else:
        if stop0 > start0:
                if 0 <= start0 < buffer_size:
                    buffer[istart0] += _calc_area(start0, floor(start0 + 1), slope, intercept)
                for i in range(max(istart0 + 1, 0), min(istop0, buffer_size)):
                    buffer[i] += _calc_area(<floating>i, <floating>(i + 1), slope, intercept)
                if buffer_size > stop0 >= 0:
                    buffer[istop0] += _calc_area(<floating> istop0, stop0, slope, intercept)
        else:
            if 0 <= start0 < buffer_size:
                buffer[istart0] += _calc_area(start0, <floating> istart0, slope, intercept)
            for i in range(min(istart0, buffer_size) - 1, max(<Py_ssize_t> floor(stop0), -1), -1):
                buffer[i] += _calc_area(<floating>(i + 1), <floating>i, slope, intercept)
            if buffer_size > stop0 >= 0:
                buffer[istop0] += _calc_area(floor(stop0 + 1), stop0, slope, intercept)

def _sp_integrate1d(buffer_t[::1] buffer,
                    floating start0, floating start1,
                    floating stop0, floating stop1):
    _integrate1d(buffer, start0, start1, stop0, stop1)

cdef inline void _integrate2d(buffer_t[:, ::1] box,
                              floating start0, floating start1,
                              floating stop0, floating stop1) noexcept nogil:
    """Integrate in a box a line between start and stop0, line defined by its slope & intercept

    :param box: buffer with the relative area. Gets modified in place
    :param start0: coodinate of the start of the segment in dim0
    :param start1: coodinate of the start of the segment in dim1
    :param start0: coodinate of the end of the segment in dim0
    :param start0: coodinate of the end of the segment in dim1

    """
    cdef:
        Py_ssize_t i, h = 0
        float P, dP, segment_area, abs_area, dA
        float slope, intercept
        # , sign

    if start0 == stop0:
        return
    else:
        slope = (stop1 - start1) / (stop0 - start0)
        intercept = stop1 - slope*stop0

    if start0 < stop0:  # positive contribution
        P = ceil(start0)
        dP = P - start0
        if P > stop0:  # start0 and stop0 are in the same unit
            segment_area = _calc_area(start0, stop0, slope, intercept)
            if segment_area != 0.0:
                abs_area = fabs(segment_area)
                dA = (stop0 - start0)  # always positive
                h = 0
                while abs_area > 0:
                    if dA > abs_area:
                        dA = abs_area
                        abs_area = -1
                    box[(<Py_ssize_t> start0), h] += copysign(dA, segment_area)
                    abs_area -= dA
                    h += 1
        else:
            if dP > 0:
                segment_area = _calc_area(start0, P, slope, intercept)
                if segment_area != 0.0:
                    abs_area = fabs(segment_area)
                    h = 0
                    dA = dP
                    while abs_area > 0:
                        if dA > abs_area:
                            dA = abs_area
                            abs_area = -1
                        box[(<Py_ssize_t> P) - 1, h] += copysign(dA, segment_area)
                        abs_area -= dA
                        h += 1
            # subsection P1->Pn
            for i in range((<Py_ssize_t> floor(P)), (<Py_ssize_t> floor(stop0))):
                segment_area = _calc_area(<floating> i, <floating> (i + 1), slope, intercept)
                if segment_area != 0:
                    abs_area = fabs(segment_area)
                    h = 0
                    dA = 1.0
                    while abs_area > 0:
                        if dA > abs_area:
                            dA = abs_area
                            abs_area = -1
                        box[i, h] += copysign(dA, segment_area)
                        abs_area -= dA
                        h += 1
            # Section Pn->B
            P = floor(stop0)
            dP = stop0 - P
            if dP > 0:
                segment_area = _calc_area(P, stop0, slope, intercept)
                if segment_area != 0:
                    abs_area = fabs(segment_area)
                    h = 0
                    dA = fabs(dP)
                    while abs_area > 0:
                        if dA > abs_area:
                            dA = abs_area
                            abs_area = -1
                        box[(<Py_ssize_t> P), h] += copysign(dA, segment_area)
                        abs_area -= dA
                        h += 1
    elif start0 > stop0:  # negative contribution. Nota if start0==stop0: no contribution
        P = floor(start0)
        if stop0 > P:  # start0 and stop0 are in the same unit
            segment_area = _calc_area(start0, stop0, slope, intercept)
            if segment_area != 0:
                abs_area = fabs(segment_area)
                # sign = segment_area / abs_area
                dA = (start0 - stop0)  # always positive
                h = 0
                while abs_area > 0:
                    if dA > abs_area:
                        dA = abs_area
                        abs_area = -1
                    box[(<Py_ssize_t> start0), h] += copysign(dA, segment_area)
                    abs_area -= dA
                    h += 1
        else:
            dP = P - start0
            if dP < 0:
                segment_area = _calc_area(start0, P, slope, intercept)
                if segment_area != 0:
                    abs_area = fabs(segment_area)
                    h = 0
                    dA = fabs(dP)
                    while abs_area > 0:
                        if dA > abs_area:
                            dA = abs_area
                            abs_area = -1
                        box[(<Py_ssize_t> P), h] += copysign(dA, segment_area)
                        abs_area -= dA
                        h += 1
            # subsection P1->Pn
            for i in range((<Py_ssize_t> start0), (<Py_ssize_t> ceil(stop0)), -1):
                segment_area = _calc_area(<floating> i, <floating> (i - 1), slope, intercept)
                if segment_area != 0:
                    abs_area = fabs(segment_area)
                    h = 0
                    dA = 1
                    while abs_area > 0:
                        if dA > abs_area:
                            dA = abs_area
                            abs_area = -1
                        box[i - 1, h] += copysign(dA, segment_area)
                        abs_area -= dA
                        h += 1
            # Section Pn->B
            P = ceil(stop0)
            dP = stop0 - P
            if dP < 0:
                segment_area = _calc_area(P, stop0, slope, intercept)
                if segment_area != 0:
                    abs_area = fabs(segment_area)
                    h = 0
                    dA = fabs(dP)
                    while abs_area > 0:
                        if dA > abs_area:
                            dA = abs_area
                            abs_area = -1
                        box[(<Py_ssize_t> stop0), h] += copysign(dA, segment_area)
                        abs_area -= dA
                        h += 1
