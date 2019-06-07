# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2011-2018 European Synchrotron Radiation Facility, Grenoble, France
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

"""
Contains a preprocessing function in charge of the dark-current subtraction,
flat-field normalization... taking care of masked values and normalization.
"""

__author__ = "Jerome Kieffer"
__license__ = "MIT"
__date__ = "17/05/2019"
__copyright__ = "2011-2018, ESRF"
__contact__ = "jerome.kieffer@esrf.fr"

include "regrid_common.pxi"

import cython
from cython.parallel import prange


from libc.math cimport fabs
from .isnan cimport isnan
from cython cimport floating


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef floating[::1]c1_preproc(floating[::1] data,
                             floating[::1] dark=None,
                             floating[::1] flat=None,
                             floating[::1] solidangle=None,
                             floating[::1] polarization=None,
                             floating[::1] absorption=None,
                             any_int_t[::1] mask=None,
                             floating dummy=0.0,
                             floating delta_dummy=0.0,
                             bint check_dummy=False,
                             floating normalization_factor=1.0,
                             ) nogil:
    """Common preprocessing step for all routines: C-implementation

    :param data: raw value, as a numpy array, 1D or 2D
    :param dark: array containing the value of the dark noise, to be subtracted
    :param flat: Array containing the flatfield image. It is also checked for dummies if relevant.
    :param solidangle: the value of the solid_angle. This processing may be performed during the rebinning instead. left for compatibility
    :param polarization: Correction for polarization of the incident beam
    :param absorption: Correction for absorption in the sensor volume
    :param mask: array non null  where data should be ignored
    :param dummy: value of invalid data
    :param delta_dummy: precision for invalid data
    :param normalization_factor: final value is divided by this

    NaN are always considered as invalid

    if neither empty nor dummy is provided, empty pixels are 0
    """
    cdef:
        int size, i
        bint check_mask, do_dark, do_flat, do_solidangle, do_absorption,
        bint do_polarization, is_valid
        floating[::1] result
        floating one_value, one_num, one_den, one_flat

    with gil:
        size = data.size
        do_dark = dark is not None
        do_flat = flat is not None
        do_solidangle = solidangle is not None
        do_absorption = absorption is not None
        do_polarization = polarization is not None
        check_mask = mask is not None
        result = numpy.zeros_like(data)

    for i in prange(size, nogil=True, schedule="static"):
        one_num = data[i]
        one_den = normalization_factor
        is_valid = not isnan(one_num)
        if is_valid and check_mask:
            is_valid = (mask[i] == 0)
        if is_valid and check_dummy:
            if delta_dummy == 0:
                is_valid = (one_num != dummy)
            else:
                is_valid = fabs(one_num - dummy) > delta_dummy

        if is_valid and do_flat:
            one_flat = flat[i]
            if delta_dummy == 0:
                is_valid = (one_flat != dummy)
            else:
                is_valid = fabs(one_flat - dummy) > delta_dummy

        if is_valid:
            # Do not use "/=" as they mean reduction for cython
            if do_dark:
                one_num = one_num - dark[i]
            if do_flat:
                one_den = one_den * one_flat
            if do_polarization:
                one_den = one_den * polarization[i]
            if do_solidangle:
                one_den = one_den * solidangle[i]
            if do_absorption:
                one_den = one_den * absorption[i]
            if (isnan(one_num) or isnan(one_den) or (one_den == 0)):
                result[i] += dummy
            else:
                result[i] += one_num / one_den
        else:
            result[i] += dummy
    return result


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef floating[:, ::1]c2_preproc(floating[::1] data,
                                floating[::1] dark=None,
                                floating[::1] flat=None,
                                floating[::1] solidangle=None,
                                floating[::1] polarization=None,
                                floating[::1] absorption=None,
                                any_int_t[::1] mask=None,
                                floating dummy=0,
                                floating delta_dummy=0,
                                bint check_dummy=False,
                                floating normalization_factor=1.0,
                                ) nogil:
    """Common preprocessing step for all routines: C-implementation
    with split_result without variance

    :param data: raw value, as a numpy array, 1D or 2D
    :param dark: array containing the value of the dark noise, to be subtracted
    :param flat: Array containing the flatfield image. It is also checked for dummies if relevant.
    :param solidangle: the value of the solid_angle. This processing may be performed during the rebinning instead. left for compatibility
    :param polarization: Correction for polarization of the incident beam
    :param absorption: Correction for absorption in the sensor volume
    :param mask: array non null  where data should be ignored
    :param dummy: value of invalid data
    :param delta_dummy: precision for invalid data
    :param normalization_factor: final value is divided by this

    NaN are always considered as invalid

    Empty pixels are 0 both num and den
    """
    cdef:
        int size, i
        bint check_mask, do_dark, do_flat, do_solidangle, do_absorption, do_polarization
        bint is_valid
        floating[:, ::1] result
        floating one_num, one_result, one_flat, one_den

    with gil:
        size = data.size
        do_dark = dark is not None
        do_flat = flat is not None
        do_solidangle = solidangle is not None
        do_absorption = absorption is not None
        do_polarization = polarization is not None
        check_mask = mask is not None
        result = numpy.zeros((size, 2), dtype=numpy.asarray(data).dtype)

    for i in prange(size, nogil=True, schedule="static"):
        one_num = data[i]
        one_den = normalization_factor
        is_valid = not isnan(one_num)
        if is_valid and check_mask:
            is_valid = (mask[i] == 0)
        if is_valid and check_dummy:
            if delta_dummy == 0:
                is_valid = (one_num != dummy)
            else:
                is_valid = fabs(one_num - dummy) > delta_dummy

        if is_valid and do_flat:
            one_flat = flat[i]
            if delta_dummy == 0:
                is_valid = (one_flat != dummy)
            else:
                is_valid = fabs(one_flat - dummy) > delta_dummy

        if is_valid:
            # Do not use "/=" as they mean reduction for cython
            if do_dark:
                one_num = one_num - dark[i]
            if do_flat:
                one_den = one_den * flat[i]
            if do_polarization:
                one_den = one_den * polarization[i]
            if do_solidangle:
                one_den = one_den * solidangle[i]
            if do_absorption:
                one_den = one_den * absorption[i]
            if (isnan(one_num) or isnan(one_den) or (one_den == 0)):
                one_num = 0.0
                one_den = 0.0
        else:
            one_num = 0.0
            one_den = 0.0

        result[i, 0] += one_num
        result[i, 1] += one_den
    return result


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef floating[:, ::1]cp_preproc(floating[::1] data,
                                floating[::1] dark=None,
                                floating[::1] flat=None,
                                floating[::1] solidangle=None,
                                floating[::1] polarization=None,
                                floating[::1] absorption=None,
                                any_int_t[::1] mask=None,
                                floating dummy=0.0,
                                floating delta_dummy=0.0,
                                bint check_dummy=False,
                                floating normalization_factor=1.0,
                                ) nogil:
    """Common preprocessing step for all routines: C-implementation
    with split_result assuming a poissonian distribution

    :param data: raw value, as a numpy array, 1D or 2D
    :param dark: array containing the value of the dark noise, to be subtracted
    :param flat: Array containing the flatfield image. It is also checked for dummies if relevant.
    :param solidangle: the value of the solid_angle. This processing may be performed during the rebinning instead. left for compatibility
    :param polarization: Correction for polarization of the incident beam
    :param absorption: Correction for absorption in the sensor volume
    :param mask: array non null  where data should be ignored
    :param dummy: value of invalid data
    :param delta_dummy: precision for invalid data
    :param normalization_factor: final value is divided by this

    NaN are always considered as invalid

    Empty pixels are 0.0 for both signal, variance and normalization
    """
    cdef:
        int size, i
        bint check_mask, do_dark, do_flat, do_solidangle, do_absorption, do_polarization
        bint is_valid
        floating[:, ::1] result
        floating one_num, one_result, one_flat, one_den, one_var

    with gil:
        size = data.size
        do_dark = dark is not None
        do_flat = flat is not None
        do_solidangle = solidangle is not None
        do_absorption = absorption is not None
        do_polarization = polarization is not None
        check_mask = mask is not None
        result = numpy.zeros((size, 3), dtype=numpy.asarray(data).dtype)

    for i in prange(size, nogil=True, schedule="static"):
        one_num = one_var = data[i]
        one_den = normalization_factor

        is_valid = not isnan(one_num)
        if is_valid and check_mask:
            is_valid = (mask[i] == 0)
        if is_valid and check_dummy:
            if delta_dummy == 0:
                is_valid = (one_num != dummy)
            else:
                is_valid = fabs(one_num - dummy) > delta_dummy

        if is_valid and do_flat:
            one_flat = flat[i]
            if delta_dummy == 0:
                is_valid = (one_flat != dummy)
            else:
                is_valid = fabs(one_flat - dummy) > delta_dummy

        if is_valid:
            # Do not use "+=" as they mean reduction for cython
            if do_dark:
                one_num = one_num - dark[i]
                one_var = one_var + dark[i]
            if do_flat:
                one_den = one_den * flat[i]
            if do_polarization:
                one_den = one_den * polarization[i]
            if do_solidangle:
                one_den = one_den * solidangle[i]
            if do_absorption:
                one_den = one_den * absorption[i]
            if (isnan(one_num) or isnan(one_den) or isnan(one_var) or (one_den == 0)):
                one_num = 0.0
                one_var = 0.0
                one_den = 0.0
        else:
            one_num = 0.0
            one_var = 0.0
            one_den = 0.0

        result[i, 0] += one_num
        result[i, 1] += one_var
        result[i, 2] += one_den
    return result


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef floating[:, ::1]c3_preproc(floating[::1] data,
                                floating[::1] dark=None,
                                floating[::1] flat=None,
                                floating[::1] solidangle=None,
                                floating[::1] polarization=None,
                                floating[::1] absorption=None,
                                any_int_t[::1] mask=None,
                                floating dummy=0.0,
                                floating delta_dummy=0.0,
                                bint check_dummy=False,
                                floating normalization_factor=1.0,
                                floating[::1] variance=None,
                                floating[::1] dark_variance=None,
                                ) nogil:
    """Common preprocessing step for all routines: C-implementation
    with split_result with variance in second position: (signal, variance, normalization)

    :param data: raw value, as a numpy array, 1D or 2D
    :param dark: array containing the value of the dark noise, to be subtracted
    :param flat: Array containing the flatfield image. It is also checked for dummies if relevant.
    :param solidangle: the value of the solid_angle. This processing may be performed during the rebinning instead. left for compatibility
    :param polarization: Correction for polarization of the incident beam
    :param absorption: Correction for absorption in the sensor volume
    :param mask: array non null  where data should be ignored
    :param dummy: value of invalid data
    :param delta_dummy: precision for invalid data
    :param normalization_factor: final value is divided by this, settles on the denominator
    :param variance: variance of the data
    :param dark_variance: variance of the dark
    NaN are always considered as invalid

    Empty pixels are 0.0 for both signal, variance and normalization
    """
    cdef:
        int size, i
        bint check_mask, do_dark, do_flat, do_solidangle, do_absorption,
        bint is_valid, do_polarization, do_variance, do_dark_variance
        floating[:, ::1] result
        floating one_num, one_result, one_flat, one_den, one_var

    with gil:
        size = data.size
        do_dark = dark is not None
        do_flat = flat is not None
        do_solidangle = solidangle is not None
        do_absorption = absorption is not None
        do_polarization = polarization is not None
        check_mask = mask is not None
        do_variance = variance is not None
        do_dark_variance = dark_variance is not None
        result = numpy.zeros((size, 3), dtype=numpy.asarray(data).dtype)

    for i in prange(size, nogil=True, schedule="static"):
        one_num = data[i]
        one_den = normalization_factor
        if do_variance:
            one_var = variance[i]
        else:
            one_var = 0.0

        is_valid = not isnan(one_num)
        if is_valid and check_mask:
            is_valid = (mask[i] == 0)
        if is_valid and check_dummy:
            if delta_dummy == 0:
                is_valid = (one_num != dummy)
            else:
                is_valid = fabs(one_num - dummy) > delta_dummy

        if is_valid and do_flat:
            one_flat = flat[i]
            if delta_dummy == 0:
                is_valid = (one_flat != dummy)
            else:
                is_valid = fabs(one_flat - dummy) > delta_dummy

        if is_valid:
            # Do not use "/=" as they mean reduction for cython
            if do_dark:
                one_num = one_num - dark[i]
                if do_dark_variance:
                    one_var = one_var + dark_variance[i]
            if do_flat:
                one_den = one_den * flat[i]
            if do_polarization:
                one_den = one_den * polarization[i]
            if do_solidangle:
                one_den = one_den * solidangle[i]
            if do_absorption:
                one_den = one_den * absorption[i]
            if (isnan(one_num) or isnan(one_den) or isnan(one_var) or (one_den == 0)):
                one_num = 0.0
                one_var = 0.0
                one_den = 0.0
        else:
            one_num = 0.0
            one_var = 0.0
            one_den = 0.0

        result[i, 0] += one_num
        result[i, 1] += one_var
        result[i, 2] += one_den
    return result


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef floating[:, ::1]c4_preproc(floating[::1] data,
                                floating[::1] dark=None,
                                floating[::1] flat=None,
                                floating[::1] solidangle=None,
                                floating[::1] polarization=None,
                                floating[::1] absorption=None,
                                any_int_t[::1] mask=None,
                                floating dummy=0.0,
                                floating delta_dummy=0.0,
                                bint check_dummy=False,
                                floating normalization_factor=1.0,
                                floating[::1] variance=None,
                                floating[::1] dark_variance=None,
                                ) nogil:
    """Common preprocessing step for all routines: C-implementation
    with split_result to return (signal, variance, normalization, count)

    :param data: raw value, as a numpy array, 1D or 2D
    :param dark: array containing the value of the dark noise, to be subtracted
    :param flat: Array containing the flatfield image. It is also checked for dummies if relevant.
    :param solidangle: the value of the solid_angle. This processing may be performed during the rebinning instead. left for compatibility
    :param polarization: Correction for polarization of the incident beam
    :param absorption: Correction for absorption in the sensor volume
    :param mask: array non null  where data should be ignored
    :param dummy: value of invalid data
    :param delta_dummy: precision for invalid data
    :param normalization_factor: final value is divided by this, settles on the denominator
    :param variance: variance of the data
    :param dark_variance: variance of the dark
    NaN are always considered as invalid

    Empty pixels are 0.0 for both signal, variance, normalization and count
    """
    cdef:
        int size, i
        bint check_mask, do_dark, do_flat, do_solidangle, do_absorption,
        bint is_valid, do_polarization, do_variance, do_dark_variance
        floating[:, ::1] result
        floating one_num, one_result, one_flat, one_den, one_var, one_count

    with gil:
        size = data.size
        do_dark = dark is not None
        do_flat = flat is not None
        do_solidangle = solidangle is not None
        do_absorption = absorption is not None
        do_polarization = polarization is not None
        check_mask = mask is not None
        do_variance = variance is not None
        do_dark_variance = dark_variance is not None
        result = numpy.zeros((size, 4), dtype=numpy.asarray(data).dtype)

    for i in prange(size, nogil=True, schedule="static"):
        one_num = data[i]
        one_den = normalization_factor
        if do_variance:
            one_var = variance[i]
        else:
            one_var = 0.0

        is_valid = not isnan(one_num)
        if is_valid and check_mask:
            is_valid = (mask[i] == 0)
        if is_valid and check_dummy:
            if delta_dummy == 0:
                is_valid = (one_num != dummy)
            else:
                is_valid = fabs(one_num - dummy) > delta_dummy

        if is_valid and do_flat:
            one_flat = flat[i]
            if delta_dummy == 0:
                is_valid = (one_flat != dummy)
            else:
                is_valid = fabs(one_flat - dummy) > delta_dummy

        if is_valid:
            # Do not use "/=" as they mean reduction for cython
            if do_dark:
                one_num = one_num - dark[i]
                if do_dark_variance:
                    one_var = one_var + dark_variance[i]
            if do_flat:
                one_den = one_den * flat[i]
            if do_polarization:
                one_den = one_den * polarization[i]
            if do_solidangle:
                one_den = one_den * solidangle[i]
            if do_absorption:
                one_den = one_den * absorption[i]
            if (isnan(one_num) or isnan(one_den) or isnan(one_var) or (one_den == 0)):
                one_num = 0.0
                one_var = 0.0
                one_den = 0.0
                one_count = 0.0
            else:
                one_count = 1.0
        else:
            one_num = 0.0
            one_var = 0.0
            one_den = 0.0
            one_count = 0.0

        result[i, 0] += one_num
        result[i, 1] += one_var
        result[i, 2] += one_den
        result[i, 3] += one_count

    return result


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _preproc(floating[::1] raw,
             shape,
             bint check_dummy,
             floating dummy,
             floating delta_dummy,
             floating normalization_factor,
             floating[::1] dark=None,
             floating[::1] flat=None,
             floating[::1] solidangle=None,
             floating[::1] polarization=None,
             floating[::1] absorption=None,
             any_int_t[::1] mask=None,
             int split_result=0,
             floating[::1] variance=None,
             floating[::1] dark_variance=None,
             bint poissonian=False,
             ):
    """specialized preprocessing step for all corrections

    :param raw: raw value, as a numpy array, 1D with specialized dtype
    :param mask: array non null  where data should be ignored
    :param dummy: value of invalid data
    :param delta_dummy: precision for invalid data
    :param dark: array containing the value of the dark noise, to be subtracted
    :param flat: Array containing the flatfield image. It is also checked for dummies if relevant.
    :param solidangle: the value of the solid_angle. This processing may be performed during the rebinning instead. left for compatibility
    :param polarization: Correction for polarization of the incident beam
    :param absorption: Correction for absorption in the sensor volume
    :param normalization_factor: final value is divided by this
    :param empty: value to be given for empty bins
    :param variance: variance of the data
    :param dark_variance: variance of the dark
    :param poissonian: set to True to consider the variance is equal to raw signal

    All calculation are performed in the precision of raw dtype

    NaN are always considered as invalid
    """
    cdef:
        int size
        floating[::1] res1d
        floating[:, ::1] res2d

    # initialization of values:
    size = raw.size

    if split_result or (variance is not None) or poissonian:
        out_shape = list(shape)
        if split_result == 4:
            out_shape += [4]
            if poissonian:
                variance = raw
            res2d = c4_preproc(raw, dark, flat, solidangle, polarization, absorption,
                               mask, dummy, delta_dummy, check_dummy, normalization_factor,
                               variance, dark_variance)
        elif (variance is not None):
            out_shape += [3]
            res2d = c3_preproc(raw, dark, flat, solidangle, polarization, absorption,
                               mask, dummy, delta_dummy, check_dummy, normalization_factor,
                               variance, dark_variance)
        elif poissonian:
            out_shape += [3]
            res2d = cp_preproc(raw, dark, flat, solidangle, polarization, absorption,
                               mask, dummy, delta_dummy, check_dummy, normalization_factor)
        else:
            out_shape += [2]
            res2d = c2_preproc(raw, dark, flat, solidangle, polarization, absorption,
                               mask, dummy, delta_dummy, check_dummy, normalization_factor)
        res = numpy.asarray(res2d)
        res.shape = out_shape
    else:
        res1d = c1_preproc(raw, dark, flat, solidangle, polarization, absorption,
                           mask, dummy, delta_dummy, check_dummy, normalization_factor)
        res = numpy.asarray(res1d)
        res.shape = shape
    return res


def preproc(raw,
            dark=None,
            flat=None,
            solidangle=None,
            polarization=None,
            absorption=None,
            mask=None,
            dummy=None,
            delta_dummy=None,
            normalization_factor=None,
            empty=None,
            split_result=False,
            variance=None,
            dark_variance=None,
            bint poissonian=False,
            dtype=numpy.float32
            ):
    """Common preprocessing step for all

    :param raw: raw value, as a numpy array, 1D or 2D
    :param mask: array non null  where data should be ignored
    :param dummy: value of invalid data
    :param delta_dummy: precision for invalid data
    :param dark: array containing the value of the dark noise, to be subtracted
    :param flat: Array containing the flatfield image. It is also checked for dummies if relevant.
    :param solidangle: the value of the solid_angle. This processing may be performed during the rebinning instead. left for compatibility
    :param polarization: Correction for polarization of the incident beam
    :param absorption: Correction for absorption in the sensor volume
    :param normalization_factor: final value is divided by this
    :param empty: value to be given for empty bins
    :param variance: variance of the data
    :param dark_variance: variance of the dark
    :param poissonian: set to True to consider the variance is equal to raw signal
    :param dtype: type for working: float32 or float64

    All calculation are performed in the `dtype` precision

    NaN are always considered as invalid

    if neither empty nor dummy is provided, empty pixels are 0
    """
    cdef:
        bint check_dummy

    shape = raw.shape
    size = raw.size
    raw = numpy.ascontiguousarray(raw.ravel(), dtype=dtype)
    if dark is not None:
        assert dark.size == size, "Dark array size is correct"
        dark = numpy.ascontiguousarray(dark.ravel(), dtype=dtype)

    if flat is not None:
        assert flat.size == size, "Flat array size is correct"
        flat = numpy.ascontiguousarray(flat.ravel(), dtype=dtype)

    if polarization is not None:
        assert polarization.size == size, "Polarization array size is correct"
        polarization = numpy.ascontiguousarray(polarization.ravel(), dtype=dtype)

    if solidangle is not None:
        assert solidangle.size == size, "Solid angle array size is correct"
        solidangle = numpy.ascontiguousarray(solidangle.ravel(), dtype=dtype)

    if absorption is not None:
        assert absorption.size == size, "Absorption array size is correct"
        absorption = numpy.ascontiguousarray(absorption.ravel(), dtype=dtype)

    if variance is not None:
        assert variance.size == size, "Variance array size is correct"
        variance = numpy.ascontiguousarray(variance.ravel(), dtype=dtype)

    if dark_variance is not None:
        assert dark_variance.size == size, "Dark_variance array size is correct"
        dark_variance = numpy.ascontiguousarray(dark_variance.ravel(), dtype=dtype)

    if (dummy is None):
        check_dummy = False
        dummy = delta_dummy = dtype(empty or 0.0)

    else:
        check_dummy = True
        dummy = dtype(dummy)
        if (delta_dummy is None):
            delta_dummy = dtype(0.0)
        else:
            delta_dummy = dtype(delta_dummy)

    if normalization_factor is not None:
        normalization_factor = dtype(normalization_factor)
    else:
        normalization_factor = dtype(1.0)

    if (mask is None) or (mask is False):
        mask = None
    else:
        assert mask.size == size, "Mask array size is correct"
        mask = numpy.ascontiguousarray(mask.ravel(), dtype=numpy.int8)

    return _preproc(raw,
                    shape,
                    check_dummy,
                    dummy,
                    delta_dummy,
                    normalization_factor,
                    dark,
                    flat,
                    solidangle,
                    polarization,
                    absorption,
                    mask,
                    split_result,
                    variance,
                    dark_variance,
                    poissonian)
