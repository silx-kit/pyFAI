# -*- coding: utf-8 -*-
#cython: embedsignature=True, language_level=3, binding=True
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False,
################################################################################
# #This is for developping
# #cython: profile=True, warn.undeclared=True, warn.unused=True, warn.unused_result=False, warn.unused_arg=True
################################################################################
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2011-2022 European Synchrotron Radiation Facility, Grenoble, France
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
__date__ = "26/04/2024"
__copyright__ = "2011-2022, ESRF"
__contact__ = "jerome.kieffer@esrf.fr"

include "regrid_common.pxi"

import cython
from ..containers import ErrorModel
from libc.math cimport fabs, isfinite


cdef floating[:, ::1] c1_preproc(floating[::1] data,
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
                              bint apply_normalization=True,
                              floating[:, ::1] result = None
                              ) noexcept with gil:
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
    :param apply_normalization: correct (directly) the raw signal & variance with normalization, WIP
    :param result: output array

    NaN are always considered as invalid

    if neither empty nor dummy is provided, empty pixels are 0
    """
    cdef:
        int size, i
        bint check_mask, do_dark, do_flat, do_solidangle, do_absorption,
        bint do_polarization, is_valid
        floating one_num, one_den, one_flat

    size = data.shape[0]
    do_dark = dark is not None
    do_flat = flat is not None
    do_solidangle = solidangle is not None
    do_absorption = absorption is not None
    do_polarization = polarization is not None
    check_mask = mask is not None
    if result is None:
        result = numpy.empty((size, 1), dtype=data.base.dtype)
    else:
        assert result.shape[0] == size, "size matches"
        assert result.shape[1] == 1, "nb component = 1"
    with nogil:
        for i in range(size):
            one_num = data[i]
            one_den = normalization_factor
            is_valid = isfinite(one_num)
            is_valid = (mask[i] == 0) if check_mask and is_valid else is_valid
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
                    one_num -= dark[i]
                if do_flat:
                    one_den *= one_flat
                if do_polarization:
                    one_den *= polarization[i]
                if do_solidangle:
                    one_den *= solidangle[i]
                if do_absorption:
                    one_den *= absorption[i]
                if (isfinite(one_num) and isfinite(one_den) and (one_den != 0)):
                    result[i, 0] = one_num / one_den
                else:
                    result[i, 0] = dummy
            else:
                result[i, 0] = dummy
    return result


cdef floating[:, ::1] c2_preproc(floating[::1] data,
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
                                 bint apply_normalization=False,
                                 floating[:, ::1] result=None
                                 ) noexcept with gil:
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
    :param normalization_factor: final value is divided by this value
    :param apply_normalization: correct (directly) the raw signal & variance with normalization, WIP
    :param out: output array pre-allocated

    NaN are always considered as invalid

    Empty pixels are 0 both num and den
    """
    cdef:
        int size, i
        bint check_mask, do_dark, do_flat, do_solidangle, do_absorption, do_polarization
        bint is_valid
        floating one_num, one_flat, one_den

    size = data.shape[0]
    do_dark = dark is not None
    do_flat = flat is not None
    do_solidangle = solidangle is not None
    do_absorption = absorption is not None
    do_polarization = polarization is not None
    check_mask = mask is not None
    if result is None:
        result = numpy.empty((size, 2), dtype=data.base.dtype)
    else:
        assert result.shape[0] == size, "size matches"
        assert result.shape[1] == 2, "nb component = 2"

    with nogil:
        for i in range(size):
            one_num = data[i]
            one_den = normalization_factor
            is_valid = isfinite(one_num)
            is_valid = (mask[i] == 0) if is_valid and check_mask else is_valid
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
                    one_num -= dark[i]
                if do_flat:
                    one_den *= flat[i]
                if do_polarization:
                    one_den *= polarization[i]
                if do_solidangle:
                    one_den *= solidangle[i]
                if do_absorption:
                    one_den *= absorption[i]
                if not (isfinite(one_num) and isfinite(one_den) and (one_den != 0)):
                    one_num = 0.0
                    one_den = 0.0
                elif apply_normalization:
                    one_num /= one_den
                    one_den = 1.0

            else:
                one_num = 0.0
                one_den = 0.0

            result[i, 0] = one_num
            result[i, 1] = one_den
    return result


cdef floating[:, ::1] c3_preproc(floating[::1] data,
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
                                 bint poissonian=False,
                                 bint apply_normalization=False,
                                 floating[:, ::1] result=None,
                                 ) noexcept with gil:
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
    :param poissonian: if True, the variance is the signal (minimum 1)
    :param apply_normalization: correct (directly) the raw signal & variance with normalization, WIP
    :param result: output array pre-allocated
    NaN are always considered as invalid

    Empty pixels are 0.0 for both signal, variance and normalization
    """
    cdef:
        int size, i
        bint check_mask, do_dark, do_flat, do_solidangle, do_absorption,
        bint is_valid, do_polarization, do_variance, do_dark_variance
        floating one_num, one_flat, one_den, one_var

    size = data.shape[0]
    do_dark = dark is not None
    do_flat = flat is not None
    do_solidangle = solidangle is not None
    do_absorption = absorption is not None
    do_polarization = polarization is not None
    check_mask = mask is not None
    do_variance = variance is not None
    do_dark_variance = dark_variance is not None
    if result is None:
        result = numpy.empty((size, 3), dtype=data.base.dtype)
    else:
        assert result.shape[0] == size, "size matches"
        assert result.shape[1] == 3, "nb component = 3"


    with nogil:
        for i in range(size):
            one_num = data[i]
            one_den = normalization_factor
            if poissonian:
                one_var = max(one_num, 1.0)
            elif do_variance:
                one_var = variance[i]
            else:
                one_var = 0.0

            is_valid = isfinite(one_num)
            is_valid = (mask[i] == 0) if is_valid and check_mask else is_valid

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
                    one_num -= dark[i]
                    if do_dark_variance:
                        one_var += dark_variance[i]
                if do_flat:
                    one_den *= flat[i]
                if do_polarization:
                    one_den *= polarization[i]
                if do_solidangle:
                    one_den *= solidangle[i]
                if do_absorption:
                    one_den *= absorption[i]
                if not (isfinite(one_num) and isfinite(one_den) and isfinite(one_var) and (one_den != 0)):
                    one_num = 0.0
                    one_var = 0.0
                    one_den = 0.0
                elif apply_normalization:
                    one_num /= one_den
                    one_var /= one_den * one_den
                    one_den = 1.0
            else:
                one_num = 0.0
                one_var = 0.0
                one_den = 0.0

            result[i, 0] = one_num
            result[i, 1] = one_var
            result[i, 2] = one_den
    return result


cdef floating[:, ::1] c4_preproc(floating[::1] data,
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
                                 bint poissonian=False,
                                 bint apply_normalization=False,
                                 floating[:, ::1] result=None,
                                 ) noexcept with gil:
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
    :param poissonian: if True, the variance is the signal (minimum 1)
    :param apply_normalization: correct (directly) the raw signal & variance with normalization, WIP
    :param result: pre-allocated array
    NaN are always considered as invalid

    Empty pixels are 0.0 for both signal, variance, normalization and count
    """
    cdef:
        int size, i
        bint check_mask, do_dark, do_flat, do_solidangle, do_absorption,
        bint is_valid, do_polarization, do_variance, do_dark_variance
        floating one_num, one_flat, one_den, one_var, one_count

    size = data.shape[0]
    do_dark = dark is not None
    do_flat = flat is not None
    do_solidangle = solidangle is not None
    do_absorption = absorption is not None
    do_polarization = polarization is not None
    check_mask = mask is not None
    do_variance = variance is not None
    do_dark_variance = dark_variance is not None

    with nogil:
        for i in range(size):
            one_num = data[i]
            one_den = normalization_factor
            if poissonian:
                one_var = max(one_num, 1.0)
            elif do_variance:
                one_var = variance[i]
            else:
                one_var = 0.0

            is_valid = isfinite(one_num)
            is_valid = (mask[i] == 0) if is_valid and check_mask else is_valid
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
                if do_dark:
                    one_num -= dark[i]
                    if do_dark_variance:
                        one_var += dark_variance[i]
                if do_flat:
                    one_den *= flat[i]
                if do_polarization:
                    one_den *= polarization[i]
                if do_solidangle:
                    one_den *= solidangle[i]
                if do_absorption:
                    one_den *= absorption[i]
                if not (isfinite(one_num) and isfinite(one_den) and isfinite(one_var) and (one_den != 0)):
                    one_num = 0.0
                    one_var = 0.0
                    one_den = 0.0
                    one_count = 0.0
                else:
                    one_count = 1.0
                    if apply_normalization:
                        one_num /= one_den
                        one_var /= one_den * one_den
                        one_den = 1.0
            else:
                one_num = 0.0
                one_var = 0.0
                one_den = 0.0
                one_count = 0.0

            result[i, 0] = one_num
            result[i, 1] = one_var
            result[i, 2] = one_den
            result[i, 3] = one_count

    return result

def _preproc(floating[::1] raw,
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
             floating[::1] variance=None,
             floating[::1] dark_variance=None,
             bint poissonian=False,
             bint apply_normalization=False,
             floating[:, ::1] result=None
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
    :param poissonian: set to True to consider the variance is equal to raw signal (minimum 1)
    :param apply_normalization: correct (directly) the raw signal & variance with normalization, WIP
    :param result: pre-allocated array

    All calculation are performed in the precision of raw dtype

    NaN are always considered as invalid
    """
    cdef int split = result.shape[1]
    if split == 4:
        result = c4_preproc(raw, dark, flat, solidangle, polarization, absorption,
                   mask, dummy, delta_dummy, check_dummy, normalization_factor,
                   variance, dark_variance, poissonian, apply_normalization, result)
    elif split == 3:
        result = c3_preproc(raw, dark, flat, solidangle, polarization, absorption,
                   mask, dummy, delta_dummy, check_dummy, normalization_factor,
                   variance, dark_variance, poissonian, apply_normalization, result)
    elif split == 2:
        result = c2_preproc(raw, dark, flat, solidangle, polarization, absorption,
                   mask, dummy, delta_dummy, check_dummy, normalization_factor,
                   apply_normalization, result)
    elif split == 1:
        result = c1_preproc(raw, dark, flat, solidangle, polarization, absorption,
                   mask, dummy, delta_dummy, check_dummy, normalization_factor,
                   apply_normalization, result)
    return result


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
            error_model=ErrorModel.NO,
            bint apply_normalization=False,
            dtype=numpy.float32,
            out=None
            ):
    """Common preprocessing step for all integrators

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
    :param error_model: set to "poisson" to consider the variance is equal to raw signal (minimum 1)
    :param apply_normalization: correct (directly) the raw signal & variance with normalization, WIP
    :param dtype: type for working: float32 or float64
    :param out: output buffer to save a malloc

    All calculation are performed in the `dtype` precision

    NaN are always considered as invalid

    if neither empty nor dummy is provided, empty pixels are 0
    """
    cdef:
        bint check_dummy, poissonian=error_model.poissonian
        tuple shape
        int size, ndim
        str key

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

    if split_result or (variance is not None) or poissonian:
        if split_result == 4:
            ndim = 4
        elif  (variance is not None) or poissonian:
            ndim = 3
        else:
            ndim = 2
    else:
        ndim = 1

    if out is None:
        result = numpy.empty((size, ndim), dtype=dtype)
    else:
        # assert out.dtype == dtype, "output dtype matches"
        if out.shape[0] == size and out.shape[1]:
            result = out
        else:
            result = out.reshape((size, ndim))

    key = ("float" if numpy.dtype(dtype).itemsize==4 else "double")+"|int8_t"
    cpreproc =_preproc.__signatures__[key]
    cpreproc(raw,
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
             variance,
             dark_variance,
             poissonian,
             apply_normalization,
             result)

    if ndim == 1:
        return numpy.asarray(result).reshape(shape)
    else:
        return numpy.asarray(result).reshape(shape+(ndim,))
