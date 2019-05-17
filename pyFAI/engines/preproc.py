#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2017-2018 European Synchrotron Radiation Facility, Grenoble, France
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


"""Module providing common pixel-wise pre-processing of data.
"""

from __future__ import absolute_import, print_function, with_statement

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "17/05/2019"
__status__ = "development"

import warnings
import numpy


def preproc(raw,
            dark=None,
            flat=None,
            solidangle=None,
            polarization=None,
            absorption=None,
            mask=None,
            dummy=None,
            delta_dummy=None,
            normalization_factor=1.0,
            empty=None,
            split_result=False,
            variance=None,
            dark_variance=None,
            poissonian=False,
            dtype=numpy.float32
            ):
    """Common preprocessing step for all integration engines

    :param data: raw value, as a numpy array, 1D or 2D
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
    :param split_result: set to true to separate signal from normalization and
            return an array of float2, float3 (with variance) ot float4 (including counts)
    :param variance: provide an estimation of the variance, enforce
            split_result=True and return an float3 array with variance in second position.
    :param dark_variance: provide an estimation of the variance of the dark_current,
            enforce split_result=True and return an float3 array with variance in second position.
    :param poissonian: set to "True" for assuming the detector is poissonian and variance = raw + dark
    :param dtype: dtype for all processing

    All calculation are performed in single precision floating point (32 bits).

    NaN are always considered as invalid values

    if neither empty nor dummy is provided, empty pixels are 0.
    Empty pixels are always zero in "split_result" mode.

    When set to False, i.e the default, the pixel-wise operation is:
    I = (raw - dark)/(flat \* solidangle \* polarization \* absorption)
    Invalid pixels are set to the dummy or empty value.

    When split_ressult is set to True, each result is a float2
    or a float3 (with an additional value for the variance) as such:
    I = [(raw - dark), (variance), (flat \* solidangle \* polarization \* absorption)]
    Empty pixels will have all their 2 or 3 values to 0 (and not to dummy or empty value)

    If poissonian is set to True, the variance is evaluated as (raw + dark).
    """
    if isinstance(dtype, str):
        dtype = numpy.dtype(dtype).type
    shape = raw.shape
    out_shape = list(shape)
    if split_result or (variance is not None) or poissonian:
        if split_result == 4:
            out_shape += [4]
        elif (variance is not None) or poissonian:
            out_shape += [3]
        else:
            out_shape += [2]
        split_result = True
    size = raw.size
    if (mask is None) or (mask is False):
        mask = numpy.zeros(size, dtype=bool)
    else:
        assert mask.size == size, "Mask array size is correct"
        mask = numpy.ascontiguousarray(mask.ravel(), dtype=bool)

    if (dummy is not None) and (delta_dummy is not None):
        check_dummy = True
        cdummy = dtype(dummy)
        ddummy = dtype(delta_dummy)
    elif (dummy is not None):
        check_dummy = True
        cdummy = dtype(dummy)
        ddummy = 0.0
    else:
        check_dummy = False
        cdummy = dtype(empty or 0.0)
        ddummy = 0.0

    signal = numpy.ascontiguousarray(raw.ravel(), dtype=dtype)
    normalization = numpy.zeros_like(signal) + normalization_factor
    if variance is not None:
        variance = numpy.ascontiguousarray(variance.ravel(), dtype=dtype)
    elif poissonian:
        variance = signal.copy()

    # runtime warning here
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        if check_dummy:
            # runtime warning here
            if ddummy == 0:
                mask |= (signal == cdummy)
            else:
                mask |= (abs(signal - cdummy) <= ddummy)

        if dark is not None:
            assert dark.size == size, "Dark array size is correct"
            dark = numpy.ascontiguousarray(dark.ravel(), dtype=dtype)
            if check_dummy:
                # runtime warning here
                if ddummy == 0:
                    mask |= (dark == cdummy)
                else:
                    mask |= abs(dark - cdummy) < ddummy
            signal -= dark
            if poissonian:
                variance += dark
            elif dark_variance is not None:
                variance += dark_variance

        if flat is not None:
            assert flat.size == size, "Flat array size is correct"
            flat = numpy.ascontiguousarray(flat.ravel(), dtype=dtype)
            if check_dummy:
                # runtime warning here
                if ddummy == 0:
                    mask |= (flat == cdummy)
                else:
                    mask |= abs(flat - cdummy) <= ddummy
            normalization *= flat

        if polarization is not None:
            assert polarization.size == size, "Polarization array size is correct"
            normalization *= numpy.ascontiguousarray(polarization.ravel(), dtype=dtype)

        if solidangle is not None:
            assert solidangle.size == size, "Solid angle array size is correct"
            normalization *= numpy.ascontiguousarray(solidangle.ravel(), dtype=dtype)

        if absorption is not None:
            assert absorption.size == size, "Absorption array size is correct"
            normalization *= numpy.ascontiguousarray(absorption.ravel(), dtype=dtype)

        mask |= numpy.logical_not(numpy.isfinite(signal))
        mask |= numpy.logical_not(numpy.isfinite(normalization))
        mask |= (normalization == 0)
        if variance is not None:
            mask |= numpy.logical_not(numpy.isfinite(variance))

        if split_result:
            result = numpy.zeros(out_shape, dtype=dtype)
            signal[mask] = 0.0
            normalization[mask] = 0.0
            result[..., 0] = signal.reshape(shape)
            if out_shape[-1] == 4:
                if variance is not None:
                    variance[mask] = 0.0
                    result[..., 1] = variance.reshape(shape)
                result[..., 2] = normalization.reshape(shape)
                result[..., 3] = 1.0 - mask.reshape(shape)
            elif variance is None:
                result[:, :, 1] = normalization.reshape(shape)
            else:
                variance[mask] = 0.0
                result[..., 1] = variance.reshape(shape)
                result[..., 2] = normalization.reshape(shape)
        else:
            result = signal / normalization
            result[mask] = cdummy
            result.shape = shape
    return result
