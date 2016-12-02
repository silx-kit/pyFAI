# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
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


__author__ = "Jerome Kieffer"
__license__ = "MIT"
__date__ = "01/12/2016"
__copyright__ = "2011-2015, ESRF"
__contact__ = "jerome.kieffer@esrf.fr"

import cython
import numpy
cimport numpy as cnp
from cython cimport floating
from cython.parallel import prange
from libc.math cimport fabs


IF UNAME_SYSNAME == "Windows":
    cdef extern from "numpy/npy_math.h" nogil:
        long double NAN "NPY_NAN"
        bint isnan "npy_isnan"(long double)
ELSE:
    from libc.math cimport isnan


ctypedef fused any_int_t:
    cnp.uint8_t
    cnp.uint16_t
    cnp.uint32_t
    cnp.uint64_t
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t
    
    
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef floating[::1]c_preproc(floating[::1] data,
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
        bint check_mask, do_dark, do_flat, do_solidangle, do_absorption, do_polarization 
        bint is_valid
        floating[::1] result   
        floating one_value, one_result, one_flat

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
        one_value = data[i]
        
        is_valid = not isnan(one_value)
        if is_valid and check_mask:
            is_valid = (mask[i] == 0)
        if is_valid and check_dummy:
            if delta_dummy == 0:
                is_valid = (one_value != dummy)
            else:
                is_valid = fabs(one_value - dummy) > delta_dummy
                
        if is_valid and do_flat:
            one_flat = flat[i]
            if delta_dummy == 0:
                is_valid = (one_flat != dummy)
            else:
                is_valid = fabs(one_flat - dummy) > delta_dummy
    
        if is_valid:
            # Do not use "/=" as they mean reduction for cython
            if do_dark:
                one_value = one_value - dark[i]
            if do_flat:
                one_value = one_value / flat[i]
            if do_polarization:
                one_value = one_value / polarization[i]
            if do_solidangle:
                one_value = one_value / solidangle[i]
            if do_absorption:
                one_value = one_value / absorption[i]
            one_result = one_value / normalization_factor
        else:
            one_result = dummy

        result[i] += one_result
    return result


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
def preproc(data,
            dark=None,
            flat=None,
            solidangle=None,
            polarization=None,
            absorption=None,
            mask=None,
            dummy=None,
            delta_dummy=None,
            float normalization_factor=1.0,
            empty=None
            ):
    """Common preprocessing step for all 
    
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
    
    All calculation are performed in single precision floating point.
    
    NaN are always considered as invalid
    
    if neither empty nor dummy is provided, empty pixels are 0  
    """
    cdef:
        int size
        bint check_dummy 
        cnp.int8_t[::1] cmask
        float[::1] cdata, cdark, cflat, csolidangle, cpolarization, cabsorpt, result   
        float cdummy, ddummy
    
    # initialization of values:
    size = data.size
    cdata = numpy.ascontiguousarray(data.ravel(), dtype=numpy.float32)
    
    if (mask is None) or (mask is False):
        cmask = None
    else:
        assert mask.size == size, "Mask array size is correct"
        cmask = numpy.ascontiguousarray(mask.ravel(), dtype=numpy.int8)

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
        assert dark.size == size, "Dark array size is correct"
        cdark = numpy.ascontiguousarray(dark.ravel(), dtype=numpy.float32)
    else:
        cdark = None

    if flat is not None:
        assert flat.size == size, "Flat array size is correct"
        cflat = numpy.ascontiguousarray(flat.ravel(), dtype=numpy.float32)
    else:
        cflat = None
        
    if polarization is not None:
        assert polarization.size == size, "Polarization array size is correct"
        cpolarization = numpy.ascontiguousarray(polarization.ravel(), dtype=numpy.float32)
    else:
        cpolarization = None

    if solidangle is not None:
        assert solidangle.size == size, "Solid angle array size is correct"
        csolidangle = numpy.ascontiguousarray(solidangle.ravel(), dtype=numpy.float32)
    else:
        csolidangle = None

    if absorption is not None:
        assert absorption.size == size, "Absorption array size is correct"
        cabsorpt = numpy.ascontiguousarray(absorption.ravel(), dtype=numpy.float32)
    else:
        cabsorpt = None

    result = c_preproc(cdata, cdark, cflat, csolidangle, cpolarization, cabsorpt,
                       cmask, cdummy, ddummy, check_dummy, normalization_factor)

    return numpy.asarray(result).reshape(data.shape)
