# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2018 European Synchrotron Radiation Facility, Grenoble, France
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

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "09/11/2018"
__status__ = "stable"
__license__ = "MIT"


include "numpy_common.pxi"

# Imports at the Python level 
import cython
import numpy

# Imports at the C level
from isnan cimport isnan
from cython cimport floating
from libc.math cimport fabs, M_PI
cimport numpy as cnp

# How position are stored
ctypedef cnp.float64_t position_t
position_d = numpy.float64

# How weights or data are stored 
ctypedef cnp.float32_t data_t
data_d = numpy.float32

# how data are accumulated 
ctypedef cnp.float64_t acc_t
acc_d = numpy.float64

# type of the mask:
ctypedef cnp.int8_t mask_t
mask_d = numpy.int8

# Type used for propagating variance
prop_d = numpy.dtype([('signal', acc_d),
                      ('variance', acc_d),
                      ('norm', acc_d),
                      ('count', acc_d)])

ctypedef fused any_int_t:
    cnp.uint8_t
    cnp.uint16_t
    cnp.uint32_t
    cnp.uint64_t
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t


cdef:
    struct preproc_t:
        data_t signal
        data_t variance
        data_t norm
 
    float pi = <float> M_PI
    float piover2 = <float> (pi * 0.5)
    float onef = <float> 1.0
    float zerof = <float> 1.0
    double EPS32 = (1.0 + numpy.finfo(numpy.float32).eps)

from collections import namedtuple

Integrate1dResult = namedtuple("Integrate1dResult", ["bins", "signal", "propagated"])
Integrate2dResult = namedtuple("Integrate2dResult", ["signal", "bins0", "bins1", "propagated"])
Integrate1dWithErrorResult = namedtuple("Integrate1dWithErrorResult", ["bins", "signal", "error", "propagated"])
Integrate2dWithErrorResult = namedtuple("Integrate2dWithErrorResult", ["signal", "error", "bins0", "bins1", "propagated"])


@cython.cdivision(True)
cdef floating  get_bin_number(floating x0, floating pos0_min, floating delta) nogil:
    """
    calculate the bin number for any point (as floating)

    :param x0: current position
    :param pos0_min: position minimum
    :param delta: bin width
    :return: bin number as floating point.
    """
    return (x0 - pos0_min) / delta


@cython.cdivision(True)
cdef floating calc_upper_bound(floating maximum_value) nogil:
    """Calculate the upper_bound for an histogram, 
    given the maximum value of all the data.
    
    :param maximum_value: maximum value over all elements
    :return: the smallest 32 bit float greater than the maximum
    """
    if maximum_value > 0:
        return maximum_value * EPS32
    else:
        return maximum_value / EPS32


cdef inline preproc_t preproc_value(floating data,
                                    floating variance=0.0,
                                    floating dark=0.0,
                                    floating flat=1.0,
                                    floating solidangle=1.0,
                                    floating polarization=1.0,
                                    floating absorption=1.0,
                                    any_int_t mask=0,
                                    floating dummy=0.0,
                                    floating delta_dummy=0.0,
                                    bint check_dummy=False,
                                    floating normalization_factor=1.0,
                                    floating dark_variance=0.0) nogil:
    """This is a Function in the C-space that performs the preprocessing
    for one data point 
    
    
    :param: data,
    :return: preproc_t which contains (signal, variance, normalisation)

    where:
    * signal = data-dark
    * variance = variance + dark_variance 
    * norm = prod(all normalisation)
    
    unless data are invalid (mask, nan, ...) , in this 


    """
    cdef:
        floating signal, norm
        preproc_t result
        bint is_valid
    signal = data

    is_valid = (not isnan(signal)) and (mask == 0) 
    if is_valid and check_dummy:
        if delta_dummy == 0.0:
            is_valid = (signal != dummy)
        else:
            is_valid = fabs(signal - dummy) > delta_dummy

    if is_valid:
        if delta_dummy == 0.0:
            is_valid = (flat != dummy)
        else:
            is_valid = fabs(flat - dummy) > delta_dummy

    if is_valid:
        # Do not use "/=" as they mean reduction for cython
        if dark:
            signal = signal - dark
            if dark_variance:
                variance = variance + dark_variance
        norm = normalization_factor * flat * polarization * solidangle * absorption
        if (isnan(signal) or isnan(norm) or isnan(variance) or (norm == 0)):
            signal = 0.0
            variance = 0.0
            norm = 0.0
    else:
        signal = 0.0
        variance = 0.0
        norm = 0.0
    result.signal = signal
    result.variance = variance
    result.norm = norm
    return result
