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
__date__ = "21/06/2021"
__status__ = "stable"
__license__ = "MIT"


# Imports at the Python level 
import cython
import numpy
import sys

# Work around for issue similar to : https://github.com/pandas-dev/pandas/issues/16358

_numpy_1_12_py2_bug = ((sys.version_info.major == 2) and 
                       ([1, 12] >= [int(i) for i in numpy.version.version.split(".", 2)[:2]]))

# Imports at the C level
from .isnan cimport isnan
from cython cimport floating
from libc.math cimport fabs, M_PI, sqrt, log, NAN

from .shared_types cimport int8_t, uint8_t, int16_t, uint16_t, \
                           int32_t, uint32_t, int64_t, uint64_t,\
                           float32_t, float64_t

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

ctypedef fused any_int_t:
    uint8_t
    uint16_t
    uint32_t
    uint64_t
    int8_t
    int16_t
    int32_t
    int64_t


cdef:
    struct preproc_t:
        data_t signal
        data_t variance
        data_t norm
        data_t count
 
    float pi = <float> M_PI
    float piover2 = <float> (pi * 0.5)
    float onef = <float> 1.0
    float zerof = <float> 1.0
    double EPS32 = (1.0 + numpy.finfo(numpy.float32).eps)


from collections import namedtuple
from ..containers import Integrate1dtpl, Integrate2dtpl


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
cdef inline floating calc_upper_bound(floating maximum_value) nogil:
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
                                       floating dark_variance=0.0) nogil:
    """This is a Function in the C-space that performs the preprocessing
    for one data point 
    
    
    :param result: the container for the result, i.e. output which contains (signal, variance, normalisation, count)
    :param data and variance: the raw value and the associated variance
    :param dark and dark_variance: the dark-noise and the associated variance to be subtracted (signal) or added (variance)  
    :param flat, solidangle, polarization, absorption, normalization_factor: all normalization to be multiplied togeather
    :param dummy, delta_dummy, mask,check_dummy: controls the masking of the pixel 
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


@cython.boundscheck(False)
cdef inline void update_1d_accumulator(acc_t[:, ::1] out_data,
                                       int bin,
                                       preproc_t value,
                                       double weight=1.0) nogil:
    """Update a 1D array at given position with the proper values 
    
    :param out_data: output 1D+(,4) accumulator
    :param bin: in which bin assign this data
    :param value: 4-uplet with (signal, variance, nomalisation, count)
    :param weight: weight associated with this value 
    :return: Nothing
    """
    out_data[bin, 0] += value.signal * weight
    out_data[bin, 1] += value.variance * weight * weight  # Important for variance propagation
    out_data[bin, 2] += value.norm * weight
    out_data[bin, 3] += value.count * weight


@cython.boundscheck(False)
cdef inline void update_2d_accumulator(acc_t[:, :, ::1] out_data,
                                       int bin0,
                                       int bin1,
                                       preproc_t value,
                                       double weight=1.0) nogil:
    """Update a 2D array at given position with the proper values 
    
    :param out_data: 2D+1 accumulator
    :param bin0, bin1: where to assign data 
    :return: Nothing
    """
    out_data[bin0, bin1, 0] += value.signal * weight
    out_data[bin0, bin1, 1] += value.variance * weight * weight  # Important for variance propagation
    out_data[bin0, bin1, 2] += value.norm * weight
    out_data[bin0, bin1, 3] += value.count * weight
