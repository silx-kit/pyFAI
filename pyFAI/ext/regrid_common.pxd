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

# hack to avoid C compiler warnings about unused functions in the NumPy header files
# Sources: Cython test suite.

cdef extern from *:
    bint FALSE "0"
    void import_array()
    void import_umath()

if FALSE:
    import_array()
    import_umath()

from cython cimport floating
from libc.math cimport fabs, M_PI
cimport numpy as cnp
# How position are stored
ctypedef cnp.float64_t position_t

# How weights or data are stored 
ctypedef cnp.float32_t data_t

# how data are accumulated 
ctypedef cnp.float64_t acc_t

# type of the mask:
ctypedef cnp.int8_t mask_t

cdef:
    struct preproc_t:
        data_t signal
        data_t variance
        data_t norm
        
    float pi = <float> M_PI
    float piover2 = <float> (pi * 0.5)
    float onef = <float> 1.0
    float zerof = <float> 1.0
    #double EPS32 = (1.0 + numpy.finfo(numpy.float32).eps)
    double EPS32 = (1.0 + 1.0 / (1<<23))

ctypedef fused any_int_t:
    cnp.uint8_t
    cnp.uint16_t
    cnp.uint32_t
    cnp.uint64_t
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t
    
################################################################################
# Function prtotypes:
################################################################################
cdef floating  get_bin_number(floating x0, floating pos0_min, floating delta) nogil
cdef floating calc_upper_bound(floating maximum_value) nogil
 