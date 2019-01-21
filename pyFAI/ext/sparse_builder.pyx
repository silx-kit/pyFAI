#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2018 European Synchrotron Radiation Facility, Grenoble, France
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

__author__ = "Valentin Valls"
__license__ = "MIT"
__date__ = "21/01/2019"
__copyright__ = "2018, ESRF"

import numpy
cimport numpy as cnumpy

from libcpp.vector cimport vector
from libcpp.list cimport list as clist
from libcpp cimport bool
from libc.math cimport fabs
cimport libc.stdlib
cimport libc.string

from cython.parallel import prange
from cython.operator cimport dereference
from cython.operator cimport preincrement
cimport cython
from cython cimport floating

include "sparse_builder.pxi"

cdef double EPS32 = (1.0 + numpy.finfo(numpy.float32).eps)


@cython.cdivision(True)
cdef inline floating calc_upper_bound(floating maximum_value) nogil:
    if maximum_value > 0:
        return maximum_value * EPS32
    else:
        return maximum_value / EPS32


@cython.cdivision(True)
cdef inline floating  get_bin_number(floating x0, floating pos0_min, floating delta) nogil:
    return (x0 - pos0_min) / delta


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def feed_histogram(SparseBuilder builder not None,
                   cnumpy.ndarray pos not None,
                   cnumpy.ndarray weights not None,
                   int bins=100,
                   double empty=0.0,
                   double normalization_factor=1.0):
    assert pos.size == weights.size
    assert bins > 1
    cdef:
        int  size = pos.size
        cnumpy.float32_t[::1] cpos = numpy.ascontiguousarray(pos.ravel(), dtype=numpy.float32)
        cnumpy.float32_t delta, min0, max0, maxin0
        cnumpy.float32_t a = 0.0
        cnumpy.float32_t d = 0.0
        cnumpy.float32_t fbin = 0.0
        cnumpy.float32_t tmp_count, tmp_data = 0.0
        cnumpy.float32_t epsilon = 1e-10
        int bin = 0, i, idx

    min0 = pos.min()
    maxin0 = pos.max()
    max0 = calc_upper_bound(maxin0)
    delta = (max0 - min0) / float(bins)

    with nogil:
        for i in range(size):
            a = cpos[i]
            fbin = get_bin_number(a, min0, delta)
            bin = < int > fbin
            builder.cinsert(bin, i, 1.0)
