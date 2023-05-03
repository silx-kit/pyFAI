#coding: utf-8
#cython: embedsignature=True, language_level=3, binding=True
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False,
## This is for developping:
##cython: profile=True, warn.undeclared=True, warn.unused=True, warn.unused_result=False, warn.unused_arg=True
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2018-2021 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Valentin Valls (Valentin.Valls@ESRF.eu)
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
__date__ = "09/03/2023"
__copyright__ = "2018-2021, ESRF"


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

include "sparse_builder.pxi"
include "regrid_common.pxi"


def feed_histogram(SparseBuilder builder not None,
                   pos,
                   weights,
                   int bins=100,
                   double empty=0.0,
                   double normalization_factor=1.0):
    """Missing docstring for feed_histogram
    Is this just a demo ?

    warning:
        * Unused argument 'empty'
        * Unused argument 'normalization_factor'

    """
    assert pos.size == weights.size
    assert bins > 1
    cdef:
        int  size = pos.size
        float[::1] cpos = numpy.ascontiguousarray(pos.ravel(), dtype=numpy.float32)
        float delta, min0, max0, maxin0
        float a = 0.0
        float fbin = 0.0
        int bin = 0, i

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
