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
