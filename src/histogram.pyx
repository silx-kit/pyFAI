#!/usr/bin/env python
# -*- coding: utf8 -*-
#
#    Project: Azimuthal integration 
#             https://forge.epn-campus.eu/projects/azimuthal
#
#    File: "$Id$"
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#



import cython
from cython.parallel cimport prange
cimport numpy
import numpy
cdef extern from "omp.h":
    ctypedef struct omp_lock_t:
        pass

    extern void omp_set_num_threads(int) nogil
    extern int omp_get_num_threads() nogil
    extern int omp_get_max_threads() nogil
    extern int omp_get_thread_num() nogil
    extern int omp_get_num_procs() nogil

    extern int omp_in_parallel() nogil
    extern void omp_init_lock(omp_lock_t *) nogil
    extern void omp_destroy_lock(omp_lock_t *) nogil
    extern void omp_set_lock(omp_lock_t *) nogil
    extern void omp_unset_lock(omp_lock_t *) nogil
    extern int omp_test_lock(omp_lock_t *) nogil


cdef extern from "stdlib.h":
    void free(void * ptr)nogil
    void * calloc(size_t nmemb, size_t size)nogil
    void * malloc(size_t size)nogil
cdef extern from "math.h":
    double floor(double)nogil
    double  fabs(double)nogil
    int     isnan(double)

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef numpy.int64_t DTYPE_int64_t
ctypedef numpy.float64_t DTYPE_float64_t

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def histogram(numpy.ndarray pos not None, \
              numpy.ndarray weights not None, \
              long bins=100,
              bin_range=None,
              pixelSize_in_Pos=None,
              nthread=None,
              double dummy=0.0):
    """
    Calculates histogram of pos weighted by weights
    
    @param pos: 2Theta array
    @param weights: array with intensities
    @param bins: number of output bins
    @param pixelSize_in_Pos: size of a pixels in 2theta
    @param nthread: maximum number of thread to use. By default: maximum available. 
        One can also limit this with OMP_NUM_THREADS environment variable
        
    @return 2theta, I, weighted histogram, raw histogram
    """

    assert pos.size == weights.size
    assert  bins > 1
    cdef long  size = pos.size
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] cpos = pos.astype("float64").ravel()
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] cdata = weights.astype("float64").ravel()
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] outData = numpy.zeros(bins, dtype="float64")
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] outCount = numpy.zeros(bins, dtype="float64")
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] outMerge = numpy.zeros(bins, dtype="float64")
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] outPos = numpy.zeros(bins, dtype="float64")
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] temp = numpy.zeros(size - 1, dtype="float64")
    cdef double bin_edge_min = pos.min()
    cdef double bin_edge_max = pos.max() * (1 + numpy.finfo(numpy.double).eps)
    if bin_range is not None:
        bin_edge_min = bin_range[0]
        bin_edge_max = bin_range[1] * (1 + numpy.finfo(numpy.double).eps)
    cdef double bin_width = (bin_edge_max - bin_edge_min) / (< double > (bins))
    cdef double inv_bin_width = (< double > (bins)) / (bin_edge_max - bin_edge_min)
    cdef double a = 0.0
    cdef double d = 0.0
    cdef double fbin = 0.0
    cdef double ffbin = 0.0
    cdef double dInt = 0.0
    cdef double dIntR = 0.0
    cdef double dIntL = 0.0
    cdef double dtmp = 0.0
    cdef double dbin, inv_dbin2 = 0.0
    cdef double tmp_count, tmp_data = 0.0
    cdef double epsilon = 1e-10

    cdef long   bin = 0
    cdef long   i, idx, t, dest = 0
    if nthread is not None:
        if isinstance(nthread, int) and (nthread > 0):
            omp_set_num_threads(< int > nthread)

    cdef double * bigCount = < double *> calloc(bins * omp_get_max_threads(), sizeof(double))
    cdef double * bigData = < double *> calloc(bins * omp_get_max_threads(), sizeof(double))
    if pixelSize_in_Pos is None:
        dbin = 0.5
        inv_dbin2 = 4.0
    elif isinstance(pixelSize_in_Pos, (int, float)):
        dbin = 0.5 * (< double > pixelSize_in_Pos) * inv_bin_width
        if dbin > 0.0:
            inv_dbin2 = 1 / dbin / dbin
        else:
            inv_dbin2 = 0.0
    elif isinstance(pixelSize_in_Pos, numpy.ndarray):
        pass #TODO

    if isnan(dbin) or isnan(inv_dbin2):
        dbin = 0.0
        inv_dbin2 = 0.0

    with nogil:
        for i in prange(size):
            d = cdata[i]
            a = cpos[i]
            if (a < bin_edge_min) or (a > bin_edge_max):
                continue
            fbin = (a - bin_edge_min) * inv_bin_width
            ffbin = floor(fbin)
            bin = < long > ffbin
            dest = omp_get_thread_num() * bins + bin
            dInt = 1.0
            if  bin > 0 :
                dtmp = ffbin - (fbin - dbin)
                if dtmp > 0:
                    dIntL = 0.5 * dtmp * dtmp * inv_dbin2
                    dInt = dInt - dIntL
                    bigCount[dest - 1] += dIntL
                    bigData[dest - 1] += d * dIntL

            if bin < bins - 1 :
                dtmp = fbin + dbin - ffbin - 1
                if dtmp > 0 :
                    dIntR = 0.5 * dtmp * dtmp * inv_dbin2
                    dInt = dInt - dIntR
                    bigCount[dest + 1] += dIntR
                    bigData[dest + 1] += d * dIntR
            bigCount[dest] += dInt
            bigData[dest] += d * dInt

        for idx in prange(bins):
            outPos[idx] = bin_edge_min + (0.5 +< double > idx) * bin_width
            tmp_count = 0.0
            tmp_data = 0.0
            for t in range(omp_get_max_threads()):
                dest = t * bins + idx
                tmp_count += bigCount[dest]
                tmp_data += bigData[dest]
            outCount[idx] += tmp_count
            outData[idx] += tmp_data
            if outCount[idx] > epsilon:
                outMerge[idx] += tmp_data / tmp_count
            else:
                outMerge[idx] += dummy

    free(bigCount)
    free(bigData)
    return  outPos, outMerge, outData, outCount


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def histogram2d(numpy.ndarray pos0 not None,
                numpy.ndarray pos1 not None,
                bins not None,
                numpy.ndarray weights not None,
                split=True,
                nthread=None,
                double dummy=0.0):
    """
    Calculate 2D histogram of pos0,pos1 weighted by weights

    @param pos0: 2Theta array
    @param pos1: Chi array
    @param weights: array with intensities
    @param bins: number of output bins int or 2-tuple of int
    @param nthread: maximum number of thread to use. By default: maximum available. 
    One can also limit this with OMP_NUM_THREADS environment variable

    
    @return  I, edges0, edges1, weighted histogram(2D), unweighted histogram (2D)
    """
    assert pos0.size == pos1.size
#    if weights is not No:
    assert pos0.size == weights.size
    cdef long  bin0, bin1, i, j, b0, b1
    cdef long  size = pos0.size
    try:
        bin0, bin1 = tuple(bins)
    except:
        bin0 = bin1 = < long > bins
    if bin0 <= 0:
        bin0 = 1
    if bin1 <= 0:
        bin1 = 1
    cdef int csplit = split
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] cpos0 = pos0.astype("float64").flatten()
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] cpos1 = pos1.astype("float64").flatten()
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] data = weights.astype("float64").flatten()
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 2] outData = numpy.zeros((bin0, bin1), dtype="float64")
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 2] outCount = numpy.zeros((bin0, bin1), dtype="float64")
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 2] outMerge = numpy.zeros((bin0, bin1), dtype="float64")
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] edges0 = numpy.zeros(bin0, dtype="float64")
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] edges1 = numpy.zeros(bin1, dtype="float64")
    cdef double min0 = pos0.min()
    cdef double max0 = pos0.max()
    cdef double min1 = pos1.min()
    cdef double max1 = pos1.max()
    cdef double idp0 = (< double > bin0) / (max0 - min0)
    cdef double idp1 = (< double > bin1) / (max1 - min1)
    cdef double dbin0 = 0.5, dbin1 = 0.5
    cdef double fbin0, fbin1, p0, p1, d, rest, delta0l, delta0r, delta1l, delta1r, aera
    if nthread is not None:
        if isinstance(nthread, int) and (nthread > 0):
            omp_set_num_threads(< int > nthread)
    with nogil:
        for i in prange(bin0):
            edges0[i] = min0 + (0.5 +< double > i) / idp0
        for i in prange(bin1):
            edges1[i] = min1 + (0.5 +< double > i) / idp1
        for i in range(size):
            p0 = cpos0[i]
            p1 = cpos1[i]
            d = data[i]
            fbin0 = (p0 - min0) * idp0
            fbin1 = (p1 - min1) * idp1
            b0 = < long > floor(fbin0)
            b1 = < long > floor(fbin1)
            if b0 == bin0:
                b0 = bin0 - 1
                fbin0 = (< double > bin0) - 0.5
            elif b0 == 0:
                fbin0 = 0.5
            if b1 == bin1:
                b1 = bin1 - 1
                fbin1 = (< double > bin1) - 0.5
            elif b1 == 0:
                fbin1 = 0.5

            delta0l = fbin0 -< double > b0 - dbin0
            delta0r = fbin0 -< double > b0 - 1 + dbin0
            delta1l = fbin1 -< double > b1 - dbin1
            delta1r = fbin1 -< double > b1 - 1 + dbin1
            rest = 1.0
            if csplit == 1:
                if delta0l < 0 and b0 > 0:
                    if delta1l < 0 and b1 > 0:
                        area = delta0l * delta1l
                        rest -= area
                        outCount[b0 - 1, b1 - 1] += area
                        outData[b0 - 1, b1 - 1] += area * d

                        area = (-delta0l) * (1 + delta1l)
                        rest -= area
                        outCount[b0 - 1, b1 ] += area
                        outData[b0 - 1, b1 ] += area * d

                        area = (1 + delta0l) * (-delta1l)
                        rest -= area
                        outCount[b0 , b1 - 1 ] += area
                        outData[b0 , b1 - 1 ] += area * d

                    elif delta1r > 0 and b1 < bin1 - 1:
                        area = -delta0l * delta1r
                        rest -= area
                        outCount[b0 - 1, b1 + 1] += area
                        outData[b0 - 1, b1 + 1] += area * d

                        area = (-delta0l) * (1 - delta1r)
                        rest -= area
                        outCount[b0 - 1, b1 ] += area
                        outData[b0 - 1, b1 ] += area * d

                        area = (1 + delta0l) * (delta1r)
                        rest -= area
                        outCount[b0 , b1 + 1 ] += area
                        outData[b0 , b1 + 1 ] += area * d
                elif delta0r > 0 and b0 < bin0 - 1:
                    if delta1l < 0 and b1 > 0:
                        area = -delta0r * delta1l
                        rest -= area
                        outCount[b0 + 1, b1 - 1] += area
                        outData[b0 + 1, b1 - 1] += area * d

                        area = (delta0r) * (1 + delta1l)
                        rest -= area
                        outCount[b0 + 1, b1 ] += area
                        outData[b0 + 1, b1 ] += area * d

                        area = (1 - delta0r) * (-delta1l)
                        rest -= area
                        outCount[b0 , b1 - 1 ] += area
                        outData[b0 , b1 - 1 ] += area * d

                    elif delta1r > 0 and b1 < bin1 - 1:
                        area = delta0r * delta1r
                        rest -= area
                        outCount[b0 + 1, b1 + 1] += area
                        outData[b0 + 1, b1 + 1] += area * d

                        area = (delta0r) * (1 - delta1r)
                        rest -= area
                        outCount[b0 + 1, b1 ] += area
                        outData[b0 + 1, b1 ] += area * d

                        area = (1 - delta0r) * (delta1r)
                        rest -= area
                        outCount[b0 , b1 + 1 ] += area
                        outData[b0 , b1 + 1 ] += area * d
            outCount[b0, b1] += rest
            outData[b0, b1] += d * rest

        for i in prange(bin0):
            for j in range(bin1):
                if outCount[i, j] > 1e-10:
                    outMerge[i, j] += outData[i, j] / outCount[i, j]
                else:
                    outMerge[i, j] += dummy


    return outMerge, edges0, edges1, outData, outCount
