#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/kif/pyFAI
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

__author__ = "Jerome Kieffer"
__date__ = "20140917"

import cython
import numpy
cimport numpy
import sys

from libc.math cimport floor,fabs
EPS32 = (1.0 + numpy.finfo(numpy.float32).eps)

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
    @param nthread: OpenMP is disabled. unused

    @return 2theta, I, weighted histogram, raw histogram
    """

    assert pos.size == weights.size
    assert  bins > 1
    cdef long  size = pos.size
    cdef double[:] cpos = numpy.ascontiguousarray(pos.ravel(),dtype=numpy.float64)
    cdef double[:] cdata = numpy.ascontiguousarray(weights.ravel(),dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim = 1] out_data = numpy.empty(bins, dtype="float64")
    cdef numpy.ndarray[numpy.float64_t, ndim = 1] out_count = numpy.empty(bins, dtype="float64")
    cdef numpy.ndarray[numpy.float64_t, ndim = 1] out_merge = numpy.empty(bins, dtype="float64")
    cdef numpy.ndarray[numpy.float64_t, ndim = 1] out_pos = numpy.empty(bins, dtype="float64")
    cdef double bin_edge_min, bin_edge_max
    if bin_range is not None:
        bin_edge_min = bin_range[0]
        bin_edge_max = bin_range[1] * EPS32
    else:
        bin_edge_min = pos.min()
        bin_edge_max = pos.max() * EPS32
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
    cdef long   i, idx, t

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

    if numpy.isnan(dbin) or numpy.isnan(inv_dbin2):
        dbin = 0.0
        inv_dbin2 = 0.0

    with nogil:
        for i in range(size):
            d = cdata[i]
            a = cpos[i]
            if (a < bin_edge_min) or (a > bin_edge_max):
                continue
            fbin = (a - bin_edge_min) * inv_bin_width
            ffbin = floor(fbin)
            bin = < long > ffbin
            dInt = 1.0
            if  bin > 0 :
                dtmp = ffbin - (fbin - dbin)
                if dtmp > 0:
                    dIntL = 0.5 * dtmp * dtmp * inv_dbin2
                    dInt = dInt - dIntL
                    out_count[bin - 1] += dIntL
                    out_data[bin - 1] += d * dIntL

            if bin < bins - 1 :
                dtmp = fbin + dbin - ffbin - 1
                if dtmp > 0 :
                    dIntR = 0.5 * dtmp * dtmp * inv_dbin2
                    dInt = dInt - dIntR
                    out_count[bin + 1] += dIntR
                    out_data[bin + 1] += d * dIntR
            out_count[bin] += dInt
            out_data[bin] += d * dInt

        for idx in range(bins):
            out_pos[idx] = bin_edge_min + (0.5 +< double > idx) * bin_width
            if out_count[idx] > epsilon:
                out_merge[idx] = out_data[idx]/out_count[idx] / tmp_count
            else:
                out_merge[idx] = dummy
    return  out_pos, out_merge, out_data, out_count


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
    cdef double[:] cpos0 = numpy.ascontiguousarray(pos0.ravel(),dtype=numpy.float64)
    cdef double[:] cpos1 = numpy.ascontiguousarray(pos1.ravel(),dtype=numpy.float64)
    cdef double[:] data = numpy.ascontiguousarray(weights.ravel(),dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim = 2] out_data = numpy.empty((bin0, bin1), dtype="float64")
    cdef numpy.ndarray[numpy.float64_t, ndim = 2] out_count = numpy.empty((bin0, bin1), dtype="float64")
    cdef numpy.ndarray[numpy.float64_t, ndim = 2] out_merge = numpy.empty((bin0, bin1), dtype="float64")
    cdef numpy.ndarray[numpy.float64_t, ndim = 1] edges0, edges1
    cdef double min0 = pos0.min()
    cdef double max0 = pos0.max() * EPS32
    cdef double min1 = pos1.min()
    cdef double max1 = pos1.max() * EPS32
    cdef double idp0 = (< double > bin0) / (max0 - min0)
    cdef double idp1 = (< double > bin1) / (max1 - min1)
    cdef double dbin0 = 0.5, dbin1 = 0.5
    cdef double fbin0, fbin1, p0, p1, d, rest, delta0l, delta0r, delta1l, delta1r, aera
    cdef double epsilon = 1e-10

    edges0 = numpy.linspace(min0+(0.5/idp0),max0-(0.5/idp0),bin0)
    edges1 = numpy.linspace(min1+(0.5/idp1),max1-(0.5/idp1),bin1)
    if nthread is not None:
        if isinstance(nthread, int) and (nthread > 0):
            omp_set_num_threads(< int > nthread)
    with nogil:
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
                        out_count[b0 - 1, b1 - 1] += area
                        out_data[b0 - 1, b1 - 1] += area * d

                        area = (-delta0l) * (1 + delta1l)
                        rest -= area
                        out_count[b0 - 1, b1 ] += area
                        out_data[b0 - 1, b1 ] += area * d

                        area = (1 + delta0l) * (-delta1l)
                        rest -= area
                        out_count[b0 , b1 - 1 ] += area
                        out_data[b0 , b1 - 1 ] += area * d

                    elif delta1r > 0 and b1 < bin1 - 1:
                        area = -delta0l * delta1r
                        rest -= area
                        out_count[b0 - 1, b1 + 1] += area
                        out_data[b0 - 1, b1 + 1] += area * d

                        area = (-delta0l) * (1 - delta1r)
                        rest -= area
                        out_count[b0 - 1, b1 ] += area
                        out_data[b0 - 1, b1 ] += area * d

                        area = (1 + delta0l) * (delta1r)
                        rest -= area
                        out_count[b0 , b1 + 1 ] += area
                        out_data[b0 , b1 + 1 ] += area * d
                elif delta0r > 0 and b0 < bin0 - 1:
                    if delta1l < 0 and b1 > 0:
                        area = -delta0r * delta1l
                        rest -= area
                        out_count[b0 + 1, b1 - 1] += area
                        out_data[b0 + 1, b1 - 1] += area * d

                        area = (delta0r) * (1 + delta1l)
                        rest -= area
                        out_count[b0 + 1, b1 ] += area
                        out_data[b0 + 1, b1 ] += area * d

                        area = (1 - delta0r) * (-delta1l)
                        rest -= area
                        out_count[b0 , b1 - 1 ] += area
                        out_data[b0 , b1 - 1 ] += area * d

                    elif delta1r > 0 and b1 < bin1 - 1:
                        area = delta0r * delta1r
                        rest -= area
                        out_count[b0 + 1, b1 + 1] += area
                        out_data[b0 + 1, b1 + 1] += area * d

                        area = (delta0r) * (1 - delta1r)
                        rest -= area
                        out_count[b0 + 1, b1 ] += area
                        out_data[b0 + 1, b1 ] += area * d

                        area = (1 - delta0r) * (delta1r)
                        rest -= area
                        out_count[b0 , b1 + 1 ] += area
                        out_data[b0 , b1 + 1 ] += area * d
            out_count[b0, b1] += rest
            out_data[b0, b1] += d * rest

        for i in prange(bin0):
            for j in range(bin1):
                if out_count[i, j] > epsilon:
                    out_merge[i, j] += out_data[i, j] / out_count[i, j]
                else:
                    out_merge[i, j] += dummy


    return out_merge, edges0, edges1, out_data, out_count
