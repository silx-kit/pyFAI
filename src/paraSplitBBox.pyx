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
cimport numpy
import numpy

from cython.parallel import prange

cdef extern from "math.h":
    float floor(float)nogil
    float fabs(float)nogil

@cython.cdivision(True)
cdef float  getBinNr(float x0, float pos0_min, float delta) nogil:
    """
    calculate the bin number for any point
    param x0: current position
    param pos0_min: position minimum
    param delta: bin width
    """
    return (x0 - pos0_min) / delta


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def histoBBox1d(numpy.ndarray weights not None,
                numpy.ndarray pos0 not None,
                numpy.ndarray delta_pos0 not None,
                pos1=None,
                delta_pos1=None,
                long bins=100,
                pos0Range=None,
                pos1Range=None,
                dummy=None,
                delta_dummy=None,
                mask=None,
                dark=None,
                flat=None
              ):
    """
    Calculates histogram of pos0 (tth) weighted by weights

    Splitting is done on the pixel's bounding box like fit2D

    @param weights: array with intensities
    @param pos0: 1D array with pos0: tth or q_vect
    @param delta_pos0: 1D array with delta pos0: max center-corner distance
    @param pos1: 1D array with pos1: chi
    @param delta_pos1: 1D array with max pos1: max center-corner distance, unused !
    @param bins: number of output bins
    @param pos0Range: minimum and maximum  of the 2th range
    @param pos1Range: minimum and maximum  of the chi range
    @param dummy: value for bins without pixels & value of "no good" pixels
    @param delta_dummy: precision of dummy value
    @param mask: array (of int8) with masked pixels with 1 (0=not masked)
    @param dark: array (of float32) with dark noise to be subtracted (or None)
    @param flat: array (of float32) with flat image (including solid angle correctons or not...)
    @return 2theta, I, weighted histogram, unweighted histogram
    """
    cdef ssize_t  size = weights.size
    assert pos0.size == size
    assert delta_pos0.size == size
    assert  bins > 1
    cdef ssize_t   bin0_max, bin0_min, bin = 0, i, idx
    cdef float data, deltaR, deltaL, deltaA,p1, epsilon = 1e-10, cdummy = 0, ddummy = 0
    cdef float pos0_min, pos0_max, pos0_maxin, pos1_min, pos1_max, pos1_maxin, min0, max0, fbin0_min, fbin0_max
    cdef bint check_pos1 = 0, check_mask = 0, check_dummy = 0, do_dark = 0, do_flat=0

    cdef numpy.ndarray[numpy.float32_t, ndim = 1] cdata = numpy.ascontiguousarray(weights.ravel(),dtype=numpy.float32)
    cdef numpy.ndarray[numpy.float32_t, ndim = 1] cpos0, dpos0, cpos1, dpos1,cpos0_lower, cpos0_upper, cdark, cflat
    cpos0 = numpy.ascontiguousarray(pos0.ravel(), dtype=numpy.float32)
    dpos0 = numpy.ascontiguousarray(delta_pos0.ravel(), dtype=numpy.float32)


#    cdef numpy.ndarray[numpy.npy_float128, ndim = 1] outData = numpy.zeros(bins, dtype=numpy.float128)
#    cdef numpy.ndarray[numpy.npy_float128, ndim = 1] outCount = numpy.zeros(bins, dtype=numpy.float128)
    cdef numpy.ndarray[numpy.float64_t, ndim = 1] outData = numpy.zeros(bins, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim = 1] outCount = numpy.zeros(bins, dtype=numpy.float64)

    cdef numpy.ndarray[numpy.float32_t, ndim = 1] outMerge = numpy.zeros(bins, dtype=numpy.float32)
    cdef numpy.ndarray[numpy.float32_t, ndim = 1] outPos = numpy.zeros(bins, dtype=numpy.float32)
    cdef numpy.ndarray[numpy.int8_t, ndim = 1] cmask

    if  mask is not None:
        assert mask.size == size
        check_mask = 1
        cmask = numpy.ascontiguousarray(mask.ravel(),dtype=numpy.int8)

    if (dummy is not None) and delta_dummy is not None:
        check_dummy = 1
        cdummy =  float(dummy)
        ddummy =  float(delta_dummy)
    elif (dummy is not None):
        cdummy = float(dummy)
    else:
        cdummy=0.0

    if dark is not None:
        assert dark.size == size
        do_dark=1
        cdark = numpy.ascontiguousarray(dark.ravel(),dtype=numpy.float32)

    if flat is not None:
        assert flat.size == size
        do_flat=1
        cflat = numpy.ascontiguousarray(flat.ravel(),dtype=numpy.float32)


    cpos0_lower = numpy.empty(size, dtype=numpy.float32)
    cpos0_upper = numpy.empty(size, dtype=numpy.float32)

    cpos0_upper = cpos0 + dpos0
    pos0_max=cpos0_upper.max()

    cpos0_lower = cpos0 - dpos0
    pos0_min=cpos0_lower.min()

    if pos0Range is not None and len(pos0Range) > 1:
        pos0_min = min(pos0Range)
        pos0_maxin = max(pos0Range)
    else:
        pos0_maxin = pos0_max
    if pos0_min<0: pos0_min=0
    pos0_max = pos0_maxin * (1.0 + numpy.finfo(numpy.float32).eps)

    if pos1Range is not None and len(pos1Range) > 1:
        assert pos1.size == size
        assert delta_pos1.size == size
        check_pos1 = 1
        cpos1 = numpy.ascontiguousarray(pos1.ravel(),dtype=numpy.float32)
        dpos1 = numpy.ascontiguousarray(delta_pos1.ravel(),dtype=numpy.float32)
        pos1_min = min(pos1Range)
        pos1_maxin = max(pos1Range)
        pos1_max = pos1_maxin * (1 + numpy.finfo(numpy.float32).eps)

    cdef float delta = (pos0_max - pos0_min) / (< float > (bins))

    with nogil:
        for i in prange(bins):
                outPos[i] += pos0_min + (0.5 +< float > i) * delta

        for idx in prange(size):
            if (check_mask) and (cmask[idx]):
                continue

            data = cdata[idx]
            if check_dummy and (fabs(data-cdummy)<=ddummy):
                continue

            min0 = cpos0_lower[idx]
            max0 = cpos0_upper[idx]

            if check_pos1 and (((cpos1[idx]+dpos1[idx]) < pos1_min) or ((cpos1[idx]-dpos1[idx]) > pos1_max)):
                    continue

            fbin0_min = getBinNr(min0, pos0_min, delta)
            fbin0_max = getBinNr(max0, pos0_min, delta)
            bin0_min = < long > floor(fbin0_min)
            bin0_max = < long > floor(fbin0_max)

            if (bin0_max<0) or (bin0_min>=bins):
                continue
            if bin0_max>=bins:
                bin0_max=bins-1
            if  bin0_min<0:
                bin0_min=0

            if do_dark:
                data=data-cdark[idx]
            if do_flat:
                data=data-cflat[idx]

            if bin0_min == bin0_max:
                #All pixel is within a single bin
                outCount[bin0_min] +=  1.0
                outData[bin0_min] +=  data

            else: #we have pixel spliting.
                deltaA = 1.0 / (fbin0_max - fbin0_min)

                deltaL = < float > (bin0_min + 1) - fbin0_min
                deltaR = fbin0_max - (< float > bin0_max)

                outCount[bin0_min] +=  (deltaA * deltaL)
                outData[bin0_min] +=  (data * deltaA * deltaL)

                outCount[bin0_max] +=  (deltaA * deltaR)
                outData[bin0_max] +=  (data * deltaA * deltaR)

                if bin0_min + 1 < bin0_max:
                    for i in range(bin0_min + 1, bin0_max):
                        outCount[i] +=  deltaA
                        outData[i] +=  (data *  deltaA)

        for i in prange(bins):
                if outCount[i] > epsilon:
                    outMerge[i] += (outData[i] / outCount[i])
                else:
                    outMerge[i] += cdummy

    return  outPos, outMerge, outData, outCount




@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def histoBBox2d(numpy.ndarray weights not None,
                numpy.ndarray pos0 not None,
                numpy.ndarray delta_pos0 not None,
                numpy.ndarray pos1 not None,
                numpy.ndarray delta_pos1 not None,
                bins=(100, 36),
                pos0Range=None,
                pos1Range=None,
                dummy=None,
                delta_dummy=None,
                mask=None):
    """
    Calculate 2D histogram of pos0(tth),pos1(chi) weighted by weights

    Splitting is done on the pixel's bounding box like fit2D


    @param weights: array with intensities
    @param pos0: 1D array with pos0: tth or q_vect
    @param delta_pos0: 1D array with delta pos0: max center-corner distance
    @param pos1: 1D array with pos1: chi
    @param delta_pos1: 1D array with max pos1: max center-corner distance, unused !
    @param bins: number of output bins (tth=100, chi=36 by default)
    @param pos0Range: minimum and maximum  of the 2th range
    @param pos1Range: minimum and maximum  of the chi range
    @param dummy: value for bins without pixels & value of "no good" pixels
    @param delta_dummy: precision of dummy value
    @param mask: array (of int8) with masked pixels with 1 (0=not masked)
    @return  I, edges0, edges1, weighted histogram(2D), unweighted histogram (2D)
    """

    cdef long bins0, bins1, i, j, idx
    cdef long size = weights.size
    assert pos0.size == size
    assert pos1.size == size
    assert delta_pos0.size == size
    assert delta_pos1.size == size
    try:
        bins0, bins1 = tuple(bins)
    except:
        bins0 = bins1 = < long > bins
    if bins0 <= 0:
        bins0 = 1
    if bins1 <= 0:
        bins1 = 1
    cdef numpy.ndarray[numpy.float32_t, ndim = 1] cdata = numpy.ascontiguousarray(weights.ravel(),dtype=numpy.float32)
    cdef numpy.ndarray[numpy.float32_t, ndim = 1] cpos0 = numpy.ascontiguousarray(pos0.ravel(),dtype=numpy.float32)
    cdef numpy.ndarray[numpy.float32_t, ndim = 1] dpos0 = numpy.ascontiguousarray(delta_pos0.ravel(),dtype=numpy.float32)
    cdef numpy.ndarray[numpy.float32_t, ndim = 1] cpos1 = numpy.ascontiguousarray(pos1.ravel(),dtype=numpy.float32)
    cdef numpy.ndarray[numpy.float32_t, ndim = 1] dpos1 = numpy.ascontiguousarray(delta_pos1.ravel(),dtype=numpy.float32)
    cdef numpy.ndarray[numpy.float32_t, ndim = 1] cpos0_upper = numpy.zeros(size, dtype=numpy.float32)
    cdef numpy.ndarray[numpy.float32_t, ndim = 1] cpos0_lower = numpy.zeros(size, dtype=numpy.float32)
    cdef numpy.ndarray[numpy.float64_t, ndim = 2] outData = numpy.zeros((bins0, bins1), dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim = 2] outCount = numpy.zeros((bins0, bins1), dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float32_t, ndim = 2] outMerge = numpy.zeros((bins0, bins1), dtype=numpy.float32)
    cdef numpy.ndarray[numpy.float32_t, ndim = 1] edges0 = numpy.zeros(bins0, dtype=numpy.float32)
    cdef numpy.ndarray[numpy.float32_t, ndim = 1] edges1 = numpy.zeros(bins1, dtype=numpy.float32)

    cdef float min0, max0, min1, max1, deltaR, deltaL, deltaU, deltaD, deltaA, tmp, delta0, delta1
    cdef float pos0_min, pos0_max, pos1_min, pos1_max, pos0_maxin, pos1_maxin
    cdef float fbin0_min, fbin0_max, fbin1_min, fbin1_max, data, epsilon = 1e-10, cdummy, ddummy
    cdef long  bin0_max, bin0_min, bin1_max, bin1_min
    cdef int check_mask = 0, check_dummy = 0
    cdef numpy.ndarray[numpy.int8_t, ndim = 1] cmask

    if  mask is not None:
        assert mask.size == size
        check_mask = 1
        cmask = numpy.ascontiguousarray(mask.ravel(),dtype=numpy.int8)

    if (dummy is not None) and delta_dummy is not None:
        check_dummy = 1
        cdummy =  float(dummy)
        ddummy =  float(delta_dummy)
    elif (dummy is not None):
        cdummy = float(dummy)
    else:
        cdummy=0.0


    pos0_min=cpos0[0]
    pos0_max=cpos0[0]
    with nogil:
        for idx in range(size):
            min0 = cpos0[idx] - dpos0[idx]
            max0 = cpos0[idx] + dpos0[idx]
            cpos0_upper[idx]=max0
            cpos0_lower[idx]=min0
            if max0>pos0_max:
                pos0_max=max0
            if min0<pos0_min:
                pos0_min=min0

    if (pos0Range is not None) and (len(pos0Range) == 2):
        pos0_min = min(pos0Range)
        pos0_maxin = max(pos0Range)
    else:
        pos0_min = pos0_min
        pos0_maxin = pos0_max
    if pos0_min<0:
        pos0_min=0
    pos0_max = pos0_maxin * (1 + numpy.finfo(numpy.float32).eps)

    if (pos1Range is not None) and (len(pos1Range) == 2):
        pos1_min = min(pos1Range)
        pos1_maxin = max(pos1Range)
    else:
        pos1_min = cpos1.min()
        pos1_maxin = cpos1.max()
    pos1_max = pos1_maxin * (1 + numpy.finfo(numpy.float32).eps)

    delta0 = (pos0_max - pos0_min) / (< float > (bins0))
    delta1 = (pos1_max - pos1_min) / (< float > (bins1))

    with nogil:
        for i in range(bins0):
                edges0[i] = pos0_min + (0.5 + i) * delta0
        for i in range(bins1):
                edges1[i] = pos1_min + (0.5 + i) * delta1

        for idx in range(size):

            if (check_mask) and cmask[idx]:
                continue

            data = cdata[idx]
            if (check_dummy) and (fabs(data-cdummy)<=ddummy):
                continue

            data = cdata[idx]
            min0 = cpos0_lower[idx]
            max0 = cpos0_upper[idx]
            min1 = cpos1[idx] - dpos1[idx]
            max1 = cpos1[idx] + dpos1[idx]

            if (max0 < pos0_min) or (max1 < pos1_min) or (min0 > pos0_maxin) or (min1 > pos1_maxin) :
                continue

            if min0 < pos0_min:
                min0 = pos0_min
            if min1 < pos1_min:
                min1 = pos1_min
            if max0 > pos0_maxin:
                max0 = pos0_maxin
            if max1 > pos1_maxin:
                max1 = pos1_maxin


            fbin0_min = getBinNr(min0, pos0_min, delta0)
            fbin0_max = getBinNr(max0, pos0_min, delta0)
            fbin1_min = getBinNr(min1, pos1_min, delta1)
            fbin1_max = getBinNr(max1, pos1_min, delta1)

            bin0_min = < long > floor(fbin0_min)
            bin0_max = < long > floor(fbin0_max)
            bin1_min = < long > floor(fbin1_min)
            bin1_max = < long > floor(fbin1_max)


            if bin0_min == bin0_max:
                if bin1_min == bin1_max:
                    #All pixel is within a single bin
                    outCount[bin0_min, bin1_min] += 1.0
                    outData[bin0_min, bin1_min] += data
                else:
                    #spread on more than 2 bins
                    deltaD = (< float > (bin1_min + 1)) - fbin1_min
                    deltaU = fbin1_max - ( bin1_max)
                    deltaA = 1.0 / (fbin1_max - fbin1_min)

                    outCount[bin0_min, bin1_min] +=  deltaA * deltaD
                    outData[bin0_min, bin1_min] += data * deltaA * deltaD

                    outCount[bin0_min, bin1_max] +=  deltaA * deltaU
                    outData[bin0_min, bin1_max] += data * deltaA * deltaU
                    for j in range(bin1_min + 1, bin1_max):
                        outCount[bin0_min, j] +=  deltaA
                        outData[bin0_min, j] += data * deltaA

            else: #spread on more than 2 bins in dim 0
                if bin1_min == bin1_max:
                    #All pixel fall on 1 bins in dim 1
                    deltaA = 1.0 / (fbin0_max - fbin0_min)
                    deltaL = (< float > (bin0_min + 1)) - fbin0_min
                    outCount[bin0_min, bin1_min] +=  deltaA * deltaL
                    outData[bin0_min, bin1_min] +=  data * deltaA * deltaL
                    deltaR = fbin0_max - (< float > bin0_max)
                    outCount[bin0_max, bin1_min] +=  deltaA * deltaR
                    outData[bin0_max, bin1_min] +=  data * deltaA * deltaR
                    for i in range(bin0_min + 1, bin0_max):
                            outCount[i, bin1_min] +=  deltaA
                            outData[i, bin1_min] +=  data * deltaA
                else:
                    #spread on n pix in dim0 and m pixel in dim1:
                    deltaL = (< float > (bin0_min + 1)) - fbin0_min
                    deltaR = fbin0_max - (< float > bin0_max)
                    deltaD = (< float > (bin1_min + 1)) - fbin1_min
                    deltaU = fbin1_max - (< float > bin1_max)
                    deltaA = 1.0 / ((fbin0_max - fbin0_min) * (fbin1_max - fbin1_min))

                    outCount[bin0_min, bin1_min] +=  deltaA * deltaL * deltaD
                    outData[bin0_min, bin1_min] +=  data * deltaA * deltaL * deltaD

                    outCount[bin0_min, bin1_max] +=  deltaA * deltaL * deltaU
                    outData[bin0_min, bin1_max] +=  data * deltaA * deltaL * deltaU

                    outCount[bin0_max, bin1_min] +=  deltaA * deltaR * deltaD
                    outData[bin0_max, bin1_min] +=  data * deltaA * deltaR * deltaD

                    outCount[bin0_max, bin1_max] +=  deltaA * deltaR * deltaU
                    outData[bin0_max, bin1_max] +=  data * deltaA * deltaR * deltaU
                    for i in range(bin0_min + 1, bin0_max):
                            outCount[i, bin1_min] +=  deltaA * deltaD
                            outData[i, bin1_min] +=  data * deltaA * deltaD
                            for j in range(bin1_min + 1, bin1_max):
                                outCount[i, j] +=  deltaA
                                outData[i, j] +=  data * deltaA
                            outCount[i, bin1_max] +=  deltaA * deltaU
                            outData[i, bin1_max] +=  data * deltaA * deltaU
                    for j in range(bin1_min + 1, bin1_max):
                            outCount[bin0_min, j] +=  deltaA * deltaL
                            outData[bin0_min, j] +=  data * deltaA * deltaL

                            outCount[bin0_max, j] +=  deltaA * deltaR
                            outData[bin0_max, j] +=  data * deltaA * deltaR

        for i in range(bins0):
            for j in range(bins1):
                if outCount[i, j] > epsilon:
                    outMerge[i, j] = <float> (outData[i, j] / outCount[i, j])
                else:
                    outMerge[i, j] = cdummy
    return outMerge.T, edges0, edges1, outData.T, outCount.T

