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

cdef extern from "math.h":
    double floor(double)nogil
    double  fabs(double)nogil


cdef extern from "stdlib.h":
    void free(void * ptr)nogil
    void * calloc(size_t nmemb, size_t size)nogil
    void * malloc(size_t size)nogil

ctypedef numpy.int64_t DTYPE_int64_t
ctypedef numpy.float64_t DTYPE_float64_t

cdef double areaTriangle(double a0,
                         double a1,
                         double b0,
                         double b1,
                         double c0,
                         double c1):
    """
    Calculate the area of the ABC triangle with corners:
    A(a0,a1)
    B(b0,b1)
    C(c0,c1)
    @return: area, i.e. 1/2 * (B-A)^(C-A)
    """
    return 0.5 * abs(((b0 - a0) * (c1 - a1)) - ((b1 - a1) * (c0 - a0)))

cdef double areaQuad(double a0,
                     double a1,
                     double b0,
                     double b1,
                     double c0,
                     double c1,
                     double d0,
                     double d1
                     ):
    """
    Calculate the area of the ABCD quadrilataire  with corners:
    A(a0,a1)
    B(b0,b1)
    C(c0,c1)
    D(d0,d1)
    @return: area, i.e. 1/2 * (AC ^ BD)
    """
    return 0.5 * abs(((c0 - a0) * (d1 - b1)) - ((c1 - a1) * (d0 - b0)))

@cython.cdivision(True)
cdef double  getBinNr(double x0, double pos0_min, double dpos) nogil:
    """
    calculate the bin number for any point
    param x0: current position
    param pos0_min: position minimum
    param dpos: bin width
    """
    return (x0 - pos0_min) / dpos

cdef double min4f(double a, double b, double c, double d) nogil:
    if (a <= b) and (a <= c) and (a <= d):
        return a
    if (b <= a) and (b <= c) and (b <= d):
        return b
    if (c <= a) and (c <= b) and (c <= d):
        return c
    else:
        return d

cdef double max4f(double a, double b, double c, double d) nogil:
    """Calculates the max of 4 double numbers"""
    if (a >= b) and (a >= c) and (a >= d):
        return a
    if (b >= a) and (b >= c) and (b >= d):
        return b
    if (c >= a) and (c >= b) and (c >= d):
        return c
    else:
        return d

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def fullSplit1D(numpy.ndarray pos not None,
              numpy.ndarray weights not None,
              long bins=100,
              pos0Range=None,
              pos1Range=None,
              double dummy=0.0
              ):
    """
    Calculates histogram of pos weighted by weights

    Splitting is done on the pixel's bounding box like fit2D


    @param pos: 3D array with pos0; Corner A,B,C,D; tth or chi
    @param weights: array with intensities
    @param bins: number of output bins
    @param pos0Range: minimum and maximum  of the 2th range
    @param pos1Range: minimum and maximum  of the chi range
    @param dummy: value for bins without pixels
    @return 2theta, I, weighted histogram, unweighted histogram
    """

    assert pos.shape[0] == weights.size
    assert pos.shape[1] == 4
    assert pos.shape[2] == 2
    assert pos.ndim == 3
    assert  bins > 1
    cdef long  size = weights.size
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 3] cpos = pos.astype("float64")
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] cdata = weights.astype("float64").ravel()
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] outData = numpy.zeros(bins, dtype="float64")
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] outCount = numpy.zeros(bins, dtype="float64")
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] outMerge = numpy.zeros(bins, dtype="float64")
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] outPos = numpy.zeros(bins, dtype="float64")
    cdef double min0, max0, deltaR, deltaL, deltaA
    cdef double pos0_min, pos0_max, pos0_maxin, pos1_min, pos1_max, pos1_maxin
    if pos0Range is not None and len(pos0Range) > 1:
        pos0_min = min(pos0Range)
        pos0_maxin = max(pos0Range)
    else:
        pos0_min = pos[:, :, 0].min()
        pos0_maxin = pos[:, :, 0].max()
    pos0_max = pos0_maxin * (1 + numpy.finfo(numpy.double).eps)
    if pos1Range is not None and len(pos1Range) > 1:
        pos1_min = min(pos1Range)
        pos1_maxin = max(pos1Range)
    else:
        pos1_min = pos[:, :, 1].min()
        pos1_max = pos[:, :, 1].max()
    pos1_max = pos1_maxin * (1 + numpy.finfo(numpy.double).eps)
    cdef double dpos = (pos0_max - pos0_min) / (< double > (bins))
    cdef long   bin = 0
    cdef long   i, idx
    cdef double fbin0_min, fbin0_max#, fbin1_min, fbin1_max
    cdef long   bin0_max, bin0_min
    cdef double aeraPixel, a0, b0, c0, d0
    cdef double epsilon = 1e-10
    outPos = numpy.linspace(pos0_min+0.5*dpos, pos0_max-0.5*dpos, bins)
    with nogil:
#        for i in range(bins):
#                outPos[i] = pos0_min + (0.5 +< double > i) * dpos

        for idx in range(size):
            data = < double > cdata[idx]
            a0 = < double > cpos[idx, 0, 0]
            b0 = < double > cpos[idx, 1, 0]
            c0 = < double > cpos[idx, 2, 0]
            d0 = < double > cpos[idx, 3, 0]
            min0 = min4f(a0, b0, c0, d0)
            max0 = max4f(a0, b0, c0, d0)

            fbin0_min = getBinNr(min0, pos0_min, dpos)
            fbin0_max = getBinNr(max0, pos0_min, dpos)
            bin0_min = < long > floor(fbin0_min)
            bin0_max = < long > floor(fbin0_max)

            if bin0_min == bin0_max:
                #All pixel is within a single bin
                outCount[bin0_min] += 1.0
                outData[bin0_min] += data

    #        else we have pixel spliting.
            else:
                aeraPixel = fbin0_max - fbin0_min
                deltaA = 1.0 / aeraPixel

                deltaL = < double > (bin0_min + 1) - fbin0_min
                deltaR = fbin0_max - (< double > bin0_max)

                outCount[bin0_min] += deltaA * deltaL
                outData[bin0_min] += data * deltaA * deltaL

                outCount[bin0_max] += deltaA * deltaR
                outData[bin0_max] += data * deltaA * deltaR

                if bin0_min + 1 != bin0_max:
                    for i in range(bin0_min + 1, bin0_max):
                        outCount[i] += deltaA
                        outData[i] += data * deltaA

        for i in range(bins):
                if outCount[i] > epsilon:
                    outMerge[i] = outData[i] / outCount[i]
                else:
                    outMerge[i] = dummy

    return  outPos, outMerge, outData, outCount






@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def fullSplit2D(numpy.ndarray pos not None,
                numpy.ndarray weights not None,
                bins not None,
                pos0Range=None,
                pos1Range=None,
                double dummy=0.0):
    """
    Calculate 2D histogram of pos weighted by weights

    Splitting is done on the pixel's bounding box like fit2D


    @param pos: 3D array with pos0; Corner A,B,C,D; tth or chi
    @param weights: array with intensities
    @param bins: number of output bins int or 2-tuple of int
    @param pos0Range: minimum and maximum  of the 2th range
    @param pos1Range: minimum and maximum  of the chi range
    @param dummy: value for bins without pixels
    @return  I, edges0, edges1, weighted histogram(2D), unweighted histogram (2D)
    """
    cdef long  bins0, bins1, i, j, idx
    cdef long  size = weights.size
    assert pos.shape[0] == weights.size
    assert pos.shape[1] == 4 # 4 corners
    assert pos.shape[2] == 2 # tth and chi
    assert pos.ndim == 3
    try:
        bins0, bins1 = tuple(bins)
    except:
        bins0 = bins1 = < long > bins
    if bins0 <= 0:
        bins0 = 1
    if bins1 <= 0:
        bins1 = 1


    cdef numpy.ndarray[DTYPE_float64_t, ndim = 3] cpos = pos.astype("float64")
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] cdata = weights.astype("float64").ravel()
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 2] outData = numpy.zeros((bins0, bins1), dtype="float64")
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 2] outCount = numpy.zeros((bins0, bins1), dtype="float64")
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 2] outMerge = numpy.zeros((bins0, bins1), dtype="float64")
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] edges0 = numpy.zeros(bins0, dtype="float64")
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] edges1 = numpy.zeros(bins1, dtype="float64")

    cdef double min0, max0, min1, max1, deltaR, deltaL, deltaU, deltaD, deltaA
    cdef double pos0_min, pos0_max, pos1_min, pos1_max, pos0_maxin, pos1_maxin

    cdef double fbin0_min, fbin0_max, fbin1_min, fbin1_max
    cdef long   bin0_max, bin0_min, bin1_max, bin1_min
    cdef double aeraPixel, a0, a1, b0, b1, c0, c1, d0, d1
    cdef double epsilon = 1e-10

    if pos0Range is not None and len(pos0Range) == 2:
        pos0_min = min(pos0Range)
        pos0_maxin = max(pos0Range)
    else:
        pos0_min = pos[:, :, 0].min()
        pos0_maxin = pos[:, :, 0].max()
    pos0_max = pos0_maxin * (1 + numpy.finfo(numpy.double).eps)

    if pos1Range is not None and len(pos1Range) > 1:
        pos1_min = min(pos1Range)
        pos1_maxin = max(pos1Range)
    else:
        pos1_min = pos[:, :, 1].min()
        pos1_maxin = pos[:, :, 1].max()
    pos1_max = pos1_maxin * (1 + numpy.finfo(numpy.double).eps)

    cdef double dpos0 = (pos0_max - pos0_min) / (< double > (bins0))
    cdef double dpos1 = (pos1_max - pos1_min) / (< double > (bins1))

    with nogil:
        for i in range(bins0):
                edges0[i] = pos0_min + (0.5 +< double > i) * dpos0
        for i in range(bins1):
                edges1[i] = pos1_min + (0.5 +< double > i) * dpos1

        for idx in range(size):
            data = < double > cdata[idx]
            a0 = < double > cpos[idx, 0, 0]
            a1 = < double > cpos[idx, 0, 1]
            b0 = < double > cpos[idx, 1, 0]
            b1 = < double > cpos[idx, 1, 1]
            c0 = < double > cpos[idx, 2, 0]
            c1 = < double > cpos[idx, 2, 1]
            d0 = < double > cpos[idx, 3, 0]
            d1 = < double > cpos[idx, 3, 1]

            min0 = min4f(a0, b0, c0, d0)
            max0 = max4f(a0, b0, c0, d0)
            min1 = min4f(a1, b1, c1, d1)
            max1 = max4f(a1, b1, c1, d1)
#            splitOnePixel2D(min0, max0, min1, max1,
#                      data,
#                      pos0_min, pos1_min,
#                      dpos0, dpos1,
#                      outCount,
#                      outData)

            if max0 < pos0_min:
                with gil:
                    print("max0 (%s) < self.pos0_min %s" % (max0 , pos0_min))
                continue
            if max1 < pos1_min:
                with gil:
                    print("max1 (%s) < pos1_min %s" % (max0 , pos0_min))
                continue
            if min0 > pos0_maxin:
                with gil:
                    print("min0 (%s) > pos0_maxin %s" % (max0 , pos0_maxin))
                continue
            if min1 > pos1_maxin:
                with gil:
                    print("min1 (%s) > pos1_maxin %s" % (max0 , pos1_maxin))
                continue

            if min0 < pos0_min:
                data = data * (pos0_min - min0) / (max0 - min0)
                min0 = pos0_min
            if min1 < pos1_min:
                data = data * (pos1_min - min1) / (max1 - min1)
                min1 = pos1_min
            if max0 > pos0_maxin:
                data = data * (max0 - pos0_maxin) / (max0 - min0)
                max0 = pos0_maxin
            if max1 > pos1_maxin:
                data = data * (max1 - pos1_maxin) / (max1 - min1)
                max1 = pos1_maxin

##                treat data for pixel on chi discontinuity
            if ((max1 - min1) / dpos1) > (bins1 / 2.0):
#                with gil:
#                    print("max1: %s; min1: %s; dpos1: %s" % (max1 , min1, dpos1))
                if pos1_maxin - max1 > min1 - pos1_min:
                    min1 = max1
                    max1 = pos1_maxin
                else:
                    max1 = min1
                    min1 = pos1_min

            fbin0_min = getBinNr(min0, pos0_min, dpos0)
            fbin0_max = getBinNr(max0, pos0_min, dpos0)
            fbin1_min = getBinNr(min1, pos1_min, dpos1)
            fbin1_max = getBinNr(max1, pos1_min, dpos1)

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
                    aeraPixel = fbin1_max - fbin1_min
                    deltaD = (< double > (bin1_min + 1)) - fbin1_min
                    deltaU = fbin1_max - (< double > bin1_max)
                    deltaA = 1.0 / aeraPixel

                    outCount[bin0_min, bin1_min] += deltaA * deltaD
                    outData[bin0_min, bin1_min] += data * deltaA * deltaD

                    outCount[bin0_min, bin1_max] += deltaA * deltaU
                    outData[bin0_min, bin1_max] += data * deltaA * deltaU
#                    if bin1_min +1< bin1_max:
                    for j in range(bin1_min + 1, bin1_max):
                            outCount[bin0_min, j] += deltaA
                            outData[bin0_min, j] += data * deltaA

            else: #spread on more than 2 bins in dim 0
                if bin1_min == bin1_max:
                    #All pixel fall on 1 bins in dim 1
                    aeraPixel = fbin0_max - fbin0_min
                    deltaL = (< double > (bin0_min + 1)) - fbin0_min
                    deltaA = deltaL / aeraPixel
                    outCount[bin0_min, bin1_min] += deltaA
                    outData[bin0_min, bin1_min] += data * deltaA
                    deltaR = fbin0_max - (< double > bin0_max)
                    deltaA = deltaR / aeraPixel
                    outCount[bin0_max, bin1_min] += deltaA
                    outData[bin0_max, bin1_min] += data * deltaA
                    deltaA = 1.0 / aeraPixel
                    for i in range(bin0_min + 1, bin0_max):
                            outCount[i, bin1_min] += deltaA
                            outData[i, bin1_min] += data * deltaA
                else:
                    #spread on n pix in dim0 and m pixel in dim1:
                    aeraPixel = (fbin0_max - fbin0_min) * (fbin1_max - fbin1_min)
                    deltaL = (< double > (bin0_min + 1)) - fbin0_min
                    deltaR = fbin0_max - (< double > bin0_max)
                    deltaD = (< double > (bin1_min + 1)) - fbin1_min
                    deltaU = fbin1_max - (< double > bin1_max)
                    deltaA = 1.0 / aeraPixel

                    outCount[bin0_min, bin1_min] += deltaA * deltaL * deltaD
                    outData[bin0_min, bin1_min] += data * deltaA * deltaL * deltaD

                    outCount[bin0_min, bin1_max] += deltaA * deltaL * deltaU
                    outData[bin0_min, bin1_max] += data * deltaA * deltaL * deltaU

                    outCount[bin0_max, bin1_min] += deltaA * deltaR * deltaD
                    outData[bin0_max, bin1_min] += data * deltaA * deltaR * deltaD

                    outCount[bin0_max, bin1_max] += deltaA * deltaR * deltaU
                    outData[bin0_max, bin1_max] += data * deltaA * deltaR * deltaU
                    for i in range(bin0_min + 1, bin0_max):
                            outCount[i, bin1_min] += deltaA * deltaD
                            outData[i, bin1_min] += data * deltaA * deltaD
                            for j in range(bin1_min + 1, bin1_max):
                                outCount[i, j] += deltaA
                                outData[i, j] += data * deltaA
                            outCount[i, bin1_max] += deltaA * deltaU
                            outData[i, bin1_max] += data * deltaA * deltaU
                    for j in range(bin1_min + 1, bin1_max):
                            outCount[bin0_min, j] += deltaA * deltaL
                            outData[bin0_min, j] += data * deltaA * deltaL

                            outCount[bin0_max, j] += deltaA * deltaR
                            outData[bin0_max, j] += data * deltaA * deltaR

    #with nogil:
        for i in range(bins0):
            for j in range(bins1):
                if outCount[i, j] > epsilon:
                    outMerge[i, j] = outData[i, j] / outCount[i, j]
                else:
                    outMerge[i, j] = dummy
    return outMerge.T, edges0, edges1, outData.T, outCount.T

