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
from libc.math cimport fabs

#cdef double areaTriangle(double a0,
#                         double a1,
#                         double b0,
#                         double b1,
#                         double c0,
#                         double c1):
#    """
#    Calculate the area of the ABC triangle with corners:
#    A(a0,a1)
#    B(b0,b1)
#    C(c0,c1)
#    @return: area, i.e. 1/2 * (B-A)^(C-A)
#    """
#    return 0.5 * abs(((b0 - a0) * (c1 - a1)) - ((b1 - a1) * (c0 - a0)))
#
#cdef double areaQuad(double a0,
#                     double a1,
#                     double b0,
#                     double b1,
#                     double c0,
#                     double c1,
#                     double d0,
#                     double d1
#                     ):
#    """
#    Calculate the area of the ABCD quadrilataire  with corners:
#    A(a0,a1)
#    B(b0,b1)
#    C(c0,c1)
#    D(d0,d1)
#    @return: area, i.e. 1/2 * (AC ^ BD)
#    """
#    return 0.5 * abs(((c0 - a0) * (d1 - b1)) - ((c1 - a1) * (d0 - b0)))

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
                ssize_t bins=100,
                pos0Range=None,
                pos1Range=None,
                dummy=None,
                delta_dummy=None,
                mask=None,
                dark=None,
                flat=None,
                solidangle=None,
                polarization=None
              ):
    """
    Calculates histogram of pos weighted by weights

    Splitting is done on the pixel's bounding box like fit2D.
    No compromise for speed has been made here.


    @param pos: 3D array with pos0; Corner A,B,C,D; tth or chi
    @param weights: array with intensities
    @param bins: number of output bins
    @param pos0Range: minimum and maximum  of the 2th range
    @param pos1Range: minimum and maximum  of the chi range
    @param dummy: value for bins without pixels
    @param delta_dummy: precision of dummy value
    @param mask: array (of int8) with masked pixels with 1 (0=not masked)
    @param dark: array (of float64) with dark noise to be subtracted (or None)
    @param flat: array (of float64) with flat image
    @param polarization: array (of float64) with polarization correction
    @param solidangle: array (of float64) with flat image
    @return 2theta, I, weighted histogram, unweighted histogram
    """
    cdef ssize_t  size = weights.size
    if pos.ndim>3: #create a view
        pos = pos.reshape((-1,4,2))
    assert pos.shape[0] == size
    assert pos.shape[1] == 4
    assert pos.shape[2] == 2
    assert pos.ndim == 3
    assert  bins > 1

    cdef numpy.ndarray[numpy.float64_t, ndim = 3] cpos = numpy.ascontiguousarray(pos,dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim = 1] cdata = numpy.ascontiguousarray(weights.ravel(), dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim = 1] outData = numpy.zeros(bins, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim = 1] outCount = numpy.zeros(bins, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim = 1] outMerge = numpy.zeros(bins, dtype=numpy.float64)
    cdef numpy.int8_t[:] cmask
    cdef double[:] cflat, cdark, cpolarization, csolidangle
    cdef double cdummy, cddummy, data
    cdef double deltaR, deltaL, deltaA
    cdef double pos0_min, pos0_max, pos0_maxin, pos1_min, pos1_max, pos1_maxin
    cdef bint check_pos1=False, check_mask=False, do_dummy=False, do_dark=False, do_flat=False, do_polarization=False, do_solidangle=False
    cdef ssize_t i, idx
    cdef double fbin0_min, fbin0_max#, fbin1_min, fbin1_max
    cdef ssize_t bin, bin0_max, bin0_min
    cdef double aeraPixel, dpos, a0, b0, c0, d0, max0, min0, a1, b1, c1, d1, max1, min1,
    cdef double epsilon

    if pos0Range is not None and len(pos0Range) > 1:
        pos0_min = min(pos0Range)
        pos0_maxin = max(pos0Range)
    else:
        pos0_min = pos[:, :, 0].min()
        pos0_maxin = pos[:, :, 0].max()
    pos0_max = pos0_maxin * (1 + numpy.finfo(numpy.float32).eps)
    if pos1Range is not None and len(pos1Range) > 1:
        pos1_min = min(pos1Range)
        pos1_maxin = max(pos1Range)
        do_pos1 = True
    else:
        pos1_min = pos[:, :, 1].min()
        pos1_maxin = pos[:, :, 1].max()
    pos1_max = pos1_maxin * (1 + numpy.finfo(numpy.float32).eps)
    dpos = (pos0_max - pos0_min) / (< double > (bins))
    bin = 0
    epsilon = 1e-10

    outPos = numpy.linspace(pos0_min+0.5*dpos, pos0_maxin-0.5*dpos, bins)

    if (dummy is not None) and (delta_dummy is not None):
        check_dummy = True
        cdummy =  float(dummy)
        cddummy =  float(delta_dummy)
    elif (dummy is not None):
        check_dummy = True
        cdummy = float(dummy)
        cddummy = 0.0
    else:
        check_dummy = False
        cdummy = 0.0
        cddummy = 0.0

    if mask is not None:
        check_mask = True
        assert mask.size == size
        cmask = numpy.ascontiguousarray(mask.ravel(), dtype=numpy.int8)
    if dark is not None:
        do_dark = True
        assert dark.size == size
        cdark = numpy.ascontiguousarray(dark.ravel(), dtype=numpy.float64)
    if flat is not None:
        do_flat = True
        assert flat.size == size
        cflat = numpy.ascontiguousarray(flat.ravel(), dtype=numpy.float64)
    if polarization is not None:
        do_polarization = True
        assert polarization.size == size
        cpolarization = numpy.ascontiguousarray(polarization.ravel(), dtype=numpy.float64)
    if solidangle is not None:
        do_solidangle = True
        assert solidangle.size == size
        csolidangle = numpy.ascontiguousarray(solidangle.ravel(), dtype=numpy.float64)

    with nogil:
        for idx in range(size):

            if (check_mask) and (cmask[idx]):
                continue

            data = cdata[idx]
            if check_dummy and ( (cddummy==0.0 and data==cdummy) or (cddummy!=0.0 and fabs(data-cdummy)<=cddummy)):
                continue

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
            if (max0<pos0_min) or (min0 > pos0_maxin):
                continue
            if check_pos1:
                min1 = min4f(a1, b1, c1, d1)
                max1 = max4f(a1, b1, c1, d1)
                if (max1<pos1_min) or (min1 > pos1_maxin):
                    continue
            if min0<pos0_min:
                min0=pos0_min
            if max0>pos1_maxin:
                max0=pos0_maxin

            if do_dark:
                data -= cdark[idx]
            if do_flat:
                data /= cflat[idx]
            if do_polarization:
                data /= cpolarization[idx]
            if do_solidangle:
                data /= csolidangle[idx]

            fbin0_min = getBinNr(min0, pos0_min, dpos)
            fbin0_max = getBinNr(max0, pos0_min, dpos)
            bin0_min = < ssize_t > fbin0_min
            bin0_max = < ssize_t > fbin0_max

            if bin0_min == bin0_max:
                #All pixel is within a single bin
                outCount[bin0_min] += 1.0
                outData[bin0_min] += data

    #        else we have pixel spliting.
            else:
                aeraPixel = fbin0_max - fbin0_min
                deltaA = 1.0 / aeraPixel

                deltaL = <double>(bin0_min) + 1.0 - fbin0_min
                deltaR = fbin0_max - <double>(bin0_max)

                tmp = deltaA * deltaL
                outCount[bin0_min] += tmp
                outData[bin0_min] += data * tmp

                tmp = deltaA * deltaR
                outCount[bin0_max] += tmp
                outData[bin0_max] += data * tmp

                if bin0_min + 1 != bin0_max:
                    for i in range(bin0_min + 1, bin0_max):
                        outCount[i] += deltaA
                        outData[i] += data * deltaA


        for i in range(bins):
            if outCount[i] > epsilon:
                outMerge[i] = outData[i] / outCount[i]
            else:
                outMerge[i] = cdummy

    return  outPos, outMerge, outData, outCount






@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def fullSplit2D(numpy.ndarray pos not None,
                numpy.ndarray weights not None,
                bins not None,
                pos0Range=None,
                pos1Range=None,
                dummy=None,
                delta_dummy=None,
                mask=None,
                dark=None,
                flat=None,
                solidangle=None,
                polarization=None):
    """
    Calculate 2D histogram of pos weighted by weights

    Splitting is done on the pixel's bounding box like fit2D


    @param pos: 3D array with pos0; Corner A,B,C,D; tth or chi
    @param weights: array with intensities
    @param bins: number of output bins int or 2-tuple of int
    @param pos0Range: minimum and maximum  of the 2th range
    @param pos1Range: minimum and maximum  of the chi range
    @param dummy: value for bins without pixels
    @param delta_dummy: precision of dummy value
    @param mask: array (of int8) with masked pixels with 1 (0=not masked)
    @param dark: array (of float64) with dark noise to be subtracted (or None)
    @param flat: array (of float64) with flat-field image
    @param polarization: array (of float64) with polarization correction
    @param solidangle: array (of float64)with solid angle corrections
    @return  I, edges0, edges1, weighted histogram(2D), unweighted histogram (2D)
    """

    cdef ssize_t size = weights.size
    cdef ssize_t  bins0, bins1, i, j, idx
    if pos.ndim>3: #create a view
        pos = pos.reshape((-1,4,2))

    assert pos.shape[0] == size
    assert pos.shape[1] == 4 # 4 corners
    assert pos.shape[2] == 2 # tth and chi
    assert pos.ndim == 3
    try:
        bins0, bins1 = tuple(bins)
    except:
        bins0 = bins1 = < ssize_t > bins
    if bins0 <= 0:
        bins0 = 1
    if bins1 <= 0:
        bins1 = 1


    cdef numpy.ndarray[numpy.float64_t, ndim = 3] cpos = pos.astype(numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim = 1] cdata = weights.astype(numpy.float64).ravel()
    cdef numpy.ndarray[numpy.float64_t, ndim = 2] outData = numpy.zeros((bins0, bins1), dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim = 2] outCount = numpy.zeros((bins0, bins1), dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim = 2] outMerge = numpy.zeros((bins0, bins1), dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim = 1] edges0 = numpy.zeros(bins0, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim = 1] edges1 = numpy.zeros(bins1, dtype=numpy.float64)
    cdef bint check_mask=False, do_dummy=False, do_dark=False, do_flat=False, do_polarization=False, do_solidangle=False
    cdef numpy.int8_t[:] cmask
    cdef double[:] cflat, cdark, cpolarization, csolidangle
    cdef double cdummy, cddummy, data

    cdef double min0, max0, min1, max1, deltaR, deltaL, deltaU, deltaD, deltaA
    cdef double pos0_min, pos0_max, pos1_min, pos1_max, pos0_maxin, pos1_maxin

    cdef double fbin0_min, fbin0_max, fbin1_min, fbin1_max
    cdef ssize_t   bin0_max, bin0_min, bin1_max, bin1_min
    cdef double aeraPixel, a0, a1, b0, b1, c0, c1, d0, d1
    cdef double epsilon = 1e-10

    if pos0Range is not None and len(pos0Range) == 2:
        pos0_min = min(pos0Range)
        pos0_maxin = max(pos0Range)
    else:
        pos0_min = pos[:, :, 0].min()
        pos0_maxin = pos[:, :, 0].max()
    pos0_max = pos0_maxin * (1 + numpy.finfo(numpy.float32).eps)

    if pos1Range is not None and len(pos1Range) > 1:
        pos1_min = min(pos1Range)
        pos1_maxin = max(pos1Range)
    else:
        pos1_min = pos[:, :, 1].min()
        pos1_maxin = pos[:, :, 1].max()
    pos1_max = pos1_maxin * (1 + numpy.finfo(numpy.float32).eps)

    cdef double dpos0 = (pos0_max - pos0_min) / (< double > (bins0))
    cdef double dpos1 = (pos1_max - pos1_min) / (< double > (bins1))
    edges0 = numpy.linspace(pos0_min+0.5*dpos0, pos0_maxin-0.5*dpos0, bins0)
    edges1 = numpy.linspace(pos1_min+0.5*dpos1, pos0_maxin-0.5*dpos1, bins1)

    if (dummy is not None) and (delta_dummy is not None):
        check_dummy = True
        cdummy =  float(dummy)
        cddummy =  float(delta_dummy)
    elif (dummy is not None):
        check_dummy = True
        cdummy = float(dummy)
        cddummy = 0.0
    else:
        check_dummy = False
        cdummy = 0.0
        cddummy = 0.0

    if mask is not None:
        check_mask = True
        assert mask.size == size
        cmask = numpy.ascontiguousarray(mask.ravel(), dtype=numpy.int8)
    if dark is not None:
        do_dark = True
        assert dark.size == size
        cdark = numpy.ascontiguousarray(dark.ravel(), dtype=numpy.float64)
    if flat is not None:
        do_flat = True
        assert flat.size == size
        cflat = numpy.ascontiguousarray(flat.ravel(), dtype=numpy.float64)
    if polarization is not None:
        do_polarization = True
        assert polarization.size == size
        cpolarization = numpy.ascontiguousarray(polarization.ravel(), dtype=numpy.float64)
    if solidangle is not None:
        do_solidangle = True
        assert solidangle.size == size
        csolidangle = numpy.ascontiguousarray(solidangle.ravel(), dtype=numpy.float64)

    with nogil:
        for idx in range(size):


            if (check_mask) and (cmask[idx]):
                continue

            data = cdata[idx]
            if check_dummy and ( (cddummy==0.0 and data==cdummy) or (cddummy!=0.0 and fabs(data-cdummy)<=cddummy)):
                continue

            a0 =  cpos[idx, 0, 0]
            a1 =  cpos[idx, 0, 1]
            b0 =  cpos[idx, 1, 0]
            b1 =  cpos[idx, 1, 1]
            c0 =  cpos[idx, 2, 0]
            c1 =  cpos[idx, 2, 1]
            d0 =  cpos[idx, 3, 0]
            d1 =  cpos[idx, 3, 1]

            min0 = min4f(a0, b0, c0, d0)
            max0 = max4f(a0, b0, c0, d0)
            min1 = min4f(a1, b1, c1, d1)
            max1 = max4f(a1, b1, c1, d1)

            if (max0<pos0_min) or (min0 > pos0_maxin) or (max1<pos1_min) or (min1 > pos1_maxin):
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

            bin0_min = < ssize_t > fbin0_min
            bin0_max = < ssize_t > fbin0_max
            bin1_min = < ssize_t > fbin1_min
            bin1_max = < ssize_t > fbin1_max


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
                    deltaL = (< double > (bin0_min + 1.0)) - fbin0_min
                    deltaR = fbin0_max - (< double > bin0_max)
                    deltaD = (< double > (bin1_min + 1.0)) - fbin1_min
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
                    outMerge[i, j] = cdummy
    return outMerge.T, edges0, edges1, outData.T, outCount.T

