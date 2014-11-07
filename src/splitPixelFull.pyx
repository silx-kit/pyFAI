#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Giannis Ashiotis
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
# 
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
# 
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

"""
Calculates histograms of pos0 (tth) weighted by Intensity

Splitting is done on the pixel's bounding box like fit2D, 
reverse implementation based on a sparse matrix multiplication
"""
__author__ = "Giannis Ashiotis"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "20141020"
__status__ = "stable"
__license__ = "GPLv3+"

import cython
cimport numpy
import numpy
from libc.math cimport fabs, floor
from libc.stdio cimport printf
from cython.view cimport array as cvarray

cdef double area4(double a0, double a1, double b0, double b1, double c0, double c1, double d0, double d1) nogil:
    """
    Calculate the area of the ABCD quadrilataire  with corners:
    A(a0,a1)
    B(b0,b1)
    C(c0,c1)
    D(d0,d1)
    @return: area, i.e. 1/2 * (AC ^ BD)
    """
    return 0.5 * fabs(((c0 - a0) * (d1 - b1)) - ((c1 - a1) * (d0 - b0)))

    
cdef struct Function:
    double slope
    double intersect
           
cdef double integrate(double A0, double B0, Function AB) nogil:
    """
    integrates the line defined by AB, from A0 to B0
    param A0: first limit
    param B0: second limit
    param AB: struct with the slope and point of intersection of the line
    """    
    if A0 == B0:
        return 0.0
    else:
        return AB.slope * (B0 * B0 - A0 * A0) * 0.5 + AB.intersect * (B0 - A0)
    
@cython.cdivision(True)
cdef double getBinNr(double x0, double pos0_min, double dpos) nogil:
    """
    calculate the bin number for any point
    param x0: current position
    param pos0_min: position minimum
    param dpos: bin width
    """
    return (x0 - pos0_min) / dpos


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def fullSplit1D(numpy.ndarray pos not None,
                numpy.ndarray weights not None,
                size_t bins=100,
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


    @param pos: 3D or 4D array with the coordinates of each pixel point
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
    cdef size_t  size = weights.size
    if pos.ndim > 3: #create a view
        pos = pos.reshape((-1,4,2))
    assert pos.shape[0] == size
    assert pos.shape[1] == 4
    assert pos.shape[2] == 2
    assert pos.ndim == 3
    assert bins > 1
    cdef:
        numpy.ndarray[numpy.float64_t, ndim = 3] cpos = numpy.ascontiguousarray(pos,dtype=numpy.float64)
        numpy.ndarray[numpy.float64_t, ndim = 1] cdata = numpy.ascontiguousarray(weights.ravel(), dtype=numpy.float64)
        numpy.ndarray[numpy.float64_t, ndim = 1] outData = numpy.zeros(bins, dtype=numpy.float64)
        numpy.ndarray[numpy.float64_t, ndim = 1] outCount = numpy.zeros(bins, dtype=numpy.float64)
        numpy.ndarray[numpy.float64_t, ndim = 1] outMerge = numpy.zeros(bins, dtype=numpy.float64)
        numpy.int8_t[:] cmask
        double[:] cflat, cdark, cpolarization, csolidangle
        double cdummy=0, cddummy=0, data=0
        double pos0_min=0, pos0_max=0, pos0_maxin=0, pos1_min=0, pos1_max=0, pos1_maxin=0
        double areaPixel=0, dpos=0, fbin0_min=0, fbin0_max=0#, fbin1_min, fbin1_max 
        double A0=0, B0=0, C0=0, D0=0, A1=0, B1=0, C1=0, D1=0
        double A_lim=0, B_lim=0, C_lim=0, D_lim=0
        double oneOverArea=0, partialArea=0, tmp=0
        double max0, min0, min1, max1
        Function AB, BC, CD, DA
        double epsilon=1e-10
        int lut_size = 0
        bint check_pos1=False, check_mask=False, do_dummy=False, do_dark=False, do_flat=False, do_polarization=False, do_solidangle=False
        int i=0, idx=0, bin=0, bin0, bin0_max=0, bin0_min=0, pixel_bins=0, cur_bin

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

    outPos = numpy.linspace(pos0_min + 0.5 * dpos, pos0_maxin - 0.5 * dpos, bins)

    if (dummy is not None) and (delta_dummy is not None):
        check_dummy = True
        cdummy = float(dummy)
        cddummy = float(delta_dummy)
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
            if check_dummy and ((cddummy == 0.0 and data == cdummy) or (cddummy != 0.0 and fabs(data - cdummy) <= cddummy)):
                continue

            A0 = getBinNr(< double > cpos[idx, 0, 0], pos0_min, dpos)
            A1 = < double > cpos[idx, 0, 1]
            B0 = getBinNr(< double > cpos[idx, 1, 0], pos0_min, dpos)
            B1 = < double > cpos[idx, 1, 1]
            C0 = getBinNr(< double > cpos[idx, 2, 0], pos0_min, dpos)
            C1 = < double > cpos[idx, 2, 1]
            D0 = getBinNr(< double > cpos[idx, 3, 0], pos0_min, dpos)
            D1 = < double > cpos[idx, 3, 1]

            min0 = min(A0, B0, C0, D0)
            max0 = max(A0, B0, C0, D0)
            if (max0 < 0) or (min0 >= bins):
                continue
            if check_pos1:
                min1 = min(A1, B1, C1, D1)
                max1 = max(A1, B1, C1, D1)
                if (max1 < pos1_min) or (min1 > pos1_maxin):
                    continue

            if do_dark:
                data -= cdark[idx]
            if do_flat:
                data /= cflat[idx]
            if do_polarization:
                data /= cpolarization[idx]
            if do_solidangle:
                data /= csolidangle[idx]

            bin0_min = < int > floor(min0)
            bin0_max = < int > floor(max0)

            if bin0_min == bin0_max:
                # All pixel is within a single bin
                outCount[bin0_min] += 1
                outData[bin0_min] += data
                lut_size += 1

            else:
                A0 -= bin0_min
                B0 -= bin0_min
                C0 -= bin0_min
                D0 -= bin0_min
                
                AB.slope = (B1 - A1) / (B0 - A0)
                AB.intersect = A1 - AB.slope * A0
                BC.slope = (C1 - B1) / (C0 - B0)
                BC.intersect = B1 - BC.slope * B0
                CD.slope = (D1 - C1) / (D0 - C0)
                CD.intersect = C1 - CD.slope * C0
                DA.slope = (A1 - D1) / (A0 - D0)
                DA.intersect = D1 - DA.slope * D0
                areaPixel = area4(A0, A1, B0, B1, C0, C1, D0, D1)
                oneOverPixelArea = 1.0 / areaPixel
                partialArea2 = 0.0
                for bin in range(bin0_min, bin0_max + 1):
                    bin0 = bin - bin0_min
                    A_lim = (A0<=bin0)*(A0<=(bin0+1))*bin0 + (A0>bin0)*(A0<=(bin0+1))*A0 + (A0>bin0)*(A0>(bin0+1))*(bin0+1)
                    B_lim = (B0<=bin0)*(B0<=(bin0+1))*bin0 + (B0>bin0)*(B0<=(bin0+1))*B0 + (B0>bin0)*(B0>(bin0+1))*(bin0+1)
                    C_lim = (C0<=bin0)*(C0<=(bin0+1))*bin0 + (C0>bin0)*(C0<=(bin0+1))*C0 + (C0>bin0)*(C0>(bin0+1))*(bin0+1)
                    D_lim = (D0<=bin0)*(D0<=(bin0+1))*bin0 + (D0>bin0)*(D0<=(bin0+1))*D0 + (D0>bin0)*(D0>(bin0+1))*(bin0+1)
                    partialArea = integrate(A_lim, B_lim, AB)
                    partialArea += integrate(B_lim, C_lim, BC)
                    partialArea += integrate(C_lim, D_lim, CD)
                    partialArea += integrate(D_lim, A_lim, DA)
                    tmp = fabs(partialArea) * oneOverPixelArea
                    partialArea2 += partialArea
                    outCount[bin] += tmp
                    outData[bin] += data * tmp
                    lut_size += 1

        for i in range(bins):
            if outCount[i] > epsilon:
                outMerge[i] = outData[i] / outCount[i]
            else:
                outMerge[i] = cdummy

    print lut_size
    return outPos, outMerge, outData, outCount





