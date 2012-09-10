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

#
#import cython
#cimport numpy
import numpy
#
#cdef extern from "math.h":
#    float floor(float)nogil
#    float fabs(float)nogil
from math import  floor
#@cython.cdivision(True)
def getBinNr(x0, pos0_min, delta):
    """
    calculate the bin number for any point
    param x0: current position
    param pos0_min: position minimum
    param delta: bin width
    """
    return (x0 - pos0_min) / delta


def histoBBox1d(weights ,
                pos0,
                delta_pos0,
                pos1=None,
                delta_pos1=None,
                bins=100,
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
    size = weights.size
    assert pos0.size == size
    assert delta_pos0.size == size
    assert  bins > 1
    bin = 0
    epsilon = 1e-10
    cdummy = 0
    ddummy = 0

    check_pos1 = 0
    check_mask = 0
    check_dummy = 0
    do_dark = 0
    do_flat = 0

    cdata = numpy.ascontiguousarray(weights.ravel(), dtype=numpy.float32)
    cpos0 = numpy.ascontiguousarray(pos0.ravel(), dtype=numpy.float32)
    dpos0 = numpy.ascontiguousarray(delta_pos0.ravel(), dtype=numpy.float32)


    outData = numpy.zeros(bins, dtype=numpy.float64)
    outCount = numpy.zeros(bins, dtype=numpy.float64)
    outMax = numpy.zeros(bins, dtype=numpy.int64)
    outMerge = numpy.zeros(bins, dtype=numpy.float32)
    outPos = numpy.zeros(bins, dtype=numpy.float32)

    if  mask is not None:
        assert mask.size == size
        check_mask = 1
        cmask = numpy.ascontiguousarray(mask.ravel(), dtype=numpy.int8)

    if (dummy is not None) and delta_dummy is not None:
        check_dummy = 1
        cdummy = float(dummy)
        ddummy = float(delta_dummy)
    elif (dummy is not None):
        cdummy = float(dummy)
    else:
        cdummy = 0.0

    if dark is not None:
        assert dark.size == size
        do_dark = 1
        cdark = numpy.ascontiguousarray(dark.ravel(), dtype=numpy.float32)

    if flat is not None:
        assert flat.size == size
        do_flat = 1
        cflat = numpy.ascontiguousarray(flat.ravel(), dtype=numpy.float32)


    cpos0_lower = numpy.zeros(size, dtype=numpy.float32)
    cpos0_upper = numpy.zeros(size, dtype=numpy.float32)
    pos0_min = cpos0[0]
    pos0_max = cpos0[0]
    for idx in range(size):
            min0 = cpos0[idx] - dpos0[idx]
            max0 = cpos0[idx] + dpos0[idx]
            cpos0_upper[idx] = max0
            cpos0_lower[idx] = min0
            if max0 > pos0_max:
                pos0_max = max0
            if min0 < pos0_min:
                pos0_min = min0

    if pos0Range is not None and len(pos0Range) > 1:
        pos0_min = min(pos0Range)
        pos0_maxin = max(pos0Range)
    else:
        pos0_maxin = pos0_max
    if pos0_min < 0: pos0_min = 0
    pos0_max = pos0_maxin * (1.0 + numpy.finfo(numpy.float32).eps)

    if pos1Range is not None and len(pos1Range) > 1:
        assert pos1.size == size
        assert delta_pos1.size == size
        check_pos1 = 1
        cpos1 = numpy.ascontiguousarray(pos1.ravel(), dtype=numpy.float32)
        dpos1 = numpy.ascontiguousarray(delta_pos1.ravel(), dtype=numpy.float32)
        pos1_min = min(pos1Range)
        pos1_maxin = max(pos1Range)
        pos1_max = pos1_maxin * (1 + numpy.finfo(numpy.float32).eps)

    delta = (pos0_max - pos0_min) / ((bins))

    for i in range(bins):
                outPos[i] = pos0_min + (0.5 + i) * delta

    for idx in range(size):
            if (check_mask) and (cmask[idx]):
                continue

            data = cdata[idx]
            if check_dummy and (abs(data - cdummy) <= ddummy):
                continue

            min0 = cpos0_lower[idx]
            max0 = cpos0_upper[idx]

            if check_pos1 and (((cpos1[idx] + dpos1[idx]) < pos1_min) or ((cpos1[idx] - dpos1[idx]) > pos1_max)):
                    continue

            fbin0_min = getBinNr(min0, pos0_min, delta)
            fbin0_max = getBinNr(max0, pos0_min, delta)
            bin0_min = int(floor(fbin0_min))
            bin0_max = int(floor(fbin0_max))

            if (bin0_max < 0) or (bin0_min >= bins):
                continue
            if bin0_max >= bins:
                bin0_max = bins - 1
            if  bin0_min < 0:
                bin0_min = 0

            if do_dark:
                data -= cdark[idx]
            if do_flat:
                data /= cflat[idx]

            if bin0_min == bin0_max:
                #All pixel is within a single bin
                outCount[bin0_min] += 1.0
                outData[bin0_min] += data
                outMax[bin0_min] += 1

            else: #we have pixel spliting.
                deltaA = 1.0 / (fbin0_max - fbin0_min)

                deltaL = (bin0_min + 1) - fbin0_min
                deltaR = fbin0_max - (bin0_max)

                outCount[bin0_min] += (deltaA * deltaL)
                outData[bin0_min] += (data * deltaA * deltaL)
                outMax[bin0_min] += 1

                outCount[bin0_max] += (deltaA * deltaR)
                outData[bin0_max] += (data * deltaA * deltaR)
                outMax[bin0_max] += 1
                if bin0_min + 1 < bin0_max:
                    for i in range(bin0_min + 1, bin0_max):
                        outCount[i] += deltaA
                        outData[i] += (data * deltaA)
                        outMax[i] += 1

    for i in range(bins):
                if outCount[i] > epsilon:
                    outMerge[i] = (outData[i] / outCount[i])
                else:
                    outMerge[i] = cdummy

    return  outPos, outMerge, outData, outCount, outMax




