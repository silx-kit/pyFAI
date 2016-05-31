# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) 2012-2016 European Synchrotron Radiation Facility, France
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

__doc__ = """Calculates histograms of pos0 (tth) weighted by Intensity

Splitting is done by full pixel splitting
Histogram (direct) implementation
"""
__author__ = "Jerome Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "31/05/2016"
__status__ = "stable"
__license__ = "GPLv3+"

import cython
cimport numpy
import numpy
from libc.math cimport fabs, ceil, floor
from libc.string cimport memset
from cython cimport view

include "regrid_common.pxi"

ctypedef double position_t
ctypedef double data_t

cdef inline position_t area4(position_t a0,
                             position_t a1,
                             position_t b0,
                             position_t b1,
                             position_t c0,
                             position_t c1,
                             position_t d0,
                             position_t d1) nogil:
    """
    Calculate the area of the ABCD polygon with 4 with corners:
    A(a0,a1)
    B(b0,b1)
    C(c0,c1)
    D(d0,d1)
    @return: area, i.e. 1/2 * (AC ^ BD)
    """
    return 0.5 * fabs(((c0 - a0) * (d1 - b1)) - ((c1 - a1) * (d0 - b0)))


cdef inline position_t calc_area(position_t I1, position_t I2, position_t slope, position_t intercept) nogil:
    "Calculate the area between I1 and I2 of a line with a given slope & intercept"
    return 0.5 * ((I2 - I1) * (slope * (I2 + I1) + 2 * intercept))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void integrate(position_t[:] buffer, int buffer_size, position_t start0, position_t start1, position_t stop0, position_t stop1) nogil:
    "Integrate in a box a line between start and stop"

    if stop0 == start0:
        #slope is infinite, area is null: no change to the buffer
        return
    cdef position_t slope, intercept
    cdef int i, istart0 = <int> floor(start0), istop0 = <int> floor(stop0)
    slope = (stop1 - start1) / (stop0 - start0)
    intercept = start1 - slope * start0
    if buffer_size > istop0 == istart0 >= 0:
        buffer[istart0] += calc_area(start0, stop0, slope, intercept)
    else:
        if stop0 > start0:
                if 0 <= start0 < buffer_size:
                    buffer[istart0] += calc_area(start0, floor(start0+1), slope, intercept)
                for i in range(max(istart0 + 1, 0), min(istop0, buffer_size)):
                    buffer[i] += calc_area(i, i+1, slope, intercept)
                if buffer_size > stop0 >= 0:
                    buffer[istop0] += calc_area(istop0, stop0, slope, intercept)
        else:
            if 0 <= start0 < buffer_size:
                buffer[istart0] += calc_area(start0, istart0, slope, intercept)
            for i in range(min(istart0, buffer_size)-1, max(<int> floor(stop0), -1), -1):
                buffer[i] += calc_area(i+1, i, slope, intercept)
            if buffer_size > stop0 >= 0:
                buffer[istop0] += calc_area(floor(stop0+1), stop0, slope, intercept)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def fullSplit1D(numpy.ndarray pos not None,
                numpy.ndarray weights not None,
                int bins=100,
                pos0Range=None,
                pos1Range=None,
                dummy=None,
                delta_dummy=None,
                mask=None,
                dark=None,
                flat=None,
                solidangle=None,
                polarization=None,
                float empty=0.0,
                double normalization_factor=1.0
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
    @param empty: value of output bins without any contribution when dummy is None
    @param normalization_factor: divide the valid result by this value

    @return 2theta, I, weighted histogram, unweighted histogram
    """
    cdef int  size = weights.size
    if pos.ndim>3: #create a view
        pos = pos.reshape((-1,4,2))
    assert pos.shape[0] == size
    assert pos.shape[1] == 4
    assert pos.shape[2] == 2
    assert pos.ndim == 3
    assert bins > 1
    cdef:
        position_t[:,:,:] cpos = numpy.ascontiguousarray(pos,dtype=numpy.float64)
        data_t[:] cdata = numpy.ascontiguousarray(weights.ravel(), dtype=numpy.float64)
        numpy.ndarray[numpy.float64_t, ndim = 1] outData = numpy.zeros(bins, dtype=numpy.float64)
        numpy.ndarray[numpy.float64_t, ndim = 1] outCount = numpy.zeros(bins, dtype=numpy.float64)
        numpy.ndarray[numpy.float64_t, ndim = 1] outMerge = numpy.zeros(bins, dtype=numpy.float64)
        numpy.int8_t[:] cmask
        data_t[:] cflat, cdark, cpolarization, csolidangle
        position_t[:] buffer

        data_t cdummy=0, cddummy=0, data=0
        position_t deltaR=0, deltaL=0, one_over_area=0
        position_t pos0_min=0, pos0_max=0, pos0_maxin=0, pos1_min=0, pos1_max=0, pos1_maxin=0
        position_t aera_pixel=0, sum_area=0, sub_area=0,  dpos=0, fbin0_min=0, fbin0_max=0
        position_t a0=0, b0=0, c0=0, d0=0, max0=0, min0=0, a1=0, b1=0, c1=0, d1=0, max1=0, min1=0
        double epsilon=1e-10

        bint check_pos1=False, check_mask=False, do_dummy=False, do_dark=False, do_flat=False, do_polarization=False, do_solidangle=False
        int i=0, b=0, idx=0, bin=0, bin0_max=0, bin0_min=0
    buffer = view.array(shape=(bins,),itemsize=sizeof(position_t), format="d")
    buffer[:] = 0

    if mask is not None:
        check_mask = True
        assert mask.size == size
        cmask = numpy.ascontiguousarray(mask.ravel(), dtype=numpy.int8)

    if pos0Range is not None and len(pos0Range) > 1:
        pos0_min = min(pos0Range)
        pos0_maxin = max(pos0Range)
    else:
        with nogil:
            for idx in range(size):
                if (check_mask) and (cmask[idx]):
                    pos0_max = pos0_min = cpos[idx, 0, 0]
                    pos1_max = pos1_min = cpos[idx, 0, 1]
                    break
            for idx in range(size):
                if (check_mask) and (cmask[idx]):
                    continue
                a0 = cpos[idx, 0, 0]
                a1 = cpos[idx, 0, 1]
                b0 = cpos[idx, 1, 0]
                b1 = cpos[idx, 1, 1]
                c0 = cpos[idx, 2, 0]
                c1 = cpos[idx, 2, 1]
                d0 = cpos[idx, 3, 0]
                d1 = cpos[idx, 3, 1]
                min0 = min(a0, b0, c0, d0)
                max0 = max(a0, b0, c0, d0)
                if max0>pos0_max:
                    pos0_max = max0
                if min0<pos0_min:
                    pos0_min = min0
                min1 = min(a1, b1, c1, d1)
                max1 = max(a1, b1, c1, d1)
                if max1>pos1_max:
                    pos1_max = max1
                if min1<pos1_min:
                    pos1_min = min1

            pos0_maxin = pos0_max
    if pos0_min<0:
        pos0_min=0
    pos0_max = pos0_maxin * EPS32

    if pos1Range is not None and len(pos1Range) > 1:
        pos1_min = min(pos1Range)
        pos1_maxin = max(pos1Range)
        check_pos1 = True
    else:
        if min1==max1==0:
            pos1_min = pos[:, :, 1].min()
            pos1_maxin = pos[:, :, 1].max()
    pos1_max = pos1_maxin * EPS32
    dpos = (pos0_max - pos0_min) / (<  double > (bins))

    outPos = numpy.linspace(pos0_min+0.5*dpos, pos0_maxin-0.5*dpos, bins)

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
        cdummy = empty
        cddummy = 0.0

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

            # a0, b0, c0 and d0 are in bin number (2theta, q or r)
            # a1, b1, c1 and d1 are in Chi angle in radians ...
            a0 = get_bin_number(cpos[idx, 0, 0], pos0_min, dpos)
            a1 = <  double > cpos[idx, 0, 1]
            b0 = get_bin_number(cpos[idx, 1, 0], pos0_min, dpos)
            b1 = <  double > cpos[idx, 1, 1]
            c0 = get_bin_number(cpos[idx, 2, 0], pos0_min, dpos)
            c1 = <  double > cpos[idx, 2, 1]
            d0 = get_bin_number(cpos[idx, 3, 0], pos0_min, dpos)
            d1 = <  double > cpos[idx, 3, 1]
            min0 = min(a0, b0, c0, d0)
            max0 = max(a0, b0, c0, d0)

            if (max0<0) or (min0 >=bins):
                continue
            if check_pos1:
                min1 = min(a1, b1, c1, d1)
                max1 = max(a1, b1, c1, d1)
                if (max1<pos1_min) or (min1 > pos1_maxin):
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
                outCount[bin0_min] += 1.0
                outData[bin0_min] += data

    #        else we have pixel splitting.
            else:
                bin0_min = max(0, bin0_min)
                bin0_max = min(bins, bin0_max + 1)
                aera_pixel = area4(a0, a1, b0, b1, c0, c1, d0, d1)
                one_over_area = 1.0 / aera_pixel

                integrate(buffer, bins, a0, a1, b0, b1) #A-B
                integrate(buffer, bins, b0, b1, c0, c1) #B-C
                integrate(buffer, bins, c0, c1, d0, d1) #C-D
                integrate(buffer, bins, d0, d1, a0, a1) #D-A

                #Distribute pixel area
                sum_area = 0.0
                for i in range(bin0_min, bin0_max):
                    sub_area = fabs(buffer[i])
                    sum_area += sub_area
                    sub_area = sub_area * one_over_area
                    outCount[i] += sub_area
                    outData[i] += sub_area * data

                #check the total area:
                if fabs(sum_area - aera_pixel) / aera_pixel>1e-6 and bin0_min != 0 and bin0_max != bins:
                    with gil:
                        print("area_pixel=%s area_sum=%s, Error= %s"%(aera_pixel,sum_area,(aera_pixel-sum_area)/aera_pixel))
                buffer[bin0_min:bin0_max] = 0
        for i in range(bins):
            if outCount[i] > epsilon:
                outMerge[i] = outData[i] / outCount[i] / normalization_factor
            else:
                outMerge[i] = cdummy

    return outPos, outMerge, outData, outCount






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
                polarization=None,
                float empty=0.0,
                double normalization_factor=1.0
                ):
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
    @param empty: value of output bins without any contribution when dummy is None
    @param normalization_factor: divide the valid result by this value

    @return  I, edges0, edges1, weighted histogram(2D), unweighted histogram (2D)
    """

    cdef int bins0 = 0, bins1 = 0, size = weights.size
    if pos.ndim > 3:  # create a view
        pos = pos.reshape((-1, 4, 2))

    assert pos.shape[0] == size
    assert pos.shape[1] == 4  # 4 corners
    assert pos.shape[2] == 2  # tth and chi
    assert pos.ndim == 3
    try:
        bins0, bins1 = tuple(bins)
    except:
        bins0 = bins1 = < int > bins
    if bins0 <= 0:
        bins0 = 1
    if bins1 <= 0:
        bins1 = 1
    cdef:
        numpy.ndarray[numpy.float64_t, ndim = 3] cpos = pos.astype(numpy.float64)
        numpy.ndarray[numpy.float64_t, ndim = 1] cdata = weights.astype(numpy.float64).ravel()
        numpy.ndarray[numpy.float64_t, ndim = 2] outData = numpy.zeros((bins0, bins1), dtype=numpy.float64)
        numpy.ndarray[numpy.float64_t, ndim = 2] outCount = numpy.zeros((bins0, bins1), dtype=numpy.float64)
        numpy.ndarray[numpy.float64_t, ndim = 2] outMerge = numpy.zeros((bins0, bins1), dtype=numpy.float64)
        numpy.ndarray[numpy.float64_t, ndim = 1] edges0 = numpy.zeros(bins0, dtype=numpy.float64)
        numpy.ndarray[numpy.float64_t, ndim = 1] edges1 = numpy.zeros(bins1, dtype=numpy.float64)
        numpy.int8_t[:] cmask
        double[:] cflat, cdark, cpolarization, csolidangle
        bint check_mask = False, do_dummy = False, do_dark = False, do_flat = False, do_polarization = False, do_solidangle = False
        double cdummy = 0, cddummy = 0, data = 0
        double min0 = 0, max0 = 0, min1 = 0, max1 = 0, deltaR = 0, deltaL = 0, deltaU = 0, deltaD = 0, one_over_area = 0
        double pos0_min = 0, pos0_max = 0, pos1_min = 0, pos1_max = 0, pos0_maxin = 0, pos1_maxin = 0
        double aera_pixel = 0, fbin0_min = 0, fbin0_max = 0, fbin1_min = 0, fbin1_max = 0
        double a0 = 0, a1 = 0, b0 = 0, b1 = 0, c0 = 0, c1 = 0, d0 = 0, d1 = 0
        double epsilon = 1e-10
        int bin0_max = 0, bin0_min = 0, bin1_max = 0, bin1_min = 0, i = 0, j = 0, idx = 0

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
    edges0 = numpy.linspace(pos0_min + 0.5 * dpos0, pos0_maxin - 0.5 * dpos0, bins0)
    edges1 = numpy.linspace(pos1_min + 0.5 * dpos1, pos1_maxin - 0.5 * dpos1, bins1)

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
        cdummy = <float> float(empty)
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

            a0 = cpos[idx, 0, 0]
            a1 = cpos[idx, 0, 1]
            b0 = cpos[idx, 1, 0]
            b1 = cpos[idx, 1, 1]
            c0 = cpos[idx, 2, 0]
            c1 = cpos[idx, 2, 1]
            d0 = cpos[idx, 3, 0]
            d1 = cpos[idx, 3, 1]

            min0 = min(a0, b0, c0, d0)
            max0 = max(a0, b0, c0, d0)
            min1 = min(a1, b1, c1, d1)
            max1 = max(a1, b1, c1, d1)

            if (max0 < pos0_min) or (min0 > pos0_maxin) or (max1 < pos1_min) or (min1 > pos1_maxin):
                    continue

            if do_dark:
                data -= cdark[idx]
            if do_flat:
                data /= cflat[idx]
            if do_polarization:
                data /= cpolarization[idx]
            if do_solidangle:
                data /= csolidangle[idx]

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

#                treat data for pixel on chi discontinuity
            if ((max1 - min1) / dpos1) > (bins1 / 2.0):
                if pos1_maxin - max1 > min1 - pos1_min:
                    min1 = max1
                    max1 = pos1_maxin
                else:
                    max1 = min1
                    min1 = pos1_min

            fbin0_min = get_bin_number(min0, pos0_min, dpos0)
            fbin0_max = get_bin_number(max0, pos0_min, dpos0)
            fbin1_min = get_bin_number(min1, pos1_min, dpos1)
            fbin1_max = get_bin_number(max1, pos1_min, dpos1)

            bin0_min = < int > fbin0_min
            bin0_max = < int > fbin0_max
            bin1_min = < int > fbin1_min
            bin1_max = < int > fbin1_max

            if bin0_min == bin0_max:
                if bin1_min == bin1_max:
                    # All pixel is within a single bin
                    outCount[bin0_min, bin1_min] += 1.0
                    outData[bin0_min, bin1_min] += data
                else:
                    # spread on more than 2 bins
                    aera_pixel = fbin1_max - fbin1_min
                    deltaD = (< double > (bin1_min + 1)) - fbin1_min
                    deltaU = fbin1_max - (< double > bin1_max)
                    one_over_area = 1.0 / aera_pixel

                    outCount[bin0_min, bin1_min] += one_over_area * deltaD
                    outData[bin0_min, bin1_min] += data * one_over_area * deltaD

                    outCount[bin0_min, bin1_max] += one_over_area * deltaU
                    outData[bin0_min, bin1_max] += data * one_over_area * deltaU
#                    if bin1_min +1< bin1_max:
                    for j in range(bin1_min + 1, bin1_max):
                            outCount[bin0_min, j] += one_over_area
                            outData[bin0_min, j] += data * one_over_area

            else:
                # spread on more than 2 bins in dim 0
                if bin1_min == bin1_max:
                    # All pixel fall on 1 bins in dim 1
                    aera_pixel = fbin0_max - fbin0_min
                    deltaL = (< double > (bin0_min + 1)) - fbin0_min
                    one_over_area = deltaL / aera_pixel
                    outCount[bin0_min, bin1_min] += one_over_area
                    outData[bin0_min, bin1_min] += data * one_over_area
                    deltaR = fbin0_max - (< double > bin0_max)
                    one_over_area = deltaR / aera_pixel
                    outCount[bin0_max, bin1_min] += one_over_area
                    outData[bin0_max, bin1_min] += data * one_over_area
                    one_over_area = 1.0 / aera_pixel
                    for i in range(bin0_min + 1, bin0_max):
                            outCount[i, bin1_min] += one_over_area
                            outData[i, bin1_min] += data * one_over_area
                else:
                    # spread on n pix in dim0 and m pixel in dim1:
                    aera_pixel = (fbin0_max - fbin0_min) * (fbin1_max - fbin1_min)
                    deltaL = (< double > (bin0_min + 1.0)) - fbin0_min
                    deltaR = fbin0_max - (< double > bin0_max)
                    deltaD = (< double > (bin1_min + 1.0)) - fbin1_min
                    deltaU = fbin1_max - (< double > bin1_max)
                    one_over_area = 1.0 / aera_pixel

                    outCount[bin0_min, bin1_min] += one_over_area * deltaL * deltaD
                    outData[bin0_min, bin1_min] += data * one_over_area * deltaL * deltaD

                    outCount[bin0_min, bin1_max] += one_over_area * deltaL * deltaU
                    outData[bin0_min, bin1_max] += data * one_over_area * deltaL * deltaU

                    outCount[bin0_max, bin1_min] += one_over_area * deltaR * deltaD
                    outData[bin0_max, bin1_min] += data * one_over_area * deltaR * deltaD

                    outCount[bin0_max, bin1_max] += one_over_area * deltaR * deltaU
                    outData[bin0_max, bin1_max] += data * one_over_area * deltaR * deltaU
                    for i in range(bin0_min + 1, bin0_max):
                            outCount[i, bin1_min] += one_over_area * deltaD
                            outData[i, bin1_min] += data * one_over_area * deltaD
                            for j in range(bin1_min + 1, bin1_max):
                                outCount[i, j] += one_over_area
                                outData[i, j] += data * one_over_area
                            outCount[i, bin1_max] += one_over_area * deltaU
                            outData[i, bin1_max] += data * one_over_area * deltaU
                    for j in range(bin1_min + 1, bin1_max):
                            outCount[bin0_min, j] += one_over_area * deltaL
                            outData[bin0_min, j] += data * one_over_area * deltaL

                            outCount[bin0_max, j] += one_over_area * deltaR
                            outData[bin0_max, j] += data * one_over_area * deltaR

    # with nogil:
        for i in range(bins0):
            for j in range(bins1):
                if outCount[i, j] > epsilon:
                    outMerge[i, j] = outData[i, j] / outCount[i, j] / normalization_factor
                else:
                    outMerge[i, j] = cdummy
    return outMerge.T, edges0, edges1, outData.T, outCount.T
