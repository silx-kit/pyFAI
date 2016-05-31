# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) 2014-2016 European Synchrotron Radiation Facility, Grenoble, France
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


"""Full pixel Splitting implemented using Sparse-matrix Dense-Vector multiplication,
Sparse matrix represented using the CompressedSparseRow.
"""
__author__ = "Jerome Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "31/05/2016"
__status__ = "stable"
__license__ = "MIT"

import cython
import os
import sys
from cython.parallel import prange
from libc.string cimport memset
import numpy
cimport numpy
from libc.math cimport fabs, floor, sqrt
from libc.stdio cimport printf, fflush, stdout

include "regrid_common.pxi"

try:
    from fastcrc import crc32
except:
    from zlib import crc32


cdef struct Function:
    float slope
    float intersect

cdef float area4(float a0, float a1, float b0, float b1, float c0, float c1, float d0, float d1) nogil:
    """
    Calculate the area of the ABCD quadrilataire  with corners:
    A(a0,a1)
    B(b0,b1)
    C(c0,c1)
    D(d0,d1)
    @return: area, i.e. 1/2 * (AC ^ BD)
    """
    return 0.5 * fabs(((c0 - a0) * (d1 - b1)) - ((c1 - a1) * (d0 - b0)))


@cython.cdivision(True)
cdef inline float getBin1Nr(float x0, float pos0_min, float delta, float var) nogil:
    """
    calculate the bin number for any point
    param x0: current position
    param pos0_min: position minimum
    param delta: bin width
    """
    if var:
        if x0 >= 0:
            return (x0 - pos0_min) / delta
        else:
            return (x0 + 2 * pi - pos0_min) / delta   # temporary fix....
    else:
        return (x0 - pos0_min) / delta


cdef inline float integrate(float A0, float B0, Function AB) nogil:
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


cdef struct MyPoint:
    float i
    float j


cdef struct MyPoly:
    int size
    MyPoint[8] data


@cython.cdivision(True)
cdef inline MyPoint ComputeIntersection0(MyPoint S, MyPoint E, float clipEdge) nogil:
    cdef MyPoint intersection
    intersection.i = clipEdge
    intersection.j = (E.j - S.j) * (clipEdge - S.i) / (E.i - S.i) + S.j
    return intersection


@cython.cdivision(True)
cdef inline MyPoint ComputeIntersection1(MyPoint S, MyPoint E, float clipEdge) nogil:
    cdef MyPoint intersection
    intersection.i = (E.i - S.i) * (clipEdge - S.j) / (E.j - S.j) + S.i
    intersection.j = clipEdge
    return intersection


cdef inline int point_and_line(float x0, float y0, float x1, float y1, float x, float y) nogil:
    cdef float tmp = (y - y0) * (x1 - x0) - (x - x0) * (y1 - y0)
    return (tmp > 0) - (tmp < 0)


cdef float area_n(MyPoly poly) nogil:
    if poly.size is 3:
            return 0.5*fabs(poly.data[0].i*poly.data[1].j+poly.data[1].i*poly.data[2].j+poly.data[2].i*poly.data[0].j-
                           poly.data[1].i*poly.data[0].j-poly.data[2].i*poly.data[1].j-poly.data[0].i*poly.data[2].j)
    elif poly.size is 4:
            return 0.5*fabs(poly.data[0].i*poly.data[1].j+poly.data[1].i*poly.data[2].j+poly.data[2].i*poly.data[3].j+poly.data[3].i*poly.data[0].j-
                           poly.data[1].i*poly.data[0].j-poly.data[2].i*poly.data[1].j-poly.data[3].i*poly.data[2].j-poly.data[0].i*poly.data[3].j)
    elif poly.size is 5:
            return 0.5*fabs(poly.data[0].i*poly.data[1].j+poly.data[1].i*poly.data[2].j+poly.data[2].i*poly.data[3].j+poly.data[3].i*poly.data[4].j+poly.data[4].i*poly.data[0].j-
                           poly.data[1].i*poly.data[0].j-poly.data[2].i*poly.data[1].j-poly.data[3].i*poly.data[2].j-poly.data[4].i*poly.data[3].j-poly.data[0].i*poly.data[4].j)
    elif poly.size is 6:
            return 0.5*fabs(poly.data[0].i*poly.data[1].j+poly.data[1].i*poly.data[2].j+poly.data[2].i*poly.data[3].j+poly.data[3].i*poly.data[4].j+poly.data[4].i*poly.data[5].j+poly.data[5].i*poly.data[0].j-
                           poly.data[1].i*poly.data[0].j-poly.data[2].i*poly.data[1].j-poly.data[3].i*poly.data[2].j-poly.data[4].i*poly.data[3].j-poly.data[5].i*poly.data[4].j-poly.data[0].i*poly.data[5].j)
    elif poly.size is 7:
            return 0.5*fabs(poly.data[0].i*poly.data[1].j+poly.data[1].i*poly.data[2].j+poly.data[2].i*poly.data[3].j+poly.data[3].i*poly.data[4].j+poly.data[4].i*poly.data[5].j+poly.data[5].i*poly.data[6].j+poly.data[6].i*poly.data[0].j-
                           poly.data[1].i*poly.data[0].j-poly.data[2].i*poly.data[1].j-poly.data[3].i*poly.data[2].j-poly.data[4].i*poly.data[3].j-poly.data[5].i*poly.data[4].j-poly.data[6].i*poly.data[5].j-poly.data[0].i*poly.data[6].j)
    elif poly.size is 8:
            return 0.5*fabs(poly.data[0].i*poly.data[1].j+poly.data[1].i*poly.data[2].j+poly.data[2].i*poly.data[3].j+poly.data[3].i*poly.data[4].j+poly.data[4].i*poly.data[5].j+poly.data[5].i*poly.data[6].j+poly.data[6].i*poly.data[7].j+poly.data[7].i*poly.data[0].j-
                           poly.data[1].i*poly.data[0].j-poly.data[2].i*poly.data[1].j-poly.data[3].i*poly.data[2].j-poly.data[4].i*poly.data[3].j-poly.data[5].i*poly.data[4].j-poly.data[6].i*poly.data[5].j-poly.data[7].i*poly.data[6].j-poly.data[0].i*poly.data[7].j)


cdef inline int on_boundary(float A, float B, float C, float D) nogil:
    """
    Check if we are on a discontinuity ....
    """
    return (((A > piover2) and (B > piover2) and (C < -piover2) and (D < -piover2)) or
            ((A < -piover2) and (B < -piover2) and (C > piover2) and (D > piover2)) or
            ((A > piover2) and (B < -piover2) and (C > piover2) and (D < -piover2)) or
            ((A < -piover2) and (B > piover2) and (C < -piover2) and (D > piover2)) or
            ((A > piover2) and (B < -piover2) and (C < -piover2) and (D > piover2)) or
            ((A < -piover2) and (B > piover2) and (C > piover2) and (D < -piover2)))


class FullSplitCSR_1d(object):
    """
    Now uses CSR (Compressed Sparse raw) with main attributes:
    * nnz: number of non zero elements
    * data: coefficient of the matrix in a 1D vector of float32
    * indices: Column index position for the data (same size as
    * indptr: row pointer indicates the start of a given row. len nrow+1

    Nota: nnz = indptr[-1]
    """
    @cython.boundscheck(False)
    def __init__(self,
                 numpy.ndarray pos not None,
                 int bins=100,
                 pos0Range=None,
                 pos1Range=None,
                 mask=None,
                 mask_checksum=None,
                 allow_pos0_neg=False,
                 unit="undefined",
                 empty=None):
        """
        @param pos: 3D or 4D array with the coordinates of each pixel point
        @param bins: number of output bins, 100 by default
        @param pos0Range: minimum and maximum  of the 2th range
        @param pos1Range: minimum and maximum  of the chi range
        @param mask: array (of int8) with masked pixels with 1 (0=not masked)
        @param allow_pos0_neg: enforce the q<0 is usually not possible
        @param unit: can be 2th_deg or r_nm^-1 ...
        @param empty: value of output bins without any contribution when dummy is None

        """

#        self.padding = int(padding)
        if pos.ndim>3: #create a view
            pos = pos.reshape((-1,4,2))
        assert pos.shape[1] == 4
        assert pos.shape[2] == 2
        assert pos.ndim == 3
        self.pos = pos
        self.size = pos.shape[0]
        self.bins = bins
        #self.bad_pixel = bad_pixel
        self.lut_size = 0
        self.allow_pos0_neg = allow_pos0_neg
        self.empty = empty or 0.0
        if  mask is not None:
            assert mask.size == self.size
            self.check_mask = True
            self.cmask = numpy.ascontiguousarray(mask.ravel(), dtype=numpy.int8)
            if mask_checksum:
                self.mask_checksum = mask_checksum
            else:
                self.mask_checksum = crc32(mask)
        else:
            self.check_mask = False
            self.mask_checksum = None
        self.data = self.nnz = self.indices = self.indptr = None
        self.pos0Range = pos0Range
        self.pos1Range = pos1Range

        self.calc_lut()
        self.outPos = numpy.linspace(self.pos0_min+0.5*self.delta, self.pos0_maxin-0.5*self.delta, self.bins)
        self.lut_checksum = crc32(self.data)
        self.unit=unit
        self.lut=(self.data,self.indices,self.indptr)
        self.lut_nbytes = sum([i.nbytes for i in self.lut])

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def calc_lut(self):
        cdef:
            numpy.ndarray[numpy.float64_t, ndim = 3] cpos = numpy.ascontiguousarray(self.pos,dtype=numpy.float64)
            numpy.int8_t[:] cmask
            numpy.ndarray[numpy.int32_t, ndim = 1] outMax = numpy.zeros(self.bins, dtype=numpy.int32)
            numpy.ndarray[numpy.int32_t, ndim = 1] indptr = numpy.zeros(self.bins+1, dtype=numpy.int32)
            float pos0_min=0, pos0_max=0, pos0_maxin=0, pos1_min=0, pos1_max=0, pos1_maxin=0
            float max0, min0
            float areaPixel=0, delta=0, areaPixel2=0
            float A0=0, B0=0, C0=0, D0=0, A1=0, B1=0, C1=0, D1=0
            float A_lim=0, B_lim=0, C_lim=0, D_lim=0
            float oneOverArea=0, partialArea=0, tmp=0
            Function AB, BC, CD, DA
            int bins, i=0, idx=0, bin=0, bin0=0, bin0_max=0, bin0_min=0, bin1_min, pixel_bins=0, k=0, size=0
            bint check_pos1=False, check_mask=False

        bins = self.bins
        if self.pos0Range is not None and len(self.pos0Range) > 1:
            self.pos0_min = min(self.pos0Range)
            self.pos0_maxin = max(self.pos0Range)
        else:
            self.pos0_min = self.pos[:, :, 0].min()
            self.pos0_maxin = self.pos[:, :, 0].max()
        self.pos0_max = self.pos0_maxin * (1 + numpy.finfo(numpy.float32).eps)
        if self.pos1Range is not None and len(self.pos1Range) > 1:
            self.pos1_min = min(self.pos1Range)
            self.pos1_maxin = max(self.pos1Range)
            self.check_pos1 = True
        else:
            self.pos1_min = self.pos[:, :, 1].min()
            self.pos1_maxin = self.pos[:, :, 1].max()
        self.pos1_max = self.pos1_maxin * (1 + numpy.finfo(numpy.float32).eps)

        self.delta = (self.pos0_max - self.pos0_min) / (< float > (bins))

        pos0_min = self.pos0_min
        pos0_max = self.pos0_max
        pos1_min = self.pos1_min
        pos1_max = self.pos1_max
        delta = self.delta

        size = self.size
        check_mask = self.check_mask
        if check_mask:
            cmask = self.cmask

        with nogil:
            for idx in range(size):
                if (check_mask) and (cmask[idx]):
                    continue

                A0 = get_bin_number(< float > cpos[idx, 0, 0], pos0_min, delta)
                A1 = < float > cpos[idx, 0, 1]
                B0 = get_bin_number(< float > cpos[idx, 1, 0], pos0_min, delta)
                B1 = < float > cpos[idx, 1, 1]
                C0 = get_bin_number(< float > cpos[idx, 2, 0], pos0_min, delta)
                C1 = < float > cpos[idx, 2, 1]
                D0 = get_bin_number(< float > cpos[idx, 3, 0], pos0_min, delta)
                D1 = < float > cpos[idx, 3, 1]

                min0 = min(A0, B0, C0, D0)
                max0 = max(A0, B0, C0, D0)

                if (max0 < 0) or (min0 >= bins):
                    continue
                if check_pos1:
                    if (max(A1, B1, C1, D1) < pos1_min) or (min(A1, B1, C1, D1) > pos1_maxin):
                        continue

                bin0_min = < int > floor(min0)
                bin0_max = < int > floor(max0)

                for bin in range(bin0_min, bin0_max+1):
                    outMax[bin] += 1

        indptr[1:] = outMax.cumsum()
        self.indptr = indptr

        cdef:
            numpy.ndarray[numpy.int32_t, ndim = 1] indices = numpy.zeros(indptr[bins], dtype=numpy.int32)
            numpy.ndarray[numpy.float32_t, ndim = 1] data = numpy.zeros(indptr[bins], dtype=numpy.float32)

        #just recycle the outMax array
        outMax[:] = 0

        with nogil:
            for idx in range(size):

                if (check_mask) and (cmask[idx]):
                    continue

                A0 = get_bin_number(< float > cpos[idx, 0, 0], pos0_min, delta)
                A1 = < float > cpos[idx, 0, 1]
                B0 = get_bin_number(< float > cpos[idx, 1, 0], pos0_min, delta)
                B1 = < float > cpos[idx, 1, 1]
                C0 = get_bin_number(< float > cpos[idx, 2, 0], pos0_min, delta)
                C1 = < float > cpos[idx, 2, 1]
                D0 = get_bin_number(< float > cpos[idx, 3, 0], pos0_min, delta)
                D1 = < float > cpos[idx, 3, 1]

                min0 = min(A0, B0, C0, D0)
                max0 = max(A0, B0, C0, D0)

                if (max0 < 0) or (min0 >= bins):
                    continue
                if check_pos1:
                    if (max(A1, B1, C1, D1) < pos1_min) or (min(A1, B1, C1, D1) > pos1_maxin):
                        continue

                bin0_min = < int > floor(min0)
                bin0_max = < int > floor(max0)

                if bin0_min == bin0_max:
                    #All pixel is within a single bin
                    k = outMax[bin0_min]
                    indices[indptr[bin0_min] + k] = idx
                    data[indptr[bin0_min] + k] = 1.0
                    outMax[bin0_min] += 1  # k+1
                else:
                    # else we have pixel spliting.
                    # offseting the min bin of the pixel to be zero to avoid percision problems
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

                    #with gil:
                        #print "Area with the 4 point formula: %e" % areaPixel
                        #print " "

                    areaPixel2  = integrate(A0, B0, AB)
                    areaPixel2 += integrate(B0, C0, BC)
                    areaPixel2 += integrate(C0, D0, CD)
                    areaPixel2 += integrate(D0, A0, DA)

                    oneOverPixelArea = 1.0 / areaPixel

                    for bin in range(bin0_min, bin0_max+1):
                        bin0 = bin - bin0_min
                        A_lim = (A0<=bin0)*(A0<=(bin0+1))*bin0 + (A0>bin0)*(A0<=(bin0+1))*A0 + (A0>bin0)*(A0>(bin0+1))*(bin0+1)
                        B_lim = (B0<=bin0)*(B0<=(bin0+1))*bin0 + (B0>bin0)*(B0<=(bin0+1))*B0 + (B0>bin0)*(B0>(bin0+1))*(bin0+1)
                        C_lim = (C0<=bin0)*(C0<=(bin0+1))*bin0 + (C0>bin0)*(C0<=(bin0+1))*C0 + (C0>bin0)*(C0>(bin0+1))*(bin0+1)
                        D_lim = (D0<=bin0)*(D0<=(bin0+1))*bin0 + (D0>bin0)*(D0<=(bin0+1))*D0 + (D0>bin0)*(D0>(bin0+1))*(bin0+1)


                        # khan summation
                        #area_sum = 0.0
                        #corr = 0.0

                        #partialArea = integrate(A_lim, B_lim, AB)
                        #y = partialArea - corr
                        #t = area_sum + y
                        #corr = (t - area_sum) - y
                        #area_sum = t

                        #partialArea = integrate(B_lim, C_lim, BC)
                        #y = partialArea - corr
                        #t = area_sum + y
                        #corr = (t - area_sum) - y
                        #area_sum = t

                        #partialArea = integrate(C_lim, D_lim, CD)
                        #y = partialArea - corr
                        #t = area_sum + y
                        #corr = (t - area_sum) - y
                        #area_sum = t

                        #partialArea = integrate(D_lim, A_lim, DA)
                        #y = partialArea - corr
                        #t = area_sum + y
                        #corr = (t - area_sum) - y
                        #area_sum = t

                        #tmp = fabs(area_sum) * oneOverPixelArea

                        partialArea  = integrate(A_lim, B_lim, AB)
                        partialArea += integrate(B_lim, C_lim, BC)
                        partialArea += integrate(C_lim, D_lim, CD)
                        partialArea += integrate(D_lim, A_lim, DA)

                        #tmp = fabs(partialArea) * oneOverPixelArea
                        #with gil:
                            #print "Partial Area: %e" % fabs(partialArea)
                            #print "Contribution: %e" % tmp
                            #print "  "

                        k = outMax[bin]
                        indices[indptr[bin] + k] = idx
                        data[indptr[bin] + k] = fabs(partialArea) * oneOverPixelArea
                        outMax[bin] += 1 # k+1

        self.data = data
        self.indices = indices
        self.outMax = outMax

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def integrate(self,
                  weights,
                  dummy=None,
                  delta_dummy=None,
                  dark=None,
                  flat=None,
                  solidAngle=None,
                  polarization=None,
                  double normalization_factor=1.0):
        """
        Actually perform the integration which in this case looks more like a matrix-vector product

        @param weights: input image
        @type weights: ndarray
        @param dummy: value for dead pixels (optional)
        @type dummy: float
        @param delta_dummy: precision for dead-pixel value in dynamic masking
        @type delta_dummy: float
        @param dark: array with the dark-current value to be subtracted (if any)
        @type dark: ndarray
        @param flat: array with the dark-current value to be divided by (if any)
        @type flat: ndarray
        @param solidAngle: array with the solid angle of each pixel to be divided by (if any)
        @type solidAngle: ndarray
        @param polarization: array with the polarization correction values to be divided by (if any)
        @type polarization: ndarray
        @param normalization_factor: divide the valid result by this value

        @return : positions, pattern, weighted_histogram and unweighted_histogram
        @rtype: 4-tuple of ndarrays

        """
        cdef:
            numpy.int32_t i=0, j=0, idx=0, bins=self.bins, size=self.size
            float sum_data=0.0, sum_count=0.0, epsilon=1e-10
            float data=0, coef=0, cdummy=0, cddummy=0
            bint do_dummy=False, do_dark=False, do_flat=False, do_polarization=False, do_solidAngle=False
            numpy.ndarray[numpy.float64_t, ndim = 1] outData = numpy.zeros(self.bins, dtype=numpy.float64)
            numpy.ndarray[numpy.float64_t, ndim = 1] outCount = numpy.zeros(self.bins, dtype=numpy.float64)
            numpy.ndarray[numpy.float32_t, ndim = 1] outMerge = numpy.zeros(self.bins, dtype=numpy.float32)
            float[:] ccoef = self.data, cdata, tdata, cflat, cdark, csolidAngle, cpolarization

            numpy.int32_t[:] indices = self.indices, indptr = self.indptr
        assert size == weights.size

        if dummy is not None:
            do_dummy = True
            cdummy =  <float>float(dummy)
            if delta_dummy is None:
                cddummy = <float>0.0
            else:
                cddummy = <float>float(delta_dummy)
        else:
            do_dummy = False
            cdummy =  <float>float(self.empty)
        if flat is not None:
            do_flat = True
            assert flat.size == size
            cflat = numpy.ascontiguousarray(flat.ravel(), dtype=numpy.float32)
        if dark is not None:
            do_dark = True
            assert dark.size == size
            cdark = numpy.ascontiguousarray(dark.ravel(), dtype=numpy.float32)
        if solidAngle is not None:
            do_solidAngle = True
            assert solidAngle.size == size
            csolidAngle = numpy.ascontiguousarray(solidAngle.ravel(), dtype=numpy.float32)
        if polarization is not None:
            do_polarization = True
            assert polarization.size == size
            cpolarization = numpy.ascontiguousarray(polarization.ravel(), dtype=numpy.float32)

        if (do_dark + do_flat + do_polarization + do_solidAngle):
            tdata = numpy.ascontiguousarray(weights.ravel(), dtype=numpy.float32)
            cdata = numpy.zeros(size,dtype=numpy.float32)
            if do_dummy:
                for i in prange(size, nogil=True, schedule="static"):
                    data = tdata[i]
                    if ((cddummy!=0) and (fabs(data-cdummy) > cddummy)) or ((cddummy==0) and (data!=cdummy)):
                        #Nota: -= and /= operatore are seen as reduction in cython parallel.
                        if do_dark:
                            data = data - cdark[i]
                        if do_flat:
                            data = data / cflat[i]
                        if do_polarization:
                            data = data / cpolarization[i]
                        if do_solidAngle:
                            data = data / csolidAngle[i]
                        cdata[i] += data
                    else: # set all dummy_like values to cdummy. simplifies further processing
                        cdata[i] += cdummy
            else:
                for i in prange(size, nogil=True, schedule="static"):
                    data = tdata[i]
                    if do_dark:
                        data = data - cdark[i]
                    if do_flat:
                        data = data / cflat[i]
                    if do_polarization:
                        data = data / cpolarization[i]
                    if do_solidAngle:
                        data = data / csolidAngle[i]
                    cdata[i] += data
        else:
            if do_dummy:
                tdata = numpy.ascontiguousarray(weights.ravel(), dtype=numpy.float32)
                cdata = numpy.zeros(size, dtype=numpy.float32)
                for i in prange(size, nogil=True, schedule="static"):
                    data = tdata[i]
                    if ((cddummy != 0) and (fabs(data - cdummy) > cddummy)) or ((cddummy == 0) and (data != cdummy)):
                        cdata[i] += data
                    else:
                        cdata[i] += cdummy
            else:
                cdata = numpy.ascontiguousarray(weights.ravel(), dtype=numpy.float32)

        for i in prange(bins, nogil=True, schedule="guided"):
            sum_data = 0.0
            sum_count = 0.0
            for j in range(indptr[i], indptr[i+1]):
                idx = indices[j]
                coef = ccoef[j]
                if coef == 0.0:
                    continue
                data = cdata[idx]
                if do_dummy and (data == cdummy):
                    continue
                sum_data = sum_data + coef * data
                sum_count = sum_count + coef
            outData[i] += sum_data
            outCount[i] += sum_count
            if sum_count > epsilon:
                outMerge[i] += <float> (sum_data / sum_count / normalization_factor)
            else:
                outMerge[i] += cdummy
        return  self.outPos, outMerge, outData, outCount


################################################################################
# Bidimensionnal regrouping
################################################################################


class FullSplitCSR_2d(object):
    """
    Now uses CSR (Compressed Sparse raw) with main attributes:
    * nnz: number of non zero elements
    * data: coefficient of the matrix in a 1D vector of float32
    * indices: Column index position for the data (same size as
    * indptr: row pointer indicates the start of a given row. len nrow+1

    Nota: nnz = indptr[-1]
    """
    @cython.boundscheck(False)
    def __init__(self,
                 numpy.ndarray pos not None,
                 bins=(100, 36),
                 pos0Range=None,
                 pos1Range=None,
                 mask=None,
                 mask_checksum=None,
                 allow_pos0_neg=False,
                 unit="undefined",
                 empty=None):

        """
        @param pos: 3D or 4D array with the coordinates of each pixel point
        @param bins: number of output bins (tth=100, chi=36 by default)
        @param pos0Range: minimum and maximum  of the 2th range
        @param pos1Range: minimum and maximum  of the chi range
        @param mask: array (of int8) with masked pixels with 1 (0=not masked)
        @param allow_pos0_neg: enforce the q<0 is usually not possible
        @param unit: can be 2th_deg or r_nm^-1 ...
        @param empty: value for bins where no pixels are contributing
        """

        if pos.ndim > 3:  # create a view
            pos = pos.reshape((-1, 4, 2))
        assert pos.shape[1] == 4
        assert pos.shape[2] == 2
        assert pos.ndim == 3
        self.pos = pos
        self.size = pos.shape[0]
        self.bins = bins
        #self.bad_pixel = bad_pixel
        self.lut_size = 0
        self.empty = empty or 0.0
        self.allow_pos0_neg = allow_pos0_neg
        if  mask is not None:
            assert mask.size == self.size
            self.check_mask = True
            self.cmask = numpy.ascontiguousarray(mask.ravel(), dtype=numpy.int64)
            if mask_checksum:
                self.mask_checksum = mask_checksum
            else:
                self.mask_checksum = crc32(mask)
        else:
            self.check_mask = False
            self.mask_checksum = None
        self.data = self.nnz = self.indices = self.indptr = None
        self.pos0Range = pos0Range
        self.pos1Range = pos1Range

        self.calc_lut()
        #self.outPos = numpy.linspace(self.pos0_min+0.5*self.delta, self.pos0_maxin-0.5*self.delta, self.bins)
        self.lut_checksum = crc32(self.data)

        self.unit = unit

        self.lut = (self.data, self.indices, self.indptr)
        self.lut_nbytes = sum([i.nbytes for i in self.lut])

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def calc_lut(self):
        cdef:
            numpy.ndarray[numpy.float64_t, ndim = 3] cpos = numpy.ascontiguousarray(self.pos, dtype=numpy.float64)
            numpy.int8_t[:] cmask
            numpy.ndarray[numpy.int32_t, ndim = 2] outMax = numpy.zeros(self.bins, dtype=numpy.int32)
            numpy.ndarray[numpy.int32_t, ndim = 1] indptr = numpy.zeros((self.bins[0] * self.bins[1]) + 1, dtype=numpy.int32)
            float pos0_min=0, pos0_max=0, pos0_maxin=0, pos1_min=0, pos1_max=0, pos1_maxin=0
            float max0, min0, min1, max1
            float areaPixel=0, delta0=0, delta1=0, areaPixel2=0
            float A0=0, B0=0, C0=0, D0=0, A1=0, B1=0, C1=0, D1=0
            float A_lim=0, B_lim=0, C_lim=0, D_lim=0
            float oneOverArea=0, partialArea=0, tmp_f=0, var=0
            Function AB, BC, CD, DA
            MyPoint A, B, C, D, S, E
            MyPoly list1, list2
            int bins0, bins1, i=0, j=0, idx=0, bin=0, bin0=0, bin1=0, bin0_max=0, bin0_min=0, bin1_min=0, bin1_max=0, k=0, size=0
            int all_bins0=self.bins[0], all_bins1=self.bins[1], all_bins=self.bins[0]*self.bins[1], pixel_bins=0, tmp_i, index
            bint check_pos1=False, check_mask=False

        bins = self.bins
        if self.pos0Range is not None and len(self.pos0Range) > 1:
            self.pos0_min = min(self.pos0Range)
            self.pos0_maxin = max(self.pos0Range)
        else:
            self.pos0_min = self.pos[:, :, 0].min()
            self.pos0_maxin = self.pos[:, :, 0].max()
        self.pos0_max = self.pos0_maxin * (1 + numpy.finfo(numpy.float32).eps)
        if self.pos1Range is not None and len(self.pos1Range) > 1:
            self.pos1_min = min(self.pos1Range)
            self.pos1_maxin = max(self.pos1Range)
            self.check_pos1 = True
        else:
            self.pos1_min = self.pos[:, :, 1].min()
            self.pos1_maxin = self.pos[:, :, 1].max()
        self.pos1_max = self.pos1_maxin * (1 + numpy.finfo(numpy.float32).eps)

        self.delta0 = (self.pos0_max - self.pos0_min) / (< float > (all_bins0))
        self.delta1 = (self.pos1_max - self.pos1_min) / (< float > (all_bins1))

        pos0_min = self.pos0_min
        pos0_max = self.pos0_max
        pos1_min = self.pos1_min
        pos1_max = self.pos1_max
        delta0 = self.delta0
        delta1 = self.delta1

        size = self.size
        check_mask = self.check_mask
        if check_mask:
            cmask = self.cmask

        cdef numpy.ndarray[numpy.int8_t, ndim = 2] is_inside = numpy.zeros((< int > (1.5*sqrt(size)/all_bins0) ,< int > (1.5*sqrt(size)/all_bins1)), dtype=numpy.int8)

        with nogil:
            for idx in range(size):
                if (check_mask) and (cmask[idx]):
                    continue

                A0 = get_bin_number(< float > cpos[idx, 0, 0], pos0_min, delta0)
                B0 = get_bin_number(< float > cpos[idx, 1, 0], pos0_min, delta0)
                C0 = get_bin_number(< float > cpos[idx, 2, 0], pos0_min, delta0)
                D0 = get_bin_number(< float > cpos[idx, 3, 0], pos0_min, delta0)

                var = on_boundary(cpos[idx, 0, 1], cpos[idx, 1, 1], cpos[idx, 2, 1], cpos[idx, 3, 1])
                A1 = getBin1Nr(< float > cpos[idx, 0, 1], pos1_min, delta1, var)
                B1 = getBin1Nr(< float > cpos[idx, 1, 1], pos1_min, delta1, var)
                C1 = getBin1Nr(< float > cpos[idx, 2, 1], pos1_min, delta1, var)
                D1 = getBin1Nr(< float > cpos[idx, 3, 1], pos1_min, delta1, var)

                min0 = min(A0, B0, C0, D0)
                max0 = max(A0, B0, C0, D0)
                min1 = min(A1, B1, C1, D1)
                max1 = max(A1, B1, C1, D1)

                if (max0<0) or (min0 >= all_bins0) or (max1<0): # or (min1 >= all_bins1+2):
                    continue

                bin0_min = < int > floor(min0)
                bin0_max = < int > floor(max0)
                bin1_min = < int > floor(min1)
                bin1_max = < int > floor(max1)

                if bin0_min == bin0_max:
                    if bin1_min == bin1_max:
                        outMax[bin0_min,bin1_min] += 1
                    else:
                        for bin in range(bin1_min, bin1_max+1):
                            outMax[bin0_min,bin] += 1
                elif bin1_min == bin1_max:
                    for bin in range(bin0_min, bin0_max+1):
                        outMax[bin,bin1_min] += 1
                else:
                    bins0 = bin0_max - bin0_min + 1
                    bins1 = bin1_max - bin1_min + 1

                    A0 -= bin0_min
                    A1 -= bin1_min
                    B0 -= bin0_min
                    B1 -= bin1_min
                    C0 -= bin0_min
                    C1 -= bin1_min
                    D0 -= bin0_min
                    D1 -= bin1_min

                    #perimeter skipped
                    for i in range(1, bins0):
                        for j in range(1, bins1):
                            tmp_i  = point_and_line(A0, A1, B0, B1, i, j)
                            tmp_i += point_and_line(B0, B1, C0, C1, i, j)
                            tmp_i += point_and_line(C0, C1, D0, D1, i, j)
                            tmp_i += point_and_line(D0, D1, A0, A1, i, j)
                            is_inside[i, j] = (< int > fabs(tmp_i)) / < int > 4

                    for i in range(bins0):
                        for j in range(bins1):
                            tmp_i  = is_inside[i, j]
                            tmp_i += is_inside[i, j + 1]
                            tmp_i += is_inside[i + 1, j]
                            tmp_i += is_inside[i + 1, j + 1]
                            if tmp_i is not 0:
                                outMax[i + bin0_min, j + bin1_min] += 1

        indptr[1:] = outMax.ravel().cumsum()
        self.indptr = indptr

        cdef numpy.ndarray[numpy.int32_t, ndim = 1] indices = numpy.zeros(indptr[all_bins], dtype=numpy.int32)
        cdef numpy.ndarray[numpy.float32_t, ndim = 1] data = numpy.zeros(indptr[all_bins], dtype=numpy.float32)

        #just recycle the outMax array
        outMax[:] = 0

        with nogil:
            for idx in range(size):
                if (check_mask) and (cmask[idx]):
                    continue

                A0 = get_bin_number(< float > cpos[idx, 0, 0], pos0_min, delta0)
                B0 = get_bin_number(< float > cpos[idx, 1, 0], pos0_min, delta0)
                C0 = get_bin_number(< float > cpos[idx, 2, 0], pos0_min, delta0)
                D0 = get_bin_number(< float > cpos[idx, 3, 0], pos0_min, delta0)

                var = on_boundary(cpos[idx, 0, 1], cpos[idx, 1, 1], cpos[idx, 2, 1], cpos[idx, 3, 1])
                A1 = getBin1Nr(< float > cpos[idx, 0, 1], pos1_min, delta1, var)
                B1 = getBin1Nr(< float > cpos[idx, 1, 1], pos1_min, delta1, var)
                C1 = getBin1Nr(< float > cpos[idx, 2, 1], pos1_min, delta1, var)
                D1 = getBin1Nr(< float > cpos[idx, 3, 1], pos1_min, delta1, var)

                min0 = min(A0, B0, C0, D0)
                max0 = max(A0, B0, C0, D0)
                min1 = min(A1, B1, C1, D1)
                max1 = max(A1, B1, C1, D1)

                if (max0 < 0) or (min0 >= all_bins0) or (max1<0):  # or (min1 >= all_bins1 + 2 ):
                    continue

                bin0_min = < int > floor(min0)
                bin0_max = < int > floor(max0)
                bin1_min = < int > floor(min1)
                bin1_max = < int > floor(max1)

                if bin0_min == bin0_max:
                    if bin1_min == bin1_max:
                        # Whole pixel is within a single bin
                        k = outMax[bin0_min, bin1_min]
                        index = bin0_min * all_bins1 + bin1_min
                        if index > all_bins:
                            printf("0 index = %d > %d!! \n",index, all_bins)
                            fflush(stdout)
                        if indptr[index] > indptr[all_bins]:
                            printf("0 indptr = %d > %d!! \n", indptr[index], indptr[all_bins])
                            fflush(stdout)
                        indices[indptr[index] + k] = idx
                        data[indptr[index] + k] = 1.0
                        outMax[bin0_min, bin1_min] += 1 #k+1
                    else:
                        #printf("  1 %d  %d \n",bin1_min,bin1_max)
                        #fflush(stdout)
                        # transpose previous code
                        #A0 -= bin0_min
                        A1 -= bin1_min
                        #B0 -= bin0_min
                        B1 -= bin1_min
                        #C0 -= bin0_min
                        C1 -= bin1_min
                        #D0 -= bin0_min
                        D1 -= bin1_min

                        AB.slope=(B0-A0)/(B1-A1)
                        AB.intersect= A0 - AB.slope*A1
                        BC.slope=(C0-B0)/(C1-B1)
                        BC.intersect= B0 - BC.slope*B1
                        CD.slope=(D0-C0)/(D1-C1)
                        CD.intersect= C0 - CD.slope*C1
                        DA.slope=(A0-D0)/(A1-D1)
                        DA.intersect= D0 - DA.slope*D1

                        areaPixel = area4(A0, A1, B0, B1, C0, C1, D0, D1)
                        oneOverPixelArea = 1.0 / areaPixel

                        #for bin in range(bin0_min, bin0_max+1):
                        for bin1 in range(bin1_max+1 - bin1_min):
                            #bin1 = bin - bin1_min
                            A_lim = (A1<=bin1)*(A1<=(bin1+1))*bin1 + (A1>bin1)*(A1<=(bin1+1))*A1 + (A1>bin1)*(A1>(bin1+1))*(bin1+1)
                            B_lim = (B1<=bin1)*(B1<=(bin1+1))*bin1 + (B1>bin1)*(B1<=(bin1+1))*B1 + (B1>bin1)*(B1>(bin1+1))*(bin1+1)
                            C_lim = (C1<=bin1)*(C1<=(bin1+1))*bin1 + (C1>bin1)*(C1<=(bin1+1))*C1 + (C1>bin1)*(C1>(bin1+1))*(bin1+1)
                            D_lim = (D1<=bin1)*(D1<=(bin1+1))*bin1 + (D1>bin1)*(D1<=(bin1+1))*D1 + (D1>bin1)*(D1>(bin1+1))*(bin1+1)

                            partialArea  = integrate(A_lim, B_lim, AB)
                            partialArea += integrate(B_lim, C_lim, BC)
                            partialArea += integrate(C_lim, D_lim, CD)
                            partialArea += integrate(D_lim, A_lim, DA)

                            k = outMax[bin0_min, bin1_min+bin1]
                            index = bin0_min * all_bins1 + bin1_min + bin1
                            if index > all_bins:
                                printf("1 index = %d > %d!! \n",index,all_bins)
                                fflush(stdout)
                            if indptr[index] > indptr[all_bins]:
                                printf("1 indptr = %d > %d!! \n",indptr[index],indptr[all_bins])
                                fflush(stdout)
                            indices[indptr[index] + k] = idx
                            data[indptr[index]+k] = fabs(partialArea) * oneOverPixelArea
                            outMax[bin0_min,bin1_min+bin1] += 1 #k+1

                elif bin1_min == bin1_max:
                    A0 -= bin0_min
                    #A1 -= bin1_min
                    B0 -= bin0_min
                    #B1 -= bin1_min
                    C0 -= bin0_min
                    #C1 -= bin1_min
                    D0 -= bin0_min
                    #D1 -= bin1_min

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

                    #for bin in range(bin0_min, bin0_max+1):
                    for bin0 in range(bin0_max+1 - bin0_min):
                        #bin0 = bin - bin0_min
                        A_lim = (A0<=bin0)*(A0<=(bin0+1))*bin0 + (A0>bin0)*(A0<=(bin0+1))*A0 + (A0>bin0)*(A0>(bin0+1))*(bin0+1)
                        B_lim = (B0<=bin0)*(B0<=(bin0+1))*bin0 + (B0>bin0)*(B0<=(bin0+1))*B0 + (B0>bin0)*(B0>(bin0+1))*(bin0+1)
                        C_lim = (C0<=bin0)*(C0<=(bin0+1))*bin0 + (C0>bin0)*(C0<=(bin0+1))*C0 + (C0>bin0)*(C0>(bin0+1))*(bin0+1)
                        D_lim = (D0<=bin0)*(D0<=(bin0+1))*bin0 + (D0>bin0)*(D0<=(bin0+1))*D0 + (D0>bin0)*(D0>(bin0+1))*(bin0+1)

                        partialArea  = integrate(A_lim, B_lim, AB)
                        partialArea += integrate(B_lim, C_lim, BC)
                        partialArea += integrate(C_lim, D_lim, CD)
                        partialArea += integrate(D_lim, A_lim, DA)

                        k = outMax[bin0_min+bin0,bin1_min]
                        index = (bin0_min+bin0)*all_bins1 + bin1_min
                        if index > all_bins:
                            printf("2 index = %d > %d!! \n",index,all_bins)
                            fflush(stdout)
                        if indptr[index] > indptr[all_bins]:
                            printf("2 indptr = %d > %d!! \n",indptr[index],indptr[all_bins])
                            fflush(stdout)
                        indices[indptr[index]+k] = idx
                        data[indptr[index]+k] = fabs(partialArea) * oneOverPixelArea
                        outMax[bin0_min+bin0,bin1_min] += 1 #k+1

                else:
                    #printf("  1 %d  %d \n",bin1_min,bin1_max)
                    #fflush(stdout)

                    bins0 = bin0_max - bin0_min + 1
                    bins1 = bin1_max - bin1_min + 1

                    A0 -= bin0_min
                    A1 -= bin1_min
                    B0 -= bin0_min
                    B1 -= bin1_min
                    C0 -= bin0_min
                    C1 -= bin1_min
                    D0 -= bin0_min
                    D1 -= bin1_min

                    areaPixel = area4(A0, A1, B0, B1, C0, C1, D0, D1)
                    oneOverPixelArea = 1.0 / areaPixel

                    #perimeter skipped - not inside for sure
                    for i in range(1,bins0):
                        for j in range(1,bins1):
                            tmp_i  = point_and_line(A0,A1,B0,B1,i,j)
                            tmp_i += point_and_line(B0,B1,C0,C1,i,j)
                            tmp_i += point_and_line(C0,C1,D0,D1,i,j)
                            tmp_i += point_and_line(D0,D1,A0,A1,i,j)
                            is_inside[i,j] = (< int > fabs(tmp_i)) / < int > 4

                    for i in range(bins0):
                        for j in range(bins1):
                            tmp_i  = is_inside[i,j]
                            tmp_i += is_inside[i,j+1]
                            tmp_i += is_inside[i+1,j]
                            tmp_i += is_inside[i+1,j+1]
                            if tmp_i is 4:
                                k = outMax[bin0_min+i,bin1_min+j]
                                index = (i+bin0_min)*all_bins1 + j+bin1_min
                                if index > all_bins:
                                    printf("3 index = %d > %d!! \n",index,all_bins)
                                    fflush(stdout)
                                if indptr[index] > indptr[all_bins]:
                                    printf("3 indptr = %d > %d!! \n",indptr[index],indptr[all_bins])
                                    fflush(stdout)
                                indices[indptr[index]+k] = idx
                                data[indptr[index]+k] = oneOverPixelArea
                                outMax[bin0_min+i,bin1_min+j] += 1 #k+1

                            elif tmp_i is 1 or tmp_i is 2 or tmp_i is 3:
                                ###################################################
                                #  Sutherland-Hodgman polygon clipping algorithm  #
                                ###################################################
                                #
                                #  ...adjusted to utilise the peculiarities of our problem
                                #

                                A.i = A0
                                A.j = A1
                                B.i = B0
                                B.j = B1
                                C.i = C0
                                C.j = C1
                                D.i = D0
                                D.j = D1

                                list1.data[0] = A
                                list1.data[1] = B
                                list1.data[2] = C
                                list1.data[3] = D
                                list1.size = 4
                                list2.size = 0

                                S = list1.data[list1.size - 1]  # last element
                                for tmp_i in range(list1.size):
                                    E = list1.data[tmp_i]
                                    if E.i > i:  # is_inside(E, clipEdge):   -- i is the x coord of current bin
                                        if S.i <= i:  # not is_inside(S, clipEdge):
                                            list2.data[list2.size] = ComputeIntersection0(S, E, i)
                                            list2.size += 1
                                        list2.data[list2.size] = E
                                        list2.size += 1
                                    elif S.i > i:  # is_inside(S, clipEdge):
                                        list2.data[list2.size] = ComputeIntersection0(S,E,i)
                                        list2.size += 1
                                    S = E;
                                #y=b+1
                                list1.size = 0
                                S = list2.data[list2.size - 1]
                                for tmp_i in range(list2.size):
                                    E = list2.data[tmp_i]
                                    if E.j < j + 1:  # is_inside(E, clipEdge):   -- j is the y coord of current bin
                                        if S.j >= j + 1:  # not is_inside(S, clipEdge):
                                            list1.data[list1.size] = ComputeIntersection1(S, E, j + 1)
                                            list1.size += 1
                                        list1.data[list1.size] = E
                                        list1.size += 1
                                    elif S.j < j + 1:  # is_inside(S, clipEdge):
                                        list1.data[list1.size] = ComputeIntersection1(S, E, j + 1)
                                        list1.size += 1
                                    S = E
                                #x=a+1
                                list2.size = 0
                                S = list1.data[list1.size-1]
                                for tmp_i in range(list1.size):
                                    E = list1.data[tmp_i]
                                    if E.i < i + 1:  # is_inside(E, clipEdge):
                                        if S.i >= i + 1:  # not is_inside(S, clipEdge):
                                            list2.data[list2.size] = ComputeIntersection0(S, E, i + 1)
                                            list2.size += 1
                                        list2.data[list2.size] = E
                                        list2.size += 1
                                    elif S.i < i + 1:  # is_inside(S, clipEdge):
                                        list2.data[list2.size] = ComputeIntersection0(S, E, i + 1)
                                        list2.size += 1
                                    S = E
                                #y=b
                                list1.size = 0
                                S = list2.data[list2.size-1]
                                for tmp_i in range(list2.size):
                                    E = list2.data[tmp_i]
                                    if E.j > j:  # is_inside(E, clipEdge):
                                        if S.j <= j:  # not is_inside(S, clipEdge):
                                            list1.data[list1.size] = ComputeIntersection1(S, E, j)
                                            list1.size += 1
                                        list1.data[list1.size] = E
                                        list1.size += 1
                                    elif S.j > j:  # is_inside(S, clipEdge):
                                        list1.data[list1.size] = ComputeIntersection1(S, E, j)
                                        list1.size += 1
                                    S = E

                                partialArea = area_n(list1)

                                k = outMax[bin0_min + i, bin1_min + j]
                                index = (i + bin0_min) * all_bins1 + j + bin1_min
                                if index > all_bins:
                                    printf("3.1 index = %d > %d!! \n", index, all_bins)
                                    fflush(stdout)
                                if indptr[index] > indptr[all_bins]:
                                    printf("3.1 indptr = %d > %d!! \n", indptr[index], indptr[all_bins])
                                    fflush(stdout)
                                indices[indptr[index] + k] = idx
                                data[indptr[index] + k] = partialArea * oneOverPixelArea
                                outMax[bin0_min + i, bin1_min + j] += 1  # k+1

        self.data = data
        self.indices = indices
        self.outMax = outMax

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def integrate(self, weights,
                  dummy=None,
                  delta_dummy=None,
                  dark=None,
                  flat=None,
                  solidAngle=None,
                  polarization=None,
                  double normalization_factor=1.0
                  ):
        """
        Actually perform the integration which in this case looks more like a matrix-vector product

        @param weights: input image
        @type weights: ndarray
        @param dummy: value for dead pixels (optional)
        @type dummy: float
        @param delta_dummy: precision for dead-pixel value in dynamic masking
        @type delta_dummy: float
        @param dark: array with the dark-current value to be subtracted (if any)
        @type dark: ndarray
        @param flat: array with the dark-current value to be divided by (if any)
        @type flat: ndarray
        @param solidAngle: array with the solid angle of each pixel to be divided by (if any)
        @type solidAngle: ndarray
        @param polarization: array with the polarization correction values to be divided by (if any)
        @type polarization: ndarray
        @param normalization_factor: divide the valid result by this value

        @return : positions, pattern, weighted_histogram and unweighted_histogram
        @rtype: 4-tuple of ndarrays

        """
        cdef:
            numpy.int32_t i=0, j=0, idx=0, bins=self.bins[0]*self.bins[1], size=self.size
            float sum_data=0.0, sum_count=0.0, epsilon=1e-10
            float data=0, coef=0, cdummy=0, cddummy=0
            bint do_dummy=False, do_dark=False, do_flat=False, do_polarization=False, do_solidAngle=False
            numpy.ndarray[numpy.float64_t, ndim = 1] outData = numpy.zeros(bins, dtype=numpy.float64)
            numpy.ndarray[numpy.float64_t, ndim = 1] outCount = numpy.zeros(bins, dtype=numpy.float64)
            numpy.ndarray[numpy.float32_t, ndim = 1] outMerge = numpy.zeros(bins, dtype=numpy.float32)
            float[:] ccoef = self.data, cdata, tdata, cflat, cdark, csolidAngle, cpolarization

            numpy.int32_t[:] indices = self.indices, indptr = self.indptr
        assert size == weights.size

        if dummy is not None:
            do_dummy = True
            cdummy =  <float>float(dummy)
            if delta_dummy is None:
                cddummy = <float>0.0
            else:
                cddummy = <float>float(delta_dummy)
        else:
            do_dummy = False
            cdummy =  <float>float(self.empty)

        if flat is not None:
            do_flat = True
            assert flat.size == size
            cflat = numpy.ascontiguousarray(flat.ravel(), dtype=numpy.float32)
        if dark is not None:
            do_dark = True
            assert dark.size == size
            cdark = numpy.ascontiguousarray(dark.ravel(), dtype=numpy.float32)
        if solidAngle is not None:
            do_solidAngle = True
            assert solidAngle.size == size
            csolidAngle = numpy.ascontiguousarray(solidAngle.ravel(), dtype=numpy.float32)
        if polarization is not None:
            do_polarization = True
            assert polarization.size == size
            cpolarization = numpy.ascontiguousarray(polarization.ravel(), dtype=numpy.float32)

        if (do_dark + do_flat + do_polarization + do_solidAngle):
            tdata = numpy.ascontiguousarray(weights.ravel(), dtype=numpy.float32)
            cdata = numpy.zeros(size, dtype=numpy.float32)
            if do_dummy:
                for i in prange(size, nogil=True, schedule="static"):
                    data = tdata[i]
                    if ((cddummy != 0) and (fabs(data - cdummy) > cddummy)) or ((cddummy == 0) and (data != cdummy)):
                        #Nota: -= and /= operatore are seen as reduction in cython parallel.
                        if do_dark:
                            data = data - cdark[i]
                        if do_flat:
                            data = data / cflat[i]
                        if do_polarization:
                            data = data / cpolarization[i]
                        if do_solidAngle:
                            data = data / csolidAngle[i]
                        cdata[i] += data
                    else:  # set all dummy_like values to cdummy. simplifies further processing
                        cdata[i] += cdummy
            else:
                for i in prange(size, nogil=True, schedule="static"):
                    data = tdata[i]
                    if do_dark:
                        data = data - cdark[i]
                    if do_flat:
                        data = data / cflat[i]
                    if do_polarization:
                        data = data / cpolarization[i]
                    if do_solidAngle:
                        data = data / csolidAngle[i]
                    cdata[i] += data
        else:
            if do_dummy:
                tdata = numpy.ascontiguousarray(weights.ravel(), dtype=numpy.float32)
                cdata = numpy.zeros(size, dtype=numpy.float32)
                for i in prange(size, nogil=True, schedule="static"):
                    data = tdata[i]
                    if ((cddummy != 0) and (fabs(data - cdummy) > cddummy)) or ((cddummy == 0) and (data != cdummy)):
                        cdata[i] += data
                    else:
                        cdata[i] += cdummy
            else:
                cdata = numpy.ascontiguousarray(weights.ravel(), dtype=numpy.float32)

        for i in prange(bins, nogil=True, schedule="guided"):
            sum_data = 0.0
            sum_count = 0.0
            for j in range(indptr[i], indptr[i + 1]):
                idx = indices[j]
                coef = ccoef[j]
                if coef == 0.0:
                    continue
                data = cdata[idx]
                if do_dummy and (data == cdummy):
                    continue
                sum_data = sum_data + coef * data
                sum_count = sum_count + coef
            outData[i] += sum_data
            outCount[i] += sum_count
            if sum_count > epsilon:
                outMerge[i] += <float> (sum_data / sum_count / normalization_factor)
            else:
                outMerge[i] += cdummy

        return  outMerge, outData, outCount
