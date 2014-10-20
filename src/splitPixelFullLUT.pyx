#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal Integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
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
__doc__ = """
Full pixel Splitting implemented using Sparse-matrix Dense-Vector multiplication,
Sparse matrix represented using the CompressedSparseROw.
"""
__author__ = "Jerome Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "20/10/2014"
__status__ = "stable"
__license__ = "GPLv3+"
import cython
import os
import sys
from cython.parallel import prange
from libc.string cimport memset
import numpy
cimport numpy
from libc.math cimport fabs, M_PI, floor
from libc.stdio cimport printf

cdef float pi = <float> M_PI 
cdef float onef = <float> 1.0
try:
    from fastcrc import crc32
except:
    from zlib import crc32
EPS32 = (1.0 + numpy.finfo(numpy.float32).eps)



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
cdef inline float getBinNr(float x0, float pos0_min, float delta) nogil:
    """
    calculate the bin number for any point
    param x0: current position
    param pos0_min: position minimum
    param delta: bin width
    """
    return (x0 - pos0_min) / delta

cdef float integrate(float A0, float B0, Function AB) nogil:
    """
    integrates the line defined by AB, from A0 to B0
    param A0: first limit
    param B0: second limit
    param AB: struct with the slope and point of intersection of the line
    """    
    if A0 == B0:
        return 0.0
    else:
        return AB.slope * (B0 * B0 - A0 * A0) * 0.5 + AB.intersect*(B0 - A0)
    
    
class HistoLUT1dFullSplit(object):
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
                 unit="undefined"):  
        """
        @param pos: 3D or 4D array with the coordinates of each pixel point
        @param bins: number of output bins, 100 by default
        @param pos0Range: minimum and maximum  of the 2th range
        @param pos1Range: minimum and maximum  of the chi range
        @param mask: array (of int8) with masked pixels with 1 (0=not masked)
        @param allow_pos0_neg: enforce the q<0 is usually not possible  
        @param unit: can be 2th_deg or r_nm^-1 ...
        """

        if pos.ndim>3: #  create a view
            pos = pos.reshape((-1,4,2))
        assert pos.shape[1] == 4
        assert pos.shape[2] == 2
        assert pos.ndim == 3
        self.pos = pos
        self.size = pos.shape[0]
        self.bins = bins
        self.lut_size = 0
        self.allow_pos0_neg = allow_pos0_neg
        if mask is not None:
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
        self.outPos = numpy.linspace(self.pos0_min + 0.5 * self.delta, self.pos0_maxin - 0.5 * self.delta, self.bins)
        self.lut_checksum = crc32(self.data)
        self.unit = unit
        self.lut = (self.data, self.indices, self.indptr)
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
        
        
        
        #if self.bad_pixel:
            #range1 = self.bad_pixel
            #range2 = self.bad_pixel + 1
        #else:
            #range1 = 0
            #range2 = size
            
        #print "FLOAT 32"
        #print "++++++++"
        #print "Space bounds: %e  -  %e" % (pos0_min, pos0_max)
        
        with nogil:
            for idx in range(size):
            #for idx in range(range1,range2):
                
                #with gil:
                    #print "Pixel %d" % idx
                    #print "==========="
                    #print "==========="

                if (check_mask) and (cmask[idx]):
                    continue

                A0 = getBinNr(< float > cpos[idx, 0, 0], pos0_min, delta)
                A1 = < float > cpos[idx, 0, 1]
                B0 = getBinNr(< float > cpos[idx, 1, 0], pos0_min, delta)
                B1 = < float > cpos[idx, 1, 1]
                C0 = getBinNr(< float > cpos[idx, 2, 0], pos0_min, delta)
                C1 = < float > cpos[idx, 2, 1]
                D0 = getBinNr(< float > cpos[idx, 3, 0], pos0_min, delta)
                D1 = < float > cpos[idx, 3, 1]
                
                #with gil:
                    #print "Pixel in 2-th"
                    #print "============="
                    #print "A: %e --> %e  %e" % (< double > cpos[idx, 0, 0], A0, A1)
                    #print "B: %e --> %e  %e" % (< double > cpos[idx, 1, 0], B0, B1)
                    #print "C: %e --> %e  %e" % (< double > cpos[idx, 2, 0], C0, C1)
                    #print "D: %e --> %e  %e" % (< double > cpos[idx, 3, 0], D0, D1)
                    #print " "

                

                min0 = min(A0, B0, C0, D0)
                max0 = max(A0, B0, C0, D0)
                
                #with gil:
                    #print "Min 2-th: %e" % min0
                    #print "Max 2-th: %e" % max0
                    #print " "
                
                if (max0<0) or (min0 >=bins):
                    continue
                if check_pos1:
                    if (max(A1, B1, C1, D1) < pos1_min) or (min(A1, B1, C1, D1) > pos1_maxin):
                        continue

                bin0_min = < int > floor(min0)
                bin0_max = < int > floor(max0)
                
                #with gil:
                    #print "Bin span: %d - %d" % (bin0_min, bin0_max)
                    #print " "

                for bin in range(bin0_min, bin0_max+1):
                    outMax[bin] += 1
    
        indptr[1:] = outMax.cumsum()
        self.indptr = indptr
        
        cdef numpy.ndarray[numpy.int32_t, ndim = 1] indices = numpy.zeros(indptr[bins], dtype=numpy.int32)
        cdef numpy.ndarray[numpy.float32_t, ndim = 1] data = numpy.zeros(indptr[bins], dtype=numpy.float32)
        
        #just recycle the outMax array
        outMax[:] = 0
        
        with nogil:
            for idx in range(size):

                if (check_mask) and (cmask[idx]):
                    continue

                A0 = getBinNr(< float > cpos[idx, 0, 0], pos0_min, delta)
                A1 = < float > cpos[idx, 0, 1]
                B0 = getBinNr(< float > cpos[idx, 1, 0], pos0_min, delta)
                B1 = < float > cpos[idx, 1, 1]
                C0 = getBinNr(< float > cpos[idx, 2, 0], pos0_min, delta)
                C1 = < float > cpos[idx, 2, 1]
                D0 = getBinNr(< float > cpos[idx, 3, 0], pos0_min, delta)
                D1 = < float > cpos[idx, 3, 1]
                

                min0 = min(A0, B0, C0, D0)
                max0 = max(A0, B0, C0, D0)
                
                if (max0<0) or (min0 >=bins):
                    continue
                if check_pos1:
                    if (max(A1, B1, C1, D1) < pos1_min) or (min(A1, B1, C1, D1) > pos1_maxin):
                        continue

                bin0_min = < int > floor(min0)
                bin0_max = < int > floor(max0)
                
                if bin0_min == bin0_max:
                    #All pixel is within a single bin
                    k = outMax[bin0_min]
                    indices[indptr[bin0_min]+k] = idx
                    data[indptr[bin0_min]+k] = 1.0
                    outMax[bin0_min] += 1 #k+1
                else:  #else we have pixel spliting.
                    # offseting the min bin of the pixel to be zero to avoid percision problems
                    A0 -= bin0_min
                    B0 -= bin0_min
                    C0 -= bin0_min
                    D0 -= bin0_min
                    
                    AB.slope=(B1-A1)/(B0-A0)
                    AB.intersect= A1 - AB.slope*A0
                    BC.slope=(C1-B1)/(C0-B0)
                    BC.intersect= B1 - BC.slope*B0
                    CD.slope=(D1-C1)/(D0-C0)
                    CD.intersect= C1 - CD.slope*C0
                    DA.slope=(A1-D1)/(A0-D0)
                    DA.intersect= D1 - DA.slope*D0
                    
                    areaPixel = area4(A0, A1, B0, B1, C0, C1, D0, D1)
                    
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
                       
                        partialArea  = integrate(A_lim, B_lim, AB)
                        partialArea += integrate(B_lim, C_lim, BC)
                        partialArea += integrate(C_lim, D_lim, CD)
                        partialArea += integrate(D_lim, A_lim, DA)
                        
                        tmp = fabs(partialArea) * oneOverPixelArea
                        
                        k = outMax[bin]
                        indices[indptr[bin]+k] = idx
                        data[indptr[bin]+k] = tmp
                        outMax[bin] += 1 #k+1
                        
        self.data = data
        self.indices = indices
        self.outMax = outMax
                           
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def integrate(self, weights, dummy=None, delta_dummy=None, dark=None, flat=None, solidAngle=None, polarization=None):
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
                        cdata[i]+=data
                    else: #set all dummy_like values to cdummy. simplifies further processing
                        cdata[i]+=cdummy
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
                    cdata[i]+=data
        else:
            if do_dummy:
                tdata = numpy.ascontiguousarray(weights.ravel(), dtype=numpy.float32)
                cdata = numpy.zeros(size,dtype=numpy.float32)
                for i in prange(size, nogil=True, schedule="static"):
                    data = tdata[i]
                    if ((cddummy!=0) and (fabs(data-cdummy) > cddummy)) or ((cddummy==0) and (data!=cdummy)):
                        cdata[i]+=data
                    else:
                        cdata[i]+=cdummy
            else:
                cdata = numpy.ascontiguousarray(weights.ravel(), dtype=numpy.float32)
        
        for i in prange(bins, nogil=True, schedule="guided"):
            sum_data = 0.0
            sum_count = 0.0
            for j in range(indptr[i],indptr[i+1]):
                idx = indices[j]
                coef = ccoef[j]
                if coef == 0.0:
                    continue
                data = cdata[idx]
                if do_dummy and data==cdummy:
                    continue
                sum_data = sum_data + coef * data
                sum_count = sum_count + coef
            outData[i] += sum_data
            outCount[i] += sum_count
            if sum_count > epsilon:
                outMerge[i] += sum_data / sum_count
            else:
                outMerge[i] += cdummy
        return self.outPos, outMerge, outData, outCount



