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
import cython
import os
from cython.parallel import prange
from libc.string cimport memset
import numpy
cimport numpy
from libc.math cimport fabs
cdef struct lut_point:
    numpy.uint32_t idx
    numpy.float32_t coef
try:
    from fastcrc import crc32
except:
    from zlib import crc32

@cython.cdivision(True)
cdef float getBinNr( float x0, float pos0_min, float delta) nogil:
    """
    calculate the bin number for any point
    param x0: current position
    param pos0_min: position minimum
    param delta: bin width
    """
    return (x0 - pos0_min) / delta

class HistoBBox1d(object):
    @cython.boundscheck(False)
    def __init__(self,
                 pos0,
                 delta_pos0,
                 pos1=None,
                 delta_pos1=None,
                 int bins=100,
                 pos0Range=None,
                 pos1Range=None,
                 mask=None,
                 mask_checksum=None,
                 allow_pos0_neg=False):

        cdef int i, size
        self.size = pos0.size
        assert delta_pos0.size == self.size
        self.bins = bins
        self.lut_size = 0
        self.cpos0 = numpy.ascontiguousarray(pos0.ravel(), dtype=numpy.float32)
        self.dpos0 = numpy.ascontiguousarray(delta_pos0.ravel(), dtype=numpy.float32)
        self.cpos0_sup = self.cpos0 + self.dpos0
        self.cpos0_inf = self.cpos0 - self.dpos0
        self.pos0Range = pos0Range
        self.pos1Range = pos1Range
        if pos0Range is not None and len(pos0Range) > 1:
            self.pos0_min = min(pos0Range)
            pos0_maxin = max(pos0Range)
        else:
            self.pos0_min = (self.cpos0_inf).min()
            pos0_maxin = (self.cpos0_sup).max()
        if (not allow_pos0_neg) and self.pos0_min < 0:
            self.pos0_min = 0
        self.pos0_max = pos0_maxin * (1.0 + numpy.finfo(numpy.float32).eps)

        if pos1Range is not None and len(pos1Range) > 1:
            assert pos1.size == self.size
            assert delta_pos1.size == self.size
            self.check_pos1 = True
            self.cpos1_min = numpy.ascontiguousarray((pos1-delta_pos1).ravel(), dtype=numpy.float32)
            self.cpos1_max = numpy.ascontiguousarray((pos1+delta_pos1).ravel(), dtype=numpy.float32)
            self.pos1_min = min(pos1Range)
            pos1_maxin = max(pos1Range)
            self.pos1_max = pos1_maxin * (1 + numpy.finfo(numpy.float32).eps)
        else:
            self.check_pos1 = False
            self.cpos1_min = None
            self.pos1_max = None

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
        self.delta = (self.pos0_max - self.pos0_min) / bins
        self.lut_max_idx = self.calc_lut()
        self.outPos = numpy.linspace(self.pos0_min+0.5*self.delta, pos0_maxin-0.5*self.delta, self.bins)
        self.outPos_degrees = numpy.degrees(self.outPos)
        self.lut_checksum = crc32(self.lut)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def calc_lut(self):
        'calculate the max number of elements in the LUT and populate it'
        cdef float delta=self.delta, pos0_min=self.pos0_min, pos1_min, pos1_max, min0, max0, fbin0_min, fbin0_max, deltaL, deltaR, deltaA
        cdef int bin0_min, bin0_max, bins = self.bins, lut_size, i, size
        cdef numpy.uint32_t k,idx #same as numpy.uint32
        cdef bint check_mask, check_pos1
        cdef numpy.ndarray[numpy.int_t, ndim = 1] outMax = numpy.zeros(self.bins, dtype=numpy.int)
        cdef float[:] cpos0_sup = self.cpos0_sup
        cdef float[:] cpos0_inf = self.cpos0_inf
        cdef float[:] cpos1_min, cpos1_max
        cdef numpy.ndarray[numpy.uint32_t, ndim = 1] max_idx = numpy.zeros(bins, dtype=numpy.uint32)
        cdef numpy.ndarray[lut_point, ndim = 2] lut
        cdef numpy.int8_t[:] cmask
        size = self.size
        if self.check_mask:
            cmask = self.cmask
            check_mask = True
        else:
            check_mask = False

        if self.check_pos1:
            check_pos1 = True
            cpos1_min = self.cpos1_min
            cpos1_max = self.cpos1_max
            pos1_max = self.pos1_max
            pos1_min = self.pos1_min
        else:
            check_pos1 = False
#NOGIL
        with nogil:
            for idx in range(size):
                if (check_mask) and (cmask[idx]):
                    continue

                min0 = cpos0_inf[idx]
                max0 = cpos0_sup[idx]

                if check_pos1 and ((cpos1_max[idx] < pos1_min) or (cpos1_min[idx] > pos1_max)):
                    continue

                fbin0_min = getBinNr(min0, pos0_min, delta)
                fbin0_max = getBinNr(max0, pos0_min, delta)
                bin0_min = < int > fbin0_min
                bin0_max = < int > fbin0_max

                if (bin0_max < 0) or (bin0_min >= bins):
                    continue
                if bin0_max >= bins :
                    bin0_max = bins - 1
                if  bin0_min < 0:
                    bin0_min = 0

                if bin0_min == bin0_max:
                    #All pixel is within a single bin
                    outMax[bin0_min] += 1

                else: #we have pixel spliting.
                    for i in range(bin0_min, bin0_max + 1):
                        outMax[i] += 1

        lut_size = outMax.max()
        self.lut_size = lut_size

        lut_nbytes = bins*lut_size*sizeof(lut_point)
        if (os.name == "posix") and ("SC_PAGE_SIZE" in os.sysconf_names) and ("SC_PHYS_PAGES" in os.sysconf_names):
            memsize =  os.sysconf("SC_PAGE_SIZE")*os.sysconf("SC_PHYS_PAGES")
            if memsize <  lut_nbytes:
                raise MemoryError("Lookup-table (%i, %i) is %.3fGB whereas the memory of the system is only %s"%(bins,lut_size,lut_nbytes,memsize))
        #else hope we have enough memory
        lut = numpy.recarray(shape=(bins, lut_size),dtype=[("idx",numpy.uint32),("coef",numpy.float32)])
        memset(&lut[0,0], 0, bins*lut_size*sizeof(lut_point))
        #NOGIL
        with nogil:
            for idx in range(size):
                if (check_mask) and (cmask[idx]):
                    continue

                min0 = cpos0_inf[idx]
                max0 = cpos0_sup[idx]

                if check_pos1 and ((cpos1_max[idx] < pos1_min) or (cpos1_min[idx] > pos1_max)):
                        continue

                fbin0_min = getBinNr(min0, pos0_min, delta)
                fbin0_max = getBinNr(max0, pos0_min, delta)
                bin0_min = < int > fbin0_min
                bin0_max = < int > fbin0_max

                if (bin0_max < 0) or (bin0_min >= bins):
                    continue
                if bin0_max >= bins :
                    bin0_max = bins - 1
                if  bin0_min < 0:
                    bin0_min = 0

                if bin0_min == bin0_max:
                    #All pixel is within a single bin
                    k = max_idx[bin0_min]
                    lut[bin0_min, k].idx = idx
                    lut[bin0_min, k].coef = 1.0
                    max_idx[bin0_min] = k + 1
                else: #we have pixel splitting.
                    deltaA = 1.0 / (fbin0_max - fbin0_min)

                    deltaL = (bin0_min + 1) - fbin0_min
                    deltaR = fbin0_max - (bin0_max)

                    k = max_idx[bin0_min]
                    lut[bin0_min, k].idx = idx
                    lut[bin0_min, k].coef = (deltaA * deltaL)
                    max_idx[bin0_min] = k + 1

                    k = max_idx[bin0_max]
                    lut[bin0_max, k].idx = idx
                    lut[bin0_max, k].coef = (deltaA * deltaR)
                    max_idx[bin0_max] = k + 1

                    if bin0_min + 1 < bin0_max:
                        for i in range(bin0_min + 1, bin0_max):
                            k = max_idx[i]
                            lut[i, k].idx = idx
                            lut[i, k].coef = (deltaA)
                            max_idx[i] = k + 1
        self.lut = lut
        return max_idx


    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def integrate(self, weights, dummy=None, delta_dummy=None, dark=None, flat=None, solidAngle=None, polarization=None):
        cdef ssize_t i=0, j=0, idx=0, bins=self.bins, lut_size=self.lut_size, size=self.size
        cdef double sum_data=0, sum_count=0, epsilon=1e-10
        cdef float data=0, coef=0, cdummy=0, cddummy=0
        cdef bint do_dummy=False, do_dark=False, do_flat=False, do_polarization=False, do_solidAngle=False
        cdef numpy.ndarray[numpy.float64_t, ndim = 1] outData = numpy.zeros(self.bins, dtype=numpy.float64)
        cdef numpy.ndarray[numpy.float64_t, ndim = 1] outCount = numpy.zeros(self.bins, dtype=numpy.float64)
        cdef numpy.ndarray[numpy.float32_t, ndim = 1] outMerge = numpy.zeros(self.bins, dtype=numpy.float32)
        cdef lut_point[:,:] lut = self.lut
        cdef float[:] cdata, tdata, cflat, cdark, csolidAngle, cpolarization

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
            for j in range(lut_size):
                idx = lut[i, j].idx
                coef = lut[i, j].coef
                if idx <= 0 and coef <= 0.0:
                    break
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
        return  self.outPos, outMerge, outData, outCount


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
    do_dummy = 0
    do_dark = 0
    do_flat = 0

    cdata = numpy.ascontiguousarray(weights.ravel(), dtype=numpy.float32)
    cpos0 = numpy.ascontiguousarray(pos0.ravel(), dtype=numpy.float32)
    dpos0 = numpy.ascontiguousarray(delta_pos0.ravel(), dtype=numpy.float32)


    outData = numpy.zeros(bins, dtype=numpy.float64)
    outCount = numpy.zeros(bins, dtype=numpy.float64)
    outMax = numpy.zeros(bins, dtype=numpy.int64)
    outMerge = numpy.zeros(bins, dtype=numpy.float32)
#    outPos = numpy.zeros(bins, dtype=numpy.float32)

    if  mask is not None:
        assert mask.size == size
        check_mask = 1
        cmask = numpy.ascontiguousarray(mask.ravel(), dtype=numpy.int8)

    if (dummy is not None) and delta_dummy is not None:
        do_dummy = 1
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

#    for i in range(bins):
#                outPos[i] = pos0_min + (0.5 + i) * delta
    outPos = numpy.linspace(pos0_min+0.5*delta,pos0_max-0.5*delta, bins)
    for idx in range(size):
            if (check_mask) and (cmask[idx]):
                continue

            data = cdata[idx]
            if do_dummy and (fabs(data - cdummy) <= ddummy):
                continue

            min0 = cpos0_lower[idx]
            max0 = cpos0_upper[idx]

            if check_pos1 and (((cpos1[idx] + dpos1[idx]) < pos1_min) or ((cpos1[idx] - dpos1[idx]) > pos1_max)):
                    continue

            fbin0_min = getBinNr(min0, pos0_min, delta)
            fbin0_max = getBinNr(max0, pos0_min, delta)
            bin0_min = <int> (fbin0_min)
            bin0_max = <int> (fbin0_max)

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




