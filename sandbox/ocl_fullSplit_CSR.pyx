#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2014-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import cython
import os
import sys
from cpython.ref cimport PyObject, Py_XDECREF
from cython.parallel import prange
from libc.string cimport memset, memcpy
from cython cimport view
import numpy
cimport numpy
from libc.math cimport fabs, M_PI
cdef float pi = <float> M_PI
cdef struct lut_point:
    numpy.int32_t idx
    numpy.float32_t coef
dtype_lut = numpy.dtype([("idx", numpy.int32), ("coef", numpy.float32)])
from ..utils import crc32
cdef double EPS32 = (1.0 + numpy.finfo(numpy.float32).eps)
cdef bint NEED_DECREF = sys.version_info < (2, 7) and numpy.version.version < "1.5"


class OCLFullSplit1d(object):
    """
    1D histogramming with full pixel splitting
    based on a Look-up table in CSR format

    The initialization of the class can take quite a while (operation are not parallelized)
    but each integrate is parallelized and quite efficient.
    """

    @cython.boundscheck(False)
    def __init__(self,
                 pos,
                 int bins=100,
                 pos0Range=None,
                 pos1Range=None,
                 mask=None,
                 mask_checksum=None,
                 allow_pos0_neg=False,
                 unit="undefined"):
        """
        @param pos: 4D array with the coodrinates of all pixels: tth or q_vect or r...
        @param bins: number of output bins, 100 by default
        @param pos0Range: minimum and maximum  of the 2th range
        @param pos1Range: minimum and maximum  of the chi range
        @param mask: array (of int8) with masked pixels with 1 (0=not masked)
        @param allow_pos0_neg: enforce the q<0 is usually not possible,      NOT USED
        @param unit: can be 2th_deg or r_nm^-1 ...
        """

        self.bins = bins
        self.lut_size = 0
        self.allow_pos0_neg = allow_pos0_neg

        if len(pos.shape) == 3:
            assert pos.shape[1] == 4
            assert pos.shape[2] == 2
        elif len(pos.shape) == 4:
            assert pos.shape[2] == 4
            assert pos.shape[3] == 2
        else:
            raise ValueError("Pos array dimentions are wrong")
        self.size = pos.size / 8

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

        self.pos = numpy.ascontiguousarray(pos.ravel(), dtype=numpy.float32)
        self.pos0Range = pos0Range
        self.pos1Range = pos1Range

        self.calc_boundaries(pos0Range)
        if pos1Range is not None and len(pos1Range) > 1:
            assert pos1.size == self.size
            assert delta_pos1.size == self.size
            self.check_pos1 = True
            self.cpos1_min = numpy.ascontiguousarray((pos1 - delta_pos1).ravel(), dtype=numpy.float32)
            self.cpos1_max = numpy.ascontiguousarray((pos1 + delta_pos1).ravel(), dtype=numpy.float32)
            self.pos1_min = min(pos1Range)
            pos1_maxin = max(pos1Range)
            self.pos1_max = pos1_maxin * EPS32
        else:
            self.check_pos1 = False
            self.cpos1_min = None
            self.pos1_max = None

        self.delta = (self.pos0_max - self.pos0_min) / bins
        self._lut = None
        self.lut_max_idx = None
        self.calc_lut()
        self.outPos = numpy.linspace(self.pos0_min + 0.5 * self.delta, self.pos0_maxin - 0.5 * self.delta, self.bins)
        self.lut_checksum = None
        self.unit = unit
        self.lut_nbytes = self._lut.nbytes

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def calc_boundaries(self, pos0Range):
        """
        Called by constructor to calculate the boundaries and the bin position
        """
        cdef int size = self.cpos0.size
        cdef bint check_mask = self.check_mask
        cdef numpy.int8_t[:] cmask
        cdef float[:] cpos0, dpos0, cpos0_sup, cpos0_inf,
        cdef float upper, lower, pos0_max, pos0_min, c, d
        cdef bint allow_pos0_neg = self.allow_pos0_neg

        cpos0_sup = self.cpos0_sup
        cpos0_inf = self.cpos0_inf
        cpos0 = self.cpos0
        dpos0 = self.dpos0
        pos0_min = cpos0[0]
        pos0_max = cpos0[0]

        if check_mask:
            cmask = self.cmask
        with nogil:
            for idx in range(size):
                c = cpos0[idx]
                d = dpos0[idx]
                lower = c - d
                upper = c + d
                cpos0_sup[idx] = upper
                cpos0_inf[idx] = lower
                if not allow_pos0_neg and lower < 0:
                    lower = 0
                if not (check_mask and cmask[idx]):
                    if upper > pos0_max:
                        pos0_max = upper
                    if lower < pos0_min:
                        pos0_min = lower

        if pos0Range is not None and len(pos0Range) > 1:
            self.pos0_min = min(pos0Range)
            self.pos0_maxin = max(pos0Range)
        else:
            self.pos0_min = pos0_min
            self.pos0_maxin = pos0_max
        if (not allow_pos0_neg) and self.pos0_min < 0:
            self.pos0_min = 0
        self.pos0_max = self.pos0_maxin * EPS32

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def calc_lut(self):
        """
        Calculate the max number of elements in the LUT and populate it
        """
        cdef float delta = self.delta, pos0_min = self.pos0_min, pos1_min, pos1_max, min0, max0, fbin0_min, fbin0_max, deltaL, deltaR, deltaA
        cdef numpy.int32_t k, idx, bin0_min, bin0_max, bins = self.bins, lut_size, i, size
        cdef bint check_mask, check_pos1
        cdef numpy.ndarray[numpy.int32_t, ndim=1] outMax = numpy.zeros(bins, dtype=numpy.int32)
        cdef float[:] cpos0_sup = self.cpos0_sup
        cdef float[:] cpos0_inf = self.cpos0_inf
        cdef float[:] cpos1_min, cpos1_max
        cdef lut_point[:, :] lut

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
                bin0_min = < numpy.int32_t > fbin0_min
                bin0_max = < numpy.int32_t > fbin0_max

                if (bin0_max < 0) or (bin0_min >= bins):
                    continue
                if bin0_max >= bins:
                    bin0_max = bins - 1
                if bin0_min < 0:
                    bin0_min = 0

                if bin0_min == bin0_max:
                    # All pixel is within a single bin
                    outMax[bin0_min] += 1

                else:  # we have pixel spliting.
                    for i in range(bin0_min, bin0_max + 1):
                        outMax[i] += 1

        lut_size = outMax.max()
        # just recycle the outMax array
        # outMax = numpy.zeros((bins0,bins1), dtype=numpy.int32)
        memset(&outMax[0], 0, bins * sizeof(numpy.int32_t))

        self.lut_size = lut_size

        lut_nbytes = bins * lut_size * sizeof(lut_point)
        if (os.name == "posix") and ("SC_PAGE_SIZE" in os.sysconf_names) and ("SC_PHYS_PAGES" in os.sysconf_names):
            memsize = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            if memsize < lut_nbytes:
                raise MemoryError("Lookup-table (%i, %i) is %.3fGB whereas the memory of the system is only %s" % (bins, lut_size, lut_nbytes, memsize))
        # else hope we have enough memory
        lut = view.array(shape=(bins, lut_size), itemsize=sizeof(lut_point), format="if")
        # lut = numpy.zeros(shape=(bins, lut_size),dtype=dtype_lut)
        # lut = < lut_point *>malloc(lut_nbytes)
        memset(&lut[0, 0], 0, lut_nbytes)

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
                bin0_min = < numpy.int32_t > fbin0_min
                bin0_max = < numpy.int32_t > fbin0_max

                if (bin0_max < 0) or (bin0_min >= bins):
                    continue
                if bin0_max >= bins:
                    bin0_max = bins - 1
                if bin0_min < 0:
                    bin0_min = 0

                if bin0_min == bin0_max:
                    # All pixel is within a single bin
                    k = outMax[bin0_min]
                    lut[bin0_min, k].idx = idx
                    lut[bin0_min, k].coef = 1.0
                    outMax[bin0_min] += 1
                else:  # we have pixel splitting.
                    deltaA = 1.0 / (fbin0_max - fbin0_min)

                    deltaL = (bin0_min + 1) - fbin0_min
                    deltaR = fbin0_max - (bin0_max)

                    k = outMax[bin0_min]
                    lut[bin0_min, k].idx = idx
                    lut[bin0_min, k].coef = (deltaA * deltaL)
                    outMax[bin0_min] += 1

                    k = outMax[bin0_max]
                    lut[bin0_max, k].idx = idx
                    lut[bin0_max, k].coef = (deltaA * deltaR)
                    outMax[bin0_max] += 1

                    if bin0_min + 1 < bin0_max:
                        for i in range(bin0_min + 1, bin0_max):
                            k = outMax[i]
                            lut[i, k].idx = idx
                            lut[i, k].coef = (deltaA)
                            outMax[i] += 1

        self.lut_max_idx = outMax
        self._lut = lut

    def get_lut(self):
        """Getter for the LUT as actual numpy array"""
        cdef int rc_before, rc_after
        rc_before = sys.getrefcount(self._lut)
        cdef lut_point[:, :] lut = self._lut
        rc_after = sys.getrefcount(self._lut)
        cdef bint need_decref = NEED_DECREF and ((rc_after - rc_before) >= 2)
        cdef numpy.ndarray[numpy.float64_t, ndim=2] tmp_ary = numpy.empty(shape=self._lut.shape, dtype=numpy.float64)
        memcpy(&tmp_ary[0, 0], &lut[0, 0], self._lut.nbytes)
        self.lut_checksum = crc32(tmp_ary)

        # Ugly against bug#89
        if need_decref and (sys.getrefcount(self._lut) >= rc_before+2):
            print("Decref needed")
            Py_XDECREF(<PyObject *> self._lut)

        # return tmp_ary.view(dtype=dtype_lut)
        return numpy.core.records.array(tmp_ary.view(dtype=dtype_lut),
                                        shape=self._lut.shape, dtype=dtype_lut,
                                        copy=True)

    lut = property(get_lut)

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
        cdef numpy.int32_t i = 0, j = 0, idx = 0, bins = self.bins, lut_size = self.lut_size, size = self.size
        cdef double sum_data = 0, sum_count = 0, epsilon = 1e-10
        cdef float data = 0, coef = 0, cdummy = 0, cddummy = 0
        cdef bint do_dummy = False, do_dark = False, do_flat = False, do_polarization = False, do_solidAngle = False
        cdef numpy.ndarray[numpy.float64_t, ndim=1] outData = numpy.zeros(self.bins, dtype=numpy.float64)
        cdef numpy.ndarray[numpy.float64_t, ndim=1] outCount = numpy.zeros(self.bins, dtype=numpy.float64)
        cdef numpy.ndarray[numpy.float32_t, ndim=1] outMerge = numpy.zeros(self.bins, dtype=numpy.float32)
        cdef float[:] cdata, tdata, cflat, cdark, csolidAngle, cpolarization

        # Ugly hack against bug #89: https://github.com/silx-kit/pyFAI/issues/89
        cdef int rc_before, rc_after
        rc_before = sys.getrefcount(self._lut)
        cdef lut_point[:, :] lut = self._lut
        rc_after = sys.getrefcount(self._lut)
        cdef bint need_decref = NEED_DECREF & ((rc_after - rc_before) >= 2)

        assert size == weights.size

        if dummy is not None:
            do_dummy = True
            cdummy = <float> float(dummy)
            if delta_dummy is None:
                cddummy = <float> 0.0
            else:
                cddummy = <float> float(delta_dummy)

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
                        # Nota: -= and /= operatore are seen as reduction in cython parallel.
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
        # TODO: what is the best: static or guided ?
        for i in prange(bins, nogil=True, schedule="guided"):
            sum_data = 0.0
            sum_count = 0.0
            for j in range(lut_size):
                idx = lut[i, j].idx
                coef = lut[i, j].coef
                if idx <= 0 and coef <= 0.0:
                    continue
                data = cdata[idx]
                if do_dummy and data == cdummy:
                    continue

                sum_data = sum_data + coef * data
                sum_count = sum_count + coef
            outData[i] += sum_data
            outCount[i] += sum_count
            if sum_count > epsilon:
                outMerge[i] += sum_data / sum_count
            else:
                outMerge[i] += cdummy

        # Ugly against bug#89
        if need_decref and (sys.getrefcount(self._lut) >= rc_before + 2):
            print("Decref needed")
            Py_XDECREF(<PyObject *> self._lut)
        return self.outPos, outMerge, outData, outCount
