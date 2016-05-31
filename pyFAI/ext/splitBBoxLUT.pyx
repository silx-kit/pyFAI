# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) 2012-2016 European Synchrotron Radiation Facility, Grenoble, France
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

__doc__ = """Calculates histograms of pos0 (tth) weighted by Intensity

Splitting is done on the pixel's bounding box like fit2D,
reverse implementation based on a sparse matrix multiplication
"""
__author__ = "Jerome Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "31/05/2016"
__status__ = "stable"
__license__ = "MIT"
import cython
import os
import sys
import logging
logger = logging.getLogger("pyFAI.splitBBoxLUT")
from cpython.ref cimport PyObject, Py_XDECREF
from cython.parallel import prange
from libc.string cimport memset, memcpy
from cython cimport view
import numpy
cimport numpy

include "regrid_common.pxi"

cdef struct lut_point:
    numpy.int32_t idx
    numpy.float32_t coef

dtype_lut = numpy.dtype([("idx", numpy.int32), ("coef", numpy.float32)])

try:
    from fastcrc import crc32
except:
    from zlib import crc32


def int0(a):
    try:
        res = int(a)
    except ValueError:
        res = 0
    return res
numpy_version = tuple(int0(i) for i in numpy.version.version.split("."))
cdef bint NEED_DECREF = (sys.version_info < (2, 7)) and (numpy_version < (1, 5))


class HistoBBox1d(object):
    """
    1D histogramming with pixel splitting based on a Look-up table

    The initialization of the class can take quite a while (operation are not parallelized)
    but each integrate is parallelized and quite efficient.
    """

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
                 allow_pos0_neg=False,
                 unit="undefined",
                 empty=0.0):
        """
        @param pos0: 1D array with pos0: tth or q_vect or r ...
        @param delta_pos0: 1D array with delta pos0: max center-corner distance
        @param pos1: 1D array with pos1: chi
        @param delta_pos1: 1D array with max pos1: max center-corner distance, unused !
        @param bins: number of output bins, 100 by default
        @param pos0Range: minimum and maximum  of the 2th range
        @param pos1Range: minimum and maximum  of the chi range
        @param mask: array (of int8) with masked pixels with 1 (0=not masked)
        @param allow_pos0_neg: enforce the q<0 is usually not possible
        @param unit: can be 2th_deg or r_nm^-1 ...
        @param empty: value for bins without contributing pixels
        """

        self.size = pos0.size
        assert delta_pos0.size == self.size
        self.bins = bins
        self.lut_size = 0
        self.allow_pos0_neg = allow_pos0_neg
        self.empty = empty
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

        self.cpos0 = numpy.ascontiguousarray(pos0.ravel(), dtype=numpy.float32)
        self.dpos0 = numpy.ascontiguousarray(delta_pos0.ravel(), dtype=numpy.float32)
        self.cpos0_sup = numpy.empty_like(self.cpos0)  # self.cpos0 + self.dpos0
        self.cpos0_inf = numpy.empty_like(self.cpos0)  # self.cpos0 - self.dpos0
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
        self._lut_checksum = None
        self.calc_lut()
        self.outPos = numpy.linspace(self.pos0_min + 0.5 * self.delta, self.pos0_maxin - 0.5*self.delta, self.bins)

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
        calculate the max number of elements in the LUT and populate it

        """
        cdef:
            float delta = self.delta, pos0_min = self.pos0_min, pos1_min, pos1_max, min0, max0, fbin0_min, fbin0_max, deltaL, deltaR, deltaA
            numpy.int32_t k, idx, bin0_min, bin0_max, bins = self.bins, lut_size, i, size
            bint check_mask, check_pos1
            numpy.ndarray[numpy.int32_t, ndim = 1] outMax = numpy.zeros(bins, dtype=numpy.int32)
            float[:] cpos0_sup = self.cpos0_sup
            float[:] cpos0_inf = self.cpos0_inf
            float[:] cpos1_min, cpos1_max
            lut_point[:, :] lut
            numpy.int8_t[:] cmask

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

                fbin0_min = get_bin_number(min0, pos0_min, delta)
                fbin0_max = get_bin_number(max0, pos0_min, delta)
                bin0_min = < numpy.int32_t > fbin0_min
                bin0_max = < numpy.int32_t > fbin0_max

                if (bin0_max < 0) or (bin0_min >= bins):
                    continue
                if bin0_max >= bins :
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
        memset(&outMax[0], 0, bins * sizeof(numpy.int32_t))

        self.lut_size = lut_size

        lut_nbytes = bins * lut_size * sizeof(lut_point)
        if (os.name == "posix") and ("SC_PAGE_SIZE" in os.sysconf_names) and ("SC_PHYS_PAGES" in os.sysconf_names):
            try:
                memsize = os.sysconf("SC_PAGE_SIZE")*os.sysconf("SC_PHYS_PAGES")
            except OSError:
                pass
            else:
                if memsize < lut_nbytes:
                    raise MemoryError("Lookup-table (%i, %i) is %.3fGB whereas the memory of the system is only %s" %
                                      (bins, lut_size, lut_nbytes, memsize))

        # else hope we have enough memory
        if (bins == 0) or (lut_size == 0):
            #fix 271
            raise RuntimeError("The look-up table has dimension (%s,%s) which is a non-sense."%(bins, lut_size)
                               + "Did you mask out all pixel or is your image out of the geometry range ?")
        lut = view.array(shape=(bins, lut_size), itemsize=sizeof(lut_point), format="if")
        memset(&lut[0,0], 0, lut_nbytes)

        #NOGIL
        with nogil:
            for idx in range(size):
                if (check_mask) and (cmask[idx]):
                    continue

                min0 = cpos0_inf[idx]
                max0 = cpos0_sup[idx]

                if check_pos1 and ((cpos1_max[idx] < pos1_min) or (cpos1_min[idx] > pos1_max)):
                        continue

                fbin0_min = get_bin_number(min0, pos0_min, delta)
                fbin0_max = get_bin_number(max0, pos0_min, delta)
                bin0_min = < numpy.int32_t > fbin0_min
                bin0_max = < numpy.int32_t > fbin0_max

                if (bin0_max < 0) or (bin0_min >= bins):
                    continue
                if bin0_max >= bins :
                    bin0_max = bins - 1
                if bin0_min < 0:
                    bin0_min = 0

                if bin0_min == bin0_max:
                    # All pixel is within a single bin
                    k = outMax[bin0_min]
                    lut[bin0_min, k].idx = idx
                    lut[bin0_min, k].coef = 1.0
                    outMax[bin0_min] += 1
                else:
                    # we have pixel splitting.
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
        """Getter for the LUT as actual numpy array:
        there is an issue with python2.6 and ref counting"""
        cdef int rc_before, rc_after
        rc_before = sys.getrefcount(self._lut)
        cdef lut_point[:, :] lut = self._lut
        rc_after = sys.getrefcount(self._lut)
        cdef bint need_decref = NEED_DECREF and ((rc_after - rc_before) >= 2)
        cdef numpy.ndarray[numpy.float64_t, ndim = 2] tmp_ary = numpy.empty(shape=self._lut.shape, dtype=numpy.float64)
        memcpy(&tmp_ary[0,0], &lut[0,0], self._lut.nbytes)
        self._lut_checksum = crc32(tmp_ary)

        # Ugly against bug#89
        if need_decref and (sys.getrefcount(self._lut) >= rc_before+2):
            logger.warning("Decref needed")
            Py_XDECREF(<PyObject *> self._lut)
        return numpy.core.records.array(tmp_ary.view(dtype=dtype_lut),
                                        shape=self._lut.shape, dtype=dtype_lut,
                                        copy=True)
    lut = property(get_lut)

    def get_lut_checksum(self):
        if self._lut_checksum is None:
            self.get_lut()
        return self._lut_checksum
    lut_checksum = property(get_lut_checksum)

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
            numpy.int32_t i = 0, j = 0, idx = 0, bins = self.bins, lut_size = self.lut_size, size = self.size
            double sum_data = 0, sum_count = 0, epsilon = 1e-10
            float data = 0, coef = 0, cdummy = 0, cddummy = 0
            bint do_dummy = False, do_dark = False, do_flat = False, do_polarization = False, do_solidAngle = False
            numpy.ndarray[numpy.float64_t, ndim = 1] outData = numpy.zeros(self.bins, dtype=numpy.float64)
            numpy.ndarray[numpy.float64_t, ndim = 1] outCount = numpy.zeros(self.bins, dtype=numpy.float64)
            numpy.ndarray[numpy.float32_t, ndim = 1] outMerge = numpy.zeros(self.bins, dtype=numpy.float32)
            float[:] cdata, tdata, cflat, cdark, csolidAngle, cpolarization

            #Ugly hack against bug #89: https://github.com/pyFAI/pyFAI/issues/89
            int rc_before, rc_after
        rc_before = sys.getrefcount(self._lut)
        cdef lut_point[:, :] lut = self._lut
        rc_after = sys.getrefcount(self._lut)
        cdef bint need_decref = NEED_DECREF & ((rc_after - rc_before) >= 2)

        assert size == weights.size

        if dummy is not None:
            do_dummy = True
            cdummy = <float> float(dummy)
            output_dummy = cdummy
            if delta_dummy is None:
                cddummy = zerof
            else:
                cddummy = <float> float(delta_dummy)
        else:
            cdummy = self.empty

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
                    if ((cddummy !=0 ) and (fabs(data - cdummy) > cddummy)) or ((cddummy == 0) and (data != cdummy)):
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
                    else:
                        # set all dummy_like values to cdummy. simplifies further processing
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
            for j in range(lut_size):
                idx = lut[i, j].idx
                coef = lut[i, j].coef
                if idx <= 0 and coef <= 0.0:
                    continue
                data = cdata[idx]
                if do_dummy and (data == cdummy):
                    continue

                sum_data = sum_data + coef * data
                sum_count = sum_count + coef
            outData[i] += sum_data
            outCount[i] += sum_count
            if sum_count > epsilon:
                outMerge[i] += <float>(sum_data / sum_count / normalization_factor)
            else:
                outMerge[i] += cdummy

        # Ugly against bug#89
        if need_decref and (sys.getrefcount(self._lut) >= rc_before + 2):
            print("Decref needed")
            Py_XDECREF(<PyObject *> self._lut)

        return self.outPos, outMerge, outData, outCount

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def integrate_kahan(self, weights,
                        dummy=None,
                        delta_dummy=None,
                        dark=None,
                        flat=None,
                        solidAngle=None,
                        polarization=None,
                        double normalization_factor=1.0):
        """
        Actually perform the integration which in this case looks more like a matrix-vector product
        Single precision implementation using Kahan summation

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
            numpy.int32_t i = 0, j = 0, idx = 0, bins = self.bins, lut_size = self.lut_size, size = self.size
            float sum_data = 0, sum_count = 0, epsilon = 1e-10
            float data = 0, coef = 0, cdummy = 0, cddummy = 0
            bint do_dummy = False, do_dark = False, do_flat = False, do_polarization = False, do_solidAngle = False
            numpy.ndarray[numpy.float32_t, ndim = 1] outData = numpy.zeros(self.bins, dtype=numpy.float32)
            numpy.ndarray[numpy.float32_t, ndim = 1] outCount = numpy.zeros(self.bins, dtype=numpy.float32)
            numpy.ndarray[numpy.float32_t, ndim = 1] outMerge = numpy.zeros(self.bins, dtype=numpy.float32)
            float[:] cdata, tdata, cflat, cdark, csolidAngle, cpolarization
            float c_data, y_data, t_data
            float c_count, y_count, t_count

        #Ugly hack against bug #89: https://github.com/pyFAI/pyFAI/issues/89
        cdef int rc_before, rc_after
        rc_before = sys.getrefcount(self._lut)
        cdef lut_point[:,:] lut = self._lut
        rc_after = sys.getrefcount(self._lut)
        cdef bint need_decref = NEED_DECREF & ((rc_after-rc_before)>=2)

        assert size == weights.size

        if dummy is not None:
            do_dummy = True
            cdummy = <float>float(dummy)
            if delta_dummy is None:
                cddummy = zerof
            else:
                cddummy = <float> float(delta_dummy)
        else:
            cdummy = <float> float(self.empty)

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
                    else:
                        # set all dummy_like values to cdummy. simplifies further processing
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
            c_data = 0.0
            c_count = 0.0
            for j in range(lut_size):
                idx = lut[i, j].idx
                coef = lut[i, j].coef
                if idx <= 0 and coef <= 0.0:
                    continue
                data = cdata[idx]
                if do_dummy and data==cdummy:
                    continue
# function KahanSum(input)
#     var sum = 0.0
#     var c = 0.0                  // A running compensation for lost low-order bits.
#     for i = 1 to input.length do
#         var y = input[i] - c     // So far, so good: c is zero.
#         var t = sum + y          // Alas, sum is big, y small, so low-order digits of y are lost.
#         c = (t - sum) - y // (t - sum) recovers the high-order part of y; subtracting y recovers -(low part of y)
#         sum = t           // Algebraically, c should always be zero. Beware overly-aggressive optimizing compilers!
#         // Next time around, the lost low part will be added to y in a fresh attempt.
#     return sum

                # sum_data = sum_data + coef * data
                y_data = coef * data - c_data
                t_data = sum_data + y_data
                c_data = (t_data - sum_data) - y_data
                sum_data = t_data

                # sum_count = sum_count + coef
                y_count = coef - c_count
                t_count = sum_count + y_count
                c_count = (t_count - sum_count) - y_count
                sum_count = t_count

            outData[i] += sum_data
            outCount[i] += sum_count
            if sum_count > epsilon:
                outMerge[i] += <float> (sum_data / sum_count / normalization_factor)
            else:
                outMerge[i] += cdummy

        # Ugly against bug#89
        if need_decref and (sys.getrefcount(self._lut)>=rc_before+2):
            logger.warning("Decref needed")
            Py_XDECREF(<PyObject *> self._lut)

        return self.outPos, outMerge, outData, outCount


################################################################################
# Bidimensionnal regrouping
################################################################################

class HistoBBox2d(object):
    """
    2D histogramming with pixel splitting based on a look-up table

    The initialization of the class can take quite a while (operation are not parallelized)
    but each integrate is parallelized and quite efficient.
    """
    @cython.boundscheck(False)
    def __init__(self,
                 pos0,
                 delta_pos0,
                 pos1,
                 delta_pos1,
                 bins=(100, 36),
                 pos0Range=None,
                 pos1Range=None,
                 mask=None,
                 mask_checksum=None,
                 allow_pos0_neg=False,
                 unit="undefined",
                 chiDiscAtPi=True
                 ):
        """
        @param pos0: 1D array with pos0: tth or q_vect
        @param delta_pos0: 1D array with delta pos0: max center-corner distance
        @param pos1: 1D array with pos1: chi
        @param delta_pos1: 1D array with max pos1: max center-corner distance, unused !
        @param bins: number of output bins (tth=100, chi=36 by default)
        @param pos0Range: minimum and maximum  of the 2th range
        @param pos1Range: minimum and maximum  of the chi range
        @param mask: array (of int8) with masked pixels with 1 (0=not masked)
        @param allow_pos0_neg: enforce the q<0 is usually not possible
        @param chiDiscAtPi: boolean; by default the chi_range is in the range ]-pi,pi[ set to 0 to have the range ]0,2pi[
        @param unit: can be 2th_deg or r_nm^-1 ...
        """
        cdef numpy.int32_t i, size, bin0, bin1
        self.size = pos0.size
        assert delta_pos0.size == self.size
        assert pos1.size == self.size
        assert delta_pos1.size == self.size
        self.chiDiscAtPi = 1 if chiDiscAtPi else 0
        self.allow_pos0_neg = allow_pos0_neg

        try:
            bins0, bins1 = tuple(bins)
        except:
            bins0 = bins1 = bins
        if bins0 <= 0:
            bins0 = 1
        if bins1 <= 0:
            bins1 = 1
        self.bins = (int(bins0), int(bins1))
        self.lut_size = 0
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

        self.cpos0 = numpy.ascontiguousarray(pos0.ravel(), dtype=numpy.float32)
        self.dpos0 = numpy.ascontiguousarray(delta_pos0.ravel(), dtype=numpy.float32)
        self.cpos0_sup = numpy.empty_like(self.cpos0)
        self.cpos0_inf = numpy.empty_like(self.cpos0)
        self.pos0Range = pos0Range
        self.pos1Range = pos1Range

        self.cpos1 = numpy.ascontiguousarray((pos1).ravel(), dtype=numpy.float32)
        self.dpos1 = numpy.ascontiguousarray((delta_pos1).ravel(), dtype=numpy.float32)
        self.cpos1_sup = numpy.empty_like(self.cpos1)
        self.cpos1_inf = numpy.empty_like(self.cpos1)
        self.calc_boundaries(pos0Range, pos1Range)
        self.delta0 = (self.pos0_max - self.pos0_min) / float(bins0)
        self.delta1 = (self.pos1_max - self.pos1_min) / float(bins1)
        self.lut_max_idx = None
        self._lut = None
        self.calc_lut()
        self.outPos0 = numpy.linspace(self.pos0_min + 0.5*self.delta0, self.pos0_maxin - 0.5 * self.delta0, bins0)
        self.outPos1 = numpy.linspace(self.pos1_min + 0.5*self.delta1, self.pos1_maxin - 0.5 * self.delta1, bins1)
        self.unit = unit
        # Calculated at export time to python
        self._lut_checksum = None

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def calc_boundaries(self, pos0Range, pos1Range):
        """
        Called by constructor to calculate the boundaries and the bin position
        """

        cdef numpy.int32_t size = self.cpos0.size
        cdef bint check_mask = self.check_mask
        cdef numpy.int8_t[:] cmask
        cdef float[:] cpos0, dpos0, cpos0_sup, cpos0_inf
        cdef float[:] cpos1, dpos1, cpos1_sup, cpos1_inf,
        cdef float upper0, lower0, pos0_max, pos0_min, c0, d0
        cdef float upper1, lower1, pos1_max, pos1_min, c1, d1
        cdef bint allow_pos0_neg = self.allow_pos0_neg
        cdef bint chiDiscAtPi = self.chiDiscAtPi

        cpos0_sup = self.cpos0_sup
        cpos0_inf = self.cpos0_inf
        cpos0 = self.cpos0
        dpos0 = self.dpos0
        cpos1_sup = self.cpos1_sup
        cpos1_inf = self.cpos1_inf
        cpos1 = self.cpos1
        dpos1 = self.dpos1
        pos0_min = cpos0[0]
        pos0_max = cpos0[0]
        pos1_min = cpos1[0]
        pos1_max = cpos1[0]

        if check_mask:
            cmask = self.cmask
        with nogil:
            for idx in range(size):
                c0 = cpos0[idx]
                d0 = dpos0[idx]
                lower0 = c0 - d0
                upper0 = c0 + d0
                c1 = cpos1[idx]
                d1 = dpos1[idx]
                lower1 = c1 - d1
                upper1 = c1 + d1
                if not allow_pos0_neg and lower0<0:
                    lower0 = 0
                if upper1 > (2 - chiDiscAtPi) * pi:
                    upper1 = (2 - chiDiscAtPi) * pi
                if lower1 < (-chiDiscAtPi) * pi:
                    lower1 = (-chiDiscAtPi) * pi
                cpos0_sup[idx] = upper0
                cpos0_inf[idx] = lower0
                cpos1_sup[idx] = upper1
                cpos1_inf[idx] = lower1
                if not (check_mask and cmask[idx]):
                    if upper0 > pos0_max:
                        pos0_max = upper0
                    if lower0 < pos0_min:
                        pos0_min = lower0
                    if upper1 > pos1_max:
                        pos1_max = upper1
                    if lower1 < pos1_min:
                        pos1_min = lower1

        if pos0Range is not None and len(pos0Range) > 1:
            self.pos0_min = min(pos0Range)
            self.pos0_maxin = max(pos0Range)
        else:
            self.pos0_min = pos0_min
            self.pos0_maxin = pos0_max

        if (pos1Range is not None) and (len(pos1Range) > 1):
            self.pos1_min = min(pos1Range)
            self.pos1_maxin = max(pos1Range)
        else:
            self.pos1_min = pos1_min
            self.pos1_maxin = pos1_max

        if (not allow_pos0_neg) and self.pos0_min < 0:
            self.pos0_min = 0
        self.pos0_max = self.pos0_maxin * EPS32
        self.cpos0_sup = cpos0_sup
        self.cpos0_inf = cpos0_inf
        self.pos1_max = self.pos1_maxin * EPS32
        self.cpos1_sup = cpos1_sup
        self.cpos1_inf = cpos1_inf

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def calc_lut(self):
        'calculate the max number of elements in the LUT and populate it'
        cdef:
            float delta0 = self.delta0, pos0_min = self.pos0_min, min0, max0, fbin0_min, fbin0_max
            float delta1 = self.delta1, pos1_min = self.pos1_min, min1, max1, fbin1_min, fbin1_max
            numpy.int32_t bin0_min, bin0_max, bins0 = self.bins[0]
            numpy.int32_t bin1_min, bin1_max, bins1 = self.bins[1]
            numpy.int32_t k, idx, lut_size, i, j, size = self.size
            bint check_mask
            float[:] cpos0_sup = self.cpos0_sup
            float[:] cpos0_inf = self.cpos0_inf
            float[:] cpos1_inf = self.cpos1_inf
            float[:] cpos1_sup = self.cpos1_sup
            numpy.ndarray[numpy.int32_t, ndim = 2] outMax = numpy.zeros((bins0, bins1), dtype=numpy.int32)
            lut_point[:, :, :] lut
            numpy.int8_t[:] cmask
        if self.check_mask:
            cmask = self.cmask
            check_mask = True
        else:
            check_mask = False

#NOGIL
        with nogil:
            for idx in range(size):
                if (check_mask) and (cmask[idx]):
                    continue

                min0 = cpos0_inf[idx]
                max0 = cpos0_sup[idx]
                min1 = cpos1_inf[idx]
                max1 = cpos1_sup[idx]

                bin0_min = < numpy.int32_t > get_bin_number(min0, pos0_min, delta0)
                bin0_max = < numpy.int32_t > get_bin_number(max0, pos0_min, delta0)

                bin1_min = < numpy.int32_t > get_bin_number(min1, pos1_min, delta1)
                bin1_max = < numpy.int32_t > get_bin_number(max1, pos1_min, delta1)

                if (bin0_max < 0) or (bin0_min >= bins0) or (bin1_max < 0) or (bin1_min >= bins1):
                    continue

                if bin0_max >= bins0 :
                    bin0_max = bins0 - 1
                if bin0_min < 0:
                    bin0_min = 0
                if bin1_max >= bins1 :
                    bin1_max = bins1 - 1
                if bin1_min < 0:
                    bin1_min = 0

                for i in range(bin0_min, bin0_max + 1):
                    for j in range(bin1_min, bin1_max + 1):
                        outMax[i, j] += 1

        self.lut_size = lut_size = outMax.max()
        # just recycle the outMax array
        memset(&outMax[0,0], 0, bins0 * bins1 * sizeof(numpy.int32_t))

        lut_nbytes = bins0 * bins1 * lut_size * sizeof(lut_point)
        if (os.name == "posix") and ("SC_PAGE_SIZE" in os.sysconf_names) and ("SC_PHYS_PAGES" in os.sysconf_names):
            try:
                memsize = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            except OSError:
                pass
            else:
                if memsize < lut_nbytes:
                    raise MemoryError("Lookup-table (%i, %i, %i) is %.3fGB whereas the memory of the system is only %s" %
                                      (bins0, bins1, lut_size, lut_nbytes, memsize))
        # else hope we have enough memory
        lut = view.array(shape=(bins0, bins1, lut_size), itemsize=sizeof(lut_point), format="if")
        memset(&lut[0, 0, 0], 0, lut_nbytes)

        # NOGIL
        with nogil:
            for idx in range(size):
                if (check_mask) and cmask[idx]:
                    continue

                min0 = cpos0_inf[idx]
                max0 = cpos0_sup[idx]
                min1 = cpos1_inf[idx]
                max1 = cpos1_sup[idx]

                fbin0_min = get_bin_number(min0, pos0_min, delta0)
                fbin0_max = get_bin_number(max0, pos0_min, delta0)
                fbin1_min = get_bin_number(min1, pos1_min, delta1)
                fbin1_max = get_bin_number(max1, pos1_min, delta1)

                bin0_min = <int> fbin0_min
                bin0_max = <int> fbin0_max
                bin1_min = <int> fbin1_min
                bin1_max = <int> fbin1_max

                if (bin0_max < 0) or (bin0_min >= bins0) or (bin1_max < 0) or (bin1_min >= bins1):
                    continue

                if bin0_max >= bins0:
                    bin0_max = bins0 - 1
                if bin0_min < 0:
                    bin0_min = 0
                if bin1_max >= bins1:
                    bin1_max = bins1 - 1
                if bin1_min < 0:
                    bin1_min = 0

                if bin0_min == bin0_max:
                    if bin1_min == bin1_max:
                        # All pixel is within a single bin
                        k = outMax[bin0_min, bin1_min]
                        lut[bin0_min, bin1_min, k].idx = idx
                        lut[bin0_min, bin1_min, k].coef = 1.0
                        outMax[bin0_min, bin1_min] = k+1

                    else:
                        # spread on more than 2 bins
                        deltaD = (< float > (bin1_min + 1)) - fbin1_min
                        deltaU = fbin1_max - bin1_max
                        deltaA = 1.0 / (fbin1_max - fbin1_min)

                        k = outMax[bin0_min, bin1_min]
                        lut[bin0_min, bin1_min, k].idx = idx
                        lut[bin0_min, bin1_min, k].coef = deltaA * deltaD
                        outMax[bin0_min, bin1_min] += 1

                        k = outMax[bin0_min, bin1_max]
                        lut[bin0_min, bin1_max, k].idx = idx
                        lut[bin0_min, bin1_max, k].coef = deltaA * deltaU
                        outMax[bin0_min, bin1_max] += 1

                        for j in range(bin1_min + 1, bin1_max):
                            k = outMax[bin0_min, j]
                            lut[bin0_min, j, k].idx = idx
                            lut[bin0_min, j, k].coef = deltaA
                            outMax[bin0_min, j] += 1

                else:
                    # spread on more than 2 bins in dim 0
                    if bin1_min == bin1_max:
                        # All pixel fall on 1 bins in dim 1
                        deltaA = 1.0 / (fbin0_max - fbin0_min)
                        deltaL = (< float > (bin0_min + 1)) - fbin0_min

                        k = outMax[bin0_min, bin1_min]
                        lut[bin0_min, bin1_min, k].idx = idx
                        lut[bin0_min, bin1_min, k].coef = deltaA * deltaL
                        outMax[bin0_min, bin1_min] = k+1

                        deltaR = fbin0_max - (< float > bin0_max)

                        k = outMax[bin0_max, bin1_min]
                        lut[bin0_max, bin1_min, k].idx = idx
                        lut[bin0_max, bin1_min, k].coef = deltaA * deltaR
                        outMax[bin0_max, bin1_min] += 1

                        for i in range(bin0_min + 1, bin0_max):
                            k = outMax[i, bin1_min]
                            lut[i, bin1_min, k].idx = idx
                            lut[i, bin1_min, k].coef = deltaA
                            outMax[i, bin1_min] += 1

                    else:
                        # spread on n pix in dim0 and m pixel in dim1:
                        deltaL = (< float > (bin0_min + 1)) - fbin0_min
                        deltaR = fbin0_max - (< float > bin0_max)
                        deltaD = (< float > (bin1_min + 1)) - fbin1_min
                        deltaU = fbin1_max - (< float > bin1_max)
                        deltaA = 1.0 / ((fbin0_max - fbin0_min) * (fbin1_max - fbin1_min))

                        k = outMax[bin0_min, bin1_min]
                        lut[bin0_min, bin1_min, k].idx = idx
                        lut[bin0_min, bin1_min, k].coef = deltaA * deltaL * deltaD
                        outMax[bin0_min, bin1_min] += 1

                        k = outMax[bin0_min, bin1_max]
                        lut[bin0_min, bin1_max, k].idx = idx
                        lut[bin0_min, bin1_max, k].coef = deltaA * deltaL * deltaU
                        outMax[bin0_min, bin1_max] += 1

                        k = outMax[bin0_max, bin1_min]
                        lut[bin0_max, bin1_min, k].idx = idx
                        lut[bin0_max, bin1_min, k].coef = deltaA * deltaR * deltaD
                        outMax[bin0_max, bin1_min] += 1

                        k = outMax[bin0_max, bin1_max]
                        lut[bin0_max, bin1_max, k].idx = idx
                        lut[bin0_max, bin1_max, k].coef = deltaA * deltaR * deltaU
                        outMax[bin0_max, bin1_max] += 1

                        for i in range(bin0_min + 1, bin0_max):
                            k = outMax[i, bin1_min]
                            lut[i, bin1_min, k].idx = idx
                            lut[i, bin1_min, k].coef = deltaA * deltaD
                            outMax[i, bin1_min] += 1

                            for j in range(bin1_min + 1, bin1_max):
                                k = outMax[i, j]
                                lut[i, j, k].idx = idx
                                lut[i, j, k].coef = deltaA
                                outMax[i, j] += 1

                            k = outMax[i, bin1_max]
                            lut[i, bin1_max, k].idx = idx
                            lut[i, bin1_max, k].coef = deltaA * deltaU
                            outMax[i, bin1_max] += 1

                        for j in range(bin1_min + 1, bin1_max):
                            k = outMax[bin0_min, j]
                            lut[bin0_min, j, k].idx = idx
                            lut[bin0_min, j, k].coef = deltaA * deltaL
                            outMax[bin0_min, j] += 1

                            k = outMax[bin0_max, j]
                            lut[bin0_max, j, k].idx = idx
                            lut[bin0_max, j, k].coef = deltaA * deltaR
                            outMax[bin0_max, j] += 1

        self.lut_max_idx = outMax
        self._lut = lut

    def get_lut(self):
        """Getter for the LUT as actual numpy array
        Hack against a bug in ref-counting under python2.6
        """
        cdef int rc_before, rc_after
        rc_before = sys.getrefcount(self._lut)
        cdef lut_point[:, :, :] lut = self._lut
        rc_after = sys.getrefcount(self._lut)
        cdef bint need_decref = NEED_DECREF and ((rc_after - rc_before) >= 2)
        shape = (self._lut.shape[0] * self._lut.shape[1], self._lut.shape[2])
        cdef numpy.ndarray[numpy.float64_t, ndim = 2] tmp_ary = numpy.empty(shape=shape, dtype=numpy.float64)
        memcpy(&tmp_ary[0,0], &lut[0,0,0], self._lut.nbytes)
        self._lut_checksum = crc32(tmp_ary)

        #Ugly against bug#89
        if need_decref and (sys.getrefcount(self._lut) >= rc_before+2):
            print("Warning: Decref needed")
            Py_XDECREF(<PyObject *> self._lut)

        return numpy.core.records.array(tmp_ary.view(dtype=dtype_lut),
                                        shape=shape, dtype=dtype_lut,
                                        copy=True)
    lut = property(get_lut)

    def get_lut_checksum(self):
        if self._lut_checksum is None:
            self.get_lut()
        return self._lut_checksum
    lut_checksum = property(get_lut_checksum)

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
                  double normalization_factor=1.0):
        """
        Actually perform the 2D integration which in this case looks more like a matrix-vector product

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

        @return:  I(2d), edges0(1d), edges1(1d), weighted histogram(2d), unweighted histogram (2d)
        @rtype: 5-tuple of ndarrays

        """
        cdef:
            numpy.int32_t i = 0, j = 0, idx = 0, bins0 = self.bins[0], bins1 = self.bins[1], bins = bins0 * bins1, lut_size = self.lut_size, size = self.size, i0 = 0, i1 = 0
            double sum_data = 0, sum_count = 0, epsilon = 1e-10
            float data = 0, coef = 0, cdummy = 0, cddummy = 0
            bint do_dummy = False, do_dark = False, do_flat = False, do_polarization = False, do_solidAngle = False
            numpy.ndarray[numpy.float64_t, ndim = 2] outData = numpy.zeros(self.bins, dtype=numpy.float64)
            numpy.ndarray[numpy.float64_t, ndim = 2] outCount = numpy.zeros(self.bins, dtype=numpy.float64)
            numpy.ndarray[numpy.float32_t, ndim = 2] outMerge = numpy.zeros(self.bins, dtype=numpy.float32)
            float[:] cdata, tdata, cflat, cdark, csolidAngle, cpolarization
        # Ugly hack against bug #89
            int rc_before, rc_after
        rc_before = sys.getrefcount(self._lut)
        cdef lut_point[:, :, :] lut = self._lut
        rc_after = sys.getrefcount(self._lut)
        cdef bint need_decref = NEED_DECREF and ((rc_after - rc_before) >= 2)

        assert size == weights.size

        if dummy is not None:
            do_dummy = True
            cdummy = <float> float(dummy)
            if delta_dummy is None:
                cddummy = zerof
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
                    else:
                        # set all dummy_like values to cdummy. simplifies further processing
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

        for i0 in prange(bins0, nogil=True, schedule="guided"):
            for i1 in range(bins1):
                sum_data = 0.0
                sum_count = 0.0
                for j in range(lut_size):
                    idx = lut[i0, i1, j].idx
                    coef = lut[i0, i1, j].coef
                    if idx <= 0 and coef <= 0.0:
                        continue
                    data = cdata[idx]
                    if do_dummy and data == cdummy:
                        continue

                    sum_data = sum_data + coef * data
                    sum_count = sum_count + coef
                outData[i0, i1] += sum_data
                outCount[i0, i1] += sum_count
                if sum_count > epsilon:
                    outMerge[i0, i1] += <float> (sum_data / sum_count / normalization_factor)
                else:
                    outMerge[i0, i1] += cdummy

        # Ugly against bug #89
        if need_decref and (sys.getrefcount(self._lut) >= rc_before + 2):
            Py_XDECREF(<PyObject *> self._lut)

        return outMerge.T, self.outPos0, self.outPos1, outData.T, outCount.T

