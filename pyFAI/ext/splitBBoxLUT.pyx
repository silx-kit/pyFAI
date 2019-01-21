# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2018 European Synchrotron Radiation Facility, Grenoble, France
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

"""Calculates histograms of pos0 (tth) weighted by Intensity

Splitting is done on the pixel's bounding box like fit2D,
reverse implementation based on a sparse matrix multiplication
"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "15/01/2019"
__status__ = "stable"
__license__ = "MIT"

include "sparse_common.pxi"
#check LUT is propely represented in C and numpy

import cython
import os
import sys
import logging
logger = logging.getLogger(__name__)
from cpython.ref cimport PyObject, Py_XDECREF
from cython.parallel import prange
from libc.string cimport memset, memcpy
from cython cimport view
import numpy
cimport numpy
from ..utils import crc32
from ..utils.decorators import deprecated


if LUT_ITEMSIZE == lut_d.itemsize == 8:
    logger.debug("LUT sizes C:%s \t Numpy: %s", lut_d.itemsize, LUT_ITEMSIZE)
else:
    logger.error("LUT sizes C:%s \t Numpy: %s", lut_d.itemsize, LUT_ITEMSIZE)
    raise ImportError("Numpy and C have the same internal LUT representation")


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
        :param pos0: 1D array with pos0: tth or q_vect or r ...
        :param delta_pos0: 1D array with delta pos0: max center-corner distance
        :param pos1: 1D array with pos1: chi
        :param delta_pos1: 1D array with max pos1: max center-corner distance, unused !
        :param bins: number of output bins, 100 by default
        :param pos0Range: minimum and maximum  of the 2th range
        :param pos1Range: minimum and maximum  of the chi range
        :param mask: array (of int8) with masked pixels with 1 (0=not masked)
        :param allow_pos0_neg: enforce the q<0 is usually not possible
        :param unit: can be 2th_deg or r_nm^-1 ...
        :param empty: value for bins without contributing pixels
        """

        self.size = pos0.size
        assert delta_pos0.size == self.size, "delta_pos0.size == self.size"
        self.bins = bins
        self.lut_size = 0
        self.allow_pos0_neg = allow_pos0_neg
        self.empty = empty
        if mask is not None:
            assert mask.size == self.size, "mask size"
            self.check_mask = True
            self.cmask = numpy.ascontiguousarray(mask.ravel(), dtype=mask_d)
            if mask_checksum:
                self.mask_checksum = mask_checksum
            else:
                self.mask_checksum = crc32(mask)
        else:
            self.check_mask = False
            self.mask_checksum = None

        self.cpos0 = numpy.ascontiguousarray(pos0.ravel(), dtype=position_d)
        self.dpos0 = numpy.ascontiguousarray(delta_pos0.ravel(), dtype=position_d)
        self.cpos0_sup = numpy.empty_like(self.cpos0)  # self.cpos0 + self.dpos0
        self.cpos0_inf = numpy.empty_like(self.cpos0)  # self.cpos0 - self.dpos0
        self.pos0Range = pos0Range
        self.pos1Range = pos1Range
        self.calc_boundaries(pos0Range)
        if pos1Range is not None and len(pos1Range) > 1:
            assert pos1.size == self.size, "pos1 size"
            assert delta_pos1.size == self.size, "delta_pos1.size == self.size"
            self.check_pos1 = True
            self.cpos1_min = numpy.ascontiguousarray((pos1 - delta_pos1).ravel(), dtype=position_d)
            self.cpos1_max = numpy.ascontiguousarray((pos1 + delta_pos1).ravel(), dtype=position_d)
            self.pos1_min = min(pos1Range)
            pos1_maxin = max(pos1Range)
            self.pos1_max = calc_upper_bound(<position_t> pos1_maxin)
        else:
            self.check_pos1 = False
            self.cpos1_min = None
            self.pos1_max = None

        self.delta = (self.pos0_max - self.pos0_min) / bins
        self._lut = None
        self.lut_max_idx = None
        self._lut_checksum = None
        self.calc_lut()
        self.bin_centers = numpy.linspace(self.pos0_min + 0.5 * self.delta,
                                          self.pos0_max - 0.5 * self.delta,
                                          self.bins)

        self.unit = unit
        self.lut_nbytes = self._lut.nbytes

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def calc_boundaries(self, pos0Range):
        """
        Called by constructor to calculate the boundaries and the bin position
        """
        cdef:
            int size = self.cpos0.size
            bint check_mask = self.check_mask
            mask_t[::1] cmask
            position_t[::1] cpos0, dpos0, cpos0_sup, cpos0_inf,
            position_t upper, lower, pos0_max, pos0_min, c, d
            bint allow_pos0_neg = self.allow_pos0_neg

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
        self.pos0_max = calc_upper_bound(<acc_t> self.pos0_maxin)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def calc_lut(self):
        """
        calculate the max number of elements in the LUT and populate it

        """
        cdef:
            position_t delta = self.delta, pos0_min = self.pos0_min, pos1_min, pos1_max, min0, max0, fbin0_min, fbin0_max 
            acc_t delta_left, delta_right, inv_area
            int k, idx, bin0_min, bin0_max, bins = self.bins, lut_size, i, size
            bint check_mask, check_pos1
            cnumpy.int32_t[::1] outmax = numpy.zeros(bins, dtype=numpy.int32)
            position_t[:] cpos0_sup = self.cpos0_sup
            position_t[:] cpos0_inf = self.cpos0_inf
            position_t[:] cpos1_min, cpos1_max
            lut_t[:, :] lut
            mask_t[:] cmask

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

                fbin0_min = get_bin_number(min0, pos0_min, delta)
                fbin0_max = get_bin_number(max0, pos0_min, delta)
                bin0_min = <int> fbin0_min
                bin0_max = <int> fbin0_max

                if (bin0_max < 0) or (bin0_min >= bins):
                    continue
                if bin0_max >= bins:
                    bin0_max = bins - 1
                if bin0_min < 0:
                    bin0_min = 0

                if bin0_min == bin0_max:
                    # All pixel is within a single bin
                    outmax[bin0_min] += 1

                else:  # we have pixel spliting.
                    for i in range(bin0_min, bin0_max + 1):
                        outmax[i] += 1

        lut_size = numpy.max(outmax)
        # just recycle the outmax array
        outmax[:] = 0

        self.lut_size = lut_size

        lut_nbytes = bins * lut_size * sizeof(lut_t)
        #Check we have enough memory
        if (os.name == "posix"):
            key_page_size = os.sysconf_names.get("SC_PAGE_SIZE", 0)
            key_page_cnt = os.sysconf_names.get("SC_PHYS_PAGES",0)
            if key_page_size*key_page_cnt:
                try:
                    memsize = os.sysconf(key_page_size) * os.sysconf(key_page_cnt)
                except OSError:
                    pass
                else:
                    if memsize < lut_nbytes:
                        raise MemoryError("Lookup-table (%i, %i) is %.3fGB whereas the memory of the system is only %s" %
                                          (bins, lut_size, lut_nbytes/2.**30, memsize/2.**30))
        # else hope we have enough memory

        if (bins == 0) or (lut_size == 0):
            # fix issue #271
            msg = "The look-up table has dimension (%s,%s) which is a non-sense." +\
                  "Did you mask out all pixel or is your image out of the geometry range?"
            raise RuntimeError(msg % (bins, lut_size))
        lut = view.array(shape=(bins, lut_size), itemsize=sizeof(lut_t), format="if")
        memset(&lut[0,0], 0, lut_nbytes)

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
                bin0_min = <int> fbin0_min
                bin0_max = <int> fbin0_max

                if (bin0_max < 0) or (bin0_min >= bins):
                    continue
                if bin0_max >= bins:
                    bin0_max = bins - 1
                if bin0_min < 0:
                    bin0_min = 0

                if bin0_min == bin0_max:
                    # All pixel is within a single bin
                    k = outmax[bin0_min]
                    lut[bin0_min, k].idx = idx
                    lut[bin0_min, k].coef = 1.0
                    outmax[bin0_min] += 1
                else:
                    # we have pixel splitting.
                    inv_area = 1.0 / (fbin0_max - fbin0_min)

                    delta_left = <acc_t>(bin0_min + 1) - fbin0_min
                    delta_right = fbin0_max - (<acc_t> bin0_max)

                    k = outmax[bin0_min]
                    lut[bin0_min, k].idx = idx
                    lut[bin0_min, k].coef = inv_area * delta_left
                    outmax[bin0_min] += 1

                    k = outmax[bin0_max]
                    lut[bin0_max, k].idx = idx
                    lut[bin0_max, k].coef = inv_area * delta_right
                    outmax[bin0_max] += 1

                    if bin0_min + 1 < bin0_max:
                        for i in range(bin0_min + 1, bin0_max):
                            k = outmax[i]
                            lut[i, k].idx = idx
                            lut[i, k].coef = inv_area
                            outmax[i] += 1

        self.lut_max_idx = outmax
        self._lut = lut

    @property
    def lut(self):
        """Getter for the LUT as actual numpy array:
        there is an issue with python2.6 and ref counting"""
        cdef int rc_before, rc_after
        rc_before = sys.getrefcount(self._lut)
        cdef lut_t[:, :] lut = self._lut
        rc_after = sys.getrefcount(self._lut)
        cdef bint need_decref = NEED_DECREF and ((rc_after - rc_before) >= 2)
        cdef numpy.ndarray[numpy.float64_t, ndim=2] tmp_ary = numpy.empty(shape=self._lut.shape, dtype=numpy.float64)
        memcpy(&tmp_ary[0, 0], &lut[0, 0], self._lut.nbytes)
        self._lut_checksum = crc32(tmp_ary)

        # Ugly against bug#89
        if need_decref and (sys.getrefcount(self._lut) >= rc_before + 2):
            logger.warning("Decref needed")
            Py_XDECREF(<PyObject *> self._lut)
        return numpy.core.records.array(tmp_ary.view(dtype=lut_d),
                                        shape=self._lut.shape, dtype=lut_d,
                                        copy=True)

    @property
    def lut_checksum(self):
        if self._lut_checksum is None:
            self.lut
        return self._lut_checksum

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
                  double normalization_factor=1.0,
                  int coef_power=1):
        """
        Actually perform the integration which in this case looks more like a matrix-vector product

        :param weights: input image
        :type weights: ndarray
        :param dummy: value for dead pixels (optional)
        :type dummy: float
        :param delta_dummy: precision for dead-pixel value in dynamic masking
        :type delta_dummy: float
        :param dark: array with the dark-current value to be subtracted (if any)
        :type dark: ndarray
        :param flat: array with the dark-current value to be divided by (if any)
        :type flat: ndarray
        :param solidAngle: array with the solid angle of each pixel to be divided by (if any)
        :type solidAngle: ndarray
        :param polarization: array with the polarization correction values to be divided by (if any)
        :type polarization: ndarray
        :param normalization_factor: divide the valid result by this value
        :param coef_power: put coef to a given power, 2 for variance, 1 for mean

        :return: positions, pattern, weighted_histogram and unweighted_histogram
        :rtype: 4-tuple of ndarrays

        """
        cdef:
            int i = 0, j = 0, idx = 0, bins = self.bins, lut_size = self.lut_size, size = self.size
            acc_t acc_data = 0, acc_count = 0, epsilon = 1e-10
            data_t data = 0, coef = 0, cdummy = 0, cddummy = 0
            bint do_dummy = False, do_dark = False, do_flat = False, do_polarization = False, do_solidAngle = False
            acc_t[::1] sum_data = numpy.zeros(self.bins, dtype=acc_d)
            acc_t[::1] sum_count = numpy.zeros(self.bins, dtype=acc_d)
            data_t[::1] merged = numpy.zeros(self.bins, dtype=data_d)
            float[:] cdata, tdata, cflat, cdark, csolidAngle, cpolarization

            # Ugly hack against bug #89: https://github.com/silx-kit/pyFAI/issues/89
            int rc_before, rc_after
        rc_before = sys.getrefcount(self._lut)
        cdef lut_t[:, :] lut = self._lut
        rc_after = sys.getrefcount(self._lut)
        cdef bint need_decref = NEED_DECREF & ((rc_after - rc_before) >= 2)

        assert weights.size == size, "weights size"

        if dummy is not None:
            do_dummy = True
            cdummy = <data_t> float(dummy)
            if delta_dummy is None:
                cddummy = zerof
            else:
                cddummy = <data_t> float(delta_dummy)
        else:
            cdummy = self.empty

        if flat is not None:
            do_flat = True
            assert flat.size == size, "flat-field array size"
            cflat = numpy.ascontiguousarray(flat.ravel(), dtype=data_d)
        if dark is not None:
            do_dark = True
            assert dark.size == size, "dark current array size"
            cdark = numpy.ascontiguousarray(dark.ravel(), dtype=data_d)
        if solidAngle is not None:
            do_solidAngle = True
            assert solidAngle.size == size, "Solid angle array size"
            csolidAngle = numpy.ascontiguousarray(solidAngle.ravel(), dtype=data_d)
        if polarization is not None:
            do_polarization = True
            assert polarization.size == size, "polarization array size"
            cpolarization = numpy.ascontiguousarray(polarization.ravel(), dtype=data_d)

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
            acc_data = 0.0
            acc_count = 0.0
            for j in range(lut_size):
                idx = lut[i, j].idx
                coef = lut[i, j].coef
                if idx <= 0 and coef <= 0.0:
                    continue
                data = cdata[idx]
                if do_dummy and (data == cdummy):
                    continue

                acc_data = acc_data + (coef ** coef_power) * data
                acc_count = acc_count + coef
            sum_data[i] += acc_data
            sum_count[i] += acc_count
            if acc_count > epsilon:
                merged[i] += <data_t>(acc_data / acc_count / normalization_factor)
            else:
                merged[i] += cdummy

        # Ugly against bug#89
        if need_decref and (sys.getrefcount(self._lut) >= rc_before + 2):
            print("Decref needed")
            Py_XDECREF(<PyObject *> self._lut)

        return (self.bin_centers, 
                numpy.asarray(merged), 
                numpy.asarray(sum_data), 
                numpy.asarray(sum_count))

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
                        double normalization_factor=1.0,
                        int coef_power=1):
        """
        Actually perform the integration which in this case looks more like a matrix-vector product
        Single precision implementation using Kahan summation

        :param weights: input image
        :type weights: ndarray
        :param dummy: value for dead pixels (optional)
        :type dummy: float
        :param delta_dummy: precision for dead-pixel value in dynamic masking
        :type delta_dummy: float
        :param dark: array with the dark-current value to be subtracted (if any)
        :type dark: ndarray
        :param flat: array with the dark-current value to be divided by (if any)
        :type flat: ndarray
        :param solidAngle: array with the solid angle of each pixel to be divided by (if any)
        :type solidAngle: ndarray
        :param polarization: array with the polarization correction values to be divided by (if any)
        :type polarization: ndarray
        :param normalization_factor: divide the valid result by this value
        :param coef_power: set to1 for mean and 2 for variance propagation

        :return: positions, pattern, weighted_histogram and unweighted_histogram
        :rtype: 4-tuple of ndarrays

        """
        cdef:
            cnumpy.int32_t i = 0, j = 0, idx = 0, bins = self.bins, lut_size = self.lut_size, size = self.size
            float acc_data = 0, acc_count = 0, epsilon = 1e-10
            float data = 0, coef = 0, cdummy = 0, cddummy = 0
            bint do_dummy = False, do_dark = False, do_flat = False, do_polarization = False, do_solidAngle = False
            float[::1] sum_data = numpy.zeros(self.bins, dtype=numpy.float32)
            float[::1] sum_count = numpy.zeros(self.bins, dtype=numpy.float32)
            float[::1] merged = numpy.zeros(self.bins, dtype=numpy.float32)
            float[::1] cdata, tdata, cflat, cdark, csolidAngle, cpolarization
            float c_data, y_data, t_data
            float c_count, y_count, t_count

        # Ugly hack against bug #89: https://github.com/silx-kit/pyFAI/issues/89
        cdef int rc_before, rc_after
        rc_before = sys.getrefcount(self._lut)
        cdef lut_t[:, :] lut = self._lut
        rc_after = sys.getrefcount(self._lut)
        cdef bint need_decref = NEED_DECREF & ((rc_after - rc_before) >= 2)

        assert weights.size == size, "weights size"

        if dummy is not None:
            do_dummy = True
            cdummy = <float> float(dummy)
            if delta_dummy is None:
                cddummy = zerof
            else:
                cddummy = <float> float(delta_dummy)
        else:
            cdummy = <float> float(self.empty)

        if flat is not None:
            do_flat = True
            assert flat.size == size, "flat-field array size"
            cflat = numpy.ascontiguousarray(flat.ravel(), dtype=numpy.float32)
        if dark is not None:
            do_dark = True
            assert dark.size == size, "dark current array size"
            cdark = numpy.ascontiguousarray(dark.ravel(), dtype=numpy.float32)
        if solidAngle is not None:
            do_solidAngle = True
            assert solidAngle.size == size, "Solid angle array size"
            csolidAngle = numpy.ascontiguousarray(solidAngle.ravel(), dtype=numpy.float32)
        if polarization is not None:
            do_polarization = True
            assert polarization.size == size, "polarization array size"
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
            acc_data = 0.0
            acc_count = 0.0
            c_data = 0.0
            c_count = 0.0
            for j in range(lut_size):
                idx = lut[i, j].idx
                coef = lut[i, j].coef
                if idx <= 0 and coef <= 0.0:
                    continue
                data = cdata[idx]
                if do_dummy and data == cdummy:
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

                # acc_data = acc_data + coef * data
                y_data = coef**coef_power * data - c_data
                t_data = acc_data + y_data
                c_data = (t_data - acc_data) - y_data
                acc_data = t_data

                # acc_count = acc_count + coef
                y_count = coef - c_count
                t_count = acc_count + y_count
                c_count = (t_count - acc_count) - y_count
                acc_count = t_count

            sum_data[i] += acc_data
            sum_count[i] += acc_count
            if acc_count > epsilon:
                merged[i] += <float> (acc_data / acc_count / normalization_factor)
            else:
                merged[i] += cdummy

        # Ugly against bug#89
        if need_decref and (sys.getrefcount(self._lut) >= rc_before + 2):
            logger.warning("Decref needed")
            Py_XDECREF(<PyObject *> self._lut)

        return (self.bin_centers, numpy.asarray(merged), 
                numpy.asarray(sum_data), numpy.asarray(sum_count))

    @property
    @deprecated(replacement="bin_centers", since_version="0.16", only_once=True)
    def outPos(self):
        return self.bin_centers


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
        :param pos0: 1D array with pos0: tth or q_vect
        :param delta_pos0: 1D array with delta pos0: max center-corner distance
        :param pos1: 1D array with pos1: chi
        :param delta_pos1: 1D array with max pos1: max center-corner distance, unused !
        :param bins: number of output bins (tth=100, chi=36 by default)
        :param pos0Range: minimum and maximum  of the 2th range
        :param pos1Range: minimum and maximum  of the chi range
        :param mask: array (of int8) with masked pixels with 1 (0=not masked)
        :param allow_pos0_neg: enforce the q<0 is usually not possible
        :param chiDiscAtPi: boolean; by default the chi_range is in the range ]-pi,pi[ set to 0 to have the range ]0,2pi[
        :param unit: can be 2th_deg or r_nm^-1 ...
        """
        cdef cnumpy.int32_t i, size, bin0, bin1
        self.size = pos0.size
        assert delta_pos0.size == self.size, "delta_pos0.size == self.size"
        assert pos1.size == self.size, "pos1 size"
        assert delta_pos1.size == self.size, "delta_pos1.size == self.size"
        self.chiDiscAtPi = 1 if chiDiscAtPi else 0
        self.allow_pos0_neg = allow_pos0_neg

        try:
            bins0, bins1 = tuple(bins)
        except TypeError:
            bins0 = bins1 = bins
        if bins0 <= 0:
            bins0 = 1
        if bins1 <= 0:
            bins1 = 1
        self.bins = (int(bins0), int(bins1))
        self.lut_size = 0
        if mask is not None:
            assert mask.size == self.size, "mask size"
            self.check_mask = True
            self.cmask = numpy.ascontiguousarray(mask.ravel(), dtype=numpy.int8)
            if mask_checksum:
                self.mask_checksum = mask_checksum
            else:
                self.mask_checksum = crc32(mask)
        else:
            self.check_mask = False
            self.mask_checksum = None

        self.cpos0 = numpy.ascontiguousarray(pos0.ravel(), dtype=position_d)
        self.dpos0 = numpy.ascontiguousarray(delta_pos0.ravel(), dtype=position_d)
        self.cpos0_sup = numpy.empty_like(self.cpos0)
        self.cpos0_inf = numpy.empty_like(self.cpos0)
        self.pos0Range = pos0Range
        self.pos1Range = pos1Range

        self.cpos1 = numpy.ascontiguousarray((pos1).ravel(), dtype=position_d)
        self.dpos1 = numpy.ascontiguousarray((delta_pos1).ravel(), dtype=position_d)
        self.cpos1_sup = numpy.empty_like(self.cpos1)
        self.cpos1_inf = numpy.empty_like(self.cpos1)
        self.calc_boundaries(pos0Range, pos1Range)
        self.delta0 = (self.pos0_max - self.pos0_min) / float(bins0)
        self.delta1 = (self.pos1_max - self.pos1_min) / float(bins1)
        self.lut_max_idx = None
        self._lut = None
        self.calc_lut()
        self.bin_centers0 = numpy.linspace(self.pos0_min + 0.5 * self.delta0, 
                                           self.pos0_max - 0.5 * self.delta0, 
                                           bins0)
        self.bin_centers1 = numpy.linspace(self.pos1_min + 0.5 * self.delta1, 
                                           self.pos1_max - 0.5 * self.delta1, 
                                           bins1)
        self.unit = unit
        # Calculated at export time to python
        self._lut_checksum = None

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def calc_boundaries(self, pos0Range, pos1Range):
        """
        Called by constructor to calculate the boundaries and the bin position
        """
        cdef:
            cnumpy.int32_t size = self.cpos0.size
            bint check_mask = self.check_mask
            mask_t[::1] cmask
            position_t[::1] cpos0, dpos0, cpos0_sup, cpos0_inf
            position_t[::1] cpos1, dpos1, cpos1_sup, cpos1_inf,
            position_t upper0, lower0, pos0_max, pos0_min, c0, d0
            position_t upper1, lower1, pos1_max, pos1_min, c1, d1
            bint allow_pos0_neg = self.allow_pos0_neg
            bint chiDiscAtPi = self.chiDiscAtPi

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
                if not allow_pos0_neg and lower0 < 0:
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
        self.pos0_max = calc_upper_bound(<double> self.pos0_maxin)
        self.cpos0_sup = cpos0_sup
        self.cpos0_inf = cpos0_inf
        self.pos1_max = calc_upper_bound(<double> self.pos1_maxin)
        self.cpos1_sup = cpos1_sup
        self.cpos1_inf = cpos1_inf

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def calc_lut(self):
        'calculate the max number of elements in the LUT and populate it'
        cdef:
            position_t delta0 = self.delta0, pos0_min = self.pos0_min, min0, max0, fbin0_min, fbin0_max
            position_t delta1 = self.delta1, pos1_min = self.pos1_min, min1, max1, fbin1_min, fbin1_max
            int bin0_min, bin0_max, bins0 = self.bins[0]
            int bin1_min, bin1_max, bins1 = self.bins[1]
            int k, idx, lut_size, i, j, size = self.size
            bint check_mask
            position_t[::1] cpos0_sup = self.cpos0_sup
            position_t[::1] cpos0_inf = self.cpos0_inf
            position_t[::1] cpos1_inf = self.cpos1_inf
            position_t[::1] cpos1_sup = self.cpos1_sup
            cnumpy.int32_t[:, ::1] outmax = numpy.zeros((bins0, bins1), dtype=numpy.int32)
            lut_t[:, :, ::1] lut
            mask_t[:] cmask
            acc_t inv_area, delta_down, delta_up, delta_right, delta_left
        if self.check_mask:
            cmask = self.cmask
            check_mask = True
        else:
            check_mask = False

        with nogil:
            for idx in range(size):
                if (check_mask) and (cmask[idx]):
                    continue

                min0 = cpos0_inf[idx]
                max0 = cpos0_sup[idx]
                min1 = cpos1_inf[idx]
                max1 = cpos1_sup[idx]

                bin0_min = <int> get_bin_number(min0, pos0_min, delta0)
                bin0_max = <int> get_bin_number(max0, pos0_min, delta0)

                bin1_min = <int> get_bin_number(min1, pos1_min, delta1)
                bin1_max = <int> get_bin_number(max1, pos1_min, delta1)

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

                for i in range(bin0_min, bin0_max + 1):
                    for j in range(bin1_min, bin1_max + 1):
                        outmax[i, j] += 1

        self.lut_size = lut_size = numpy.max(outmax)
        # just recycle the outmax array
        outmax[:, :] = 0

        lut_nbytes = bins0 * bins1 * lut_size * sizeof(lut_t)
        #Check we have enough memory
        if (os.name == "posix"):
            key_page_size = os.sysconf_names.get("SC_PAGE_SIZE", 0)
            key_page_cnt = os.sysconf_names.get("SC_PHYS_PAGES",0)
            if key_page_size*key_page_cnt:
                try:
                    memsize = os.sysconf(key_page_size) * os.sysconf(key_page_cnt)
                except OSError:
                    pass
                else:
                    if memsize < lut_nbytes:
                        raise MemoryError("Lookup-table (%i, %i, %i) is %.3fGB whereas the memory of the system is only %s" %
                                          (bins0, bins1, lut_size, lut_nbytes, memsize))

        # else hope we have enough memory
        lut = view.array(shape=(bins0, bins1, lut_size), itemsize=sizeof(lut_t), format="if")
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
                        k = outmax[bin0_min, bin1_min]
                        lut[bin0_min, bin1_min, k].idx = idx
                        lut[bin0_min, bin1_min, k].coef = 1.0
                        outmax[bin0_min, bin1_min] = k + 1

                    else:
                        # spread on more than 2 bins
                        delta_down = (<acc_t> (bin1_min + 1)) - fbin1_min
                        delta_up = fbin1_max - <acc_t> bin1_max
                        inv_area = 1.0 / (fbin1_max - fbin1_min)

                        k = outmax[bin0_min, bin1_min]
                        lut[bin0_min, bin1_min, k].idx = idx
                        lut[bin0_min, bin1_min, k].coef = inv_area * delta_down
                        outmax[bin0_min, bin1_min] += 1

                        k = outmax[bin0_min, bin1_max]
                        lut[bin0_min, bin1_max, k].idx = idx
                        lut[bin0_min, bin1_max, k].coef = inv_area * delta_up
                        outmax[bin0_min, bin1_max] += 1

                        for j in range(bin1_min + 1, bin1_max):
                            k = outmax[bin0_min, j]
                            lut[bin0_min, j, k].idx = idx
                            lut[bin0_min, j, k].coef = inv_area
                            outmax[bin0_min, j] += 1

                else:
                    # spread on more than 2 bins in dim 0
                    if bin1_min == bin1_max:
                        # All pixel fall on 1 bins in dim 1
                        inv_area = 1.0 / (fbin0_max - fbin0_min)
                        delta_left = (<acc_t> (bin0_min + 1)) - fbin0_min

                        k = outmax[bin0_min, bin1_min]
                        lut[bin0_min, bin1_min, k].idx = idx
                        lut[bin0_min, bin1_min, k].coef = inv_area * delta_left
                        outmax[bin0_min, bin1_min] = k + 1

                        delta_right = fbin0_max - (<acc_t> bin0_max)

                        k = outmax[bin0_max, bin1_min]
                        lut[bin0_max, bin1_min, k].idx = idx
                        lut[bin0_max, bin1_min, k].coef = inv_area * delta_right
                        outmax[bin0_max, bin1_min] += 1

                        for i in range(bin0_min + 1, bin0_max):
                            k = outmax[i, bin1_min]
                            lut[i, bin1_min, k].idx = idx
                            lut[i, bin1_min, k].coef = inv_area
                            outmax[i, bin1_min] += 1

                    else:
                        # spread on n pix in dim0 and m pixel in dim1:
                        delta_left = (<acc_t> (bin0_min + 1)) - fbin0_min
                        delta_right = fbin0_max - (<acc_t> bin0_max)
                        delta_down = (<acc_t> (bin1_min + 1)) - fbin1_min
                        delta_up = fbin1_max - (<acc_t> bin1_max)
                        inv_area = 1.0 / ((fbin0_max - fbin0_min) * (fbin1_max - fbin1_min))

                        k = outmax[bin0_min, bin1_min]
                        lut[bin0_min, bin1_min, k].idx = idx
                        lut[bin0_min, bin1_min, k].coef = inv_area * delta_left * delta_down
                        outmax[bin0_min, bin1_min] += 1

                        k = outmax[bin0_min, bin1_max]
                        lut[bin0_min, bin1_max, k].idx = idx
                        lut[bin0_min, bin1_max, k].coef = inv_area * delta_left * delta_up
                        outmax[bin0_min, bin1_max] += 1

                        k = outmax[bin0_max, bin1_min]
                        lut[bin0_max, bin1_min, k].idx = idx
                        lut[bin0_max, bin1_min, k].coef = inv_area * delta_right * delta_down
                        outmax[bin0_max, bin1_min] += 1

                        k = outmax[bin0_max, bin1_max]
                        lut[bin0_max, bin1_max, k].idx = idx
                        lut[bin0_max, bin1_max, k].coef = inv_area * delta_right * delta_up
                        outmax[bin0_max, bin1_max] += 1

                        for i in range(bin0_min + 1, bin0_max):
                            k = outmax[i, bin1_min]
                            lut[i, bin1_min, k].idx = idx
                            lut[i, bin1_min, k].coef = inv_area * delta_down
                            outmax[i, bin1_min] += 1

                            for j in range(bin1_min + 1, bin1_max):
                                k = outmax[i, j]
                                lut[i, j, k].idx = idx
                                lut[i, j, k].coef = inv_area
                                outmax[i, j] += 1

                            k = outmax[i, bin1_max]
                            lut[i, bin1_max, k].idx = idx
                            lut[i, bin1_max, k].coef = inv_area * delta_up
                            outmax[i, bin1_max] += 1

                        for j in range(bin1_min + 1, bin1_max):
                            k = outmax[bin0_min, j]
                            lut[bin0_min, j, k].idx = idx
                            lut[bin0_min, j, k].coef = inv_area * delta_left
                            outmax[bin0_min, j] += 1

                            k = outmax[bin0_max, j]
                            lut[bin0_max, j, k].idx = idx
                            lut[bin0_max, j, k].coef = inv_area * delta_right
                            outmax[bin0_max, j] += 1

        self.lut_max_idx = outmax
        self._lut = lut

    @property
    def lut(self):
        """Getter for the LUT as actual numpy array
        Hack against a bug in ref-counting under python2.6
        """
        cdef int rc_before, rc_after
        rc_before = sys.getrefcount(self._lut)
        cdef lut_t[:, :, :] lut = self._lut
        rc_after = sys.getrefcount(self._lut)
        cdef bint need_decref = NEED_DECREF and ((rc_after - rc_before) >= 2)
        shape = (self._lut.shape[0] * self._lut.shape[1], self._lut.shape[2])
        cdef numpy.ndarray[numpy.float64_t, ndim=2] tmp_ary = numpy.empty(shape=shape, dtype=numpy.float64)
        memcpy(&tmp_ary[0, 0], &lut[0, 0, 0], self._lut.nbytes)
        self._lut_checksum = crc32(tmp_ary)

        # Ugly against bug#89
        if need_decref and (sys.getrefcount(self._lut) >= rc_before + 2):
            print("Warning: Decref needed")
            Py_XDECREF(<PyObject *> self._lut)

        return numpy.core.records.array(tmp_ary.view(dtype=lut_d),
                                        shape=shape, dtype=lut_d,
                                        copy=True)

    @property
    def lut_checksum(self):
        if self._lut_checksum is None:
            self.lut
        return self._lut_checksum

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
                  double normalization_factor=1.0,
                  int coef_power=1):
        """
        Actually perform the 2D integration which in this case looks more like a matrix-vector product

        :param weights: input image
        :type weights: ndarray
        :param dummy: value for dead pixels (optional)
        :type dummy: float
        :param delta_dummy: precision for dead-pixel value in dynamic masking
        :type delta_dummy: float
        :param dark: array with the dark-current value to be subtracted (if any)
        :type dark: ndarray
        :param flat: array with the dark-current value to be divided by (if any)
        :type flat: ndarray
        :param solidAngle: array with the solid angle of each pixel to be divided by (if any)
        :type solidAngle: ndarray
        :param polarization: array with the polarization correction values to be divided by (if any)
        :type polarization: ndarray
        :param normalization_factor: divide the valid result by this value
        :param coef_power: set to 1 for mean en to 2 for variance propagation
        :return:  I(2d), edges0(1d), edges1(1d), weighted histogram(2d), unweighted histogram (2d)
        :rtype: 5-tuple of ndarrays

        """
        cdef:
            cnumpy.int32_t i = 0, j = 0, idx = 0, bins0 = self.bins[0], bins1 = self.bins[1], bins = bins0 * bins1, lut_size = self.lut_size, size = self.size, i0 = 0, i1 = 0
            acc_t acc_data = 0, acc_count = 0, epsilon = 1e-10
            data_t data = 0, coef = 0, cdummy = 0, cddummy = 0
            bint do_dummy = False, do_dark = False, do_flat = False, do_polarization = False, do_solidAngle = False
            acc_t[:, ::1] sum_data = numpy.zeros(self.bins, dtype=numpy.float64)
            acc_t[:, ::1] sum_count = numpy.zeros(self.bins, dtype=numpy.float64)
            data_t[:, ::1] merged = numpy.zeros(self.bins, dtype=numpy.float32)
            data_t[::1] cdata, tdata, cflat, cdark, csolidAngle, cpolarization
        # Ugly hack against bug #89
            int rc_before, rc_after
        rc_before = sys.getrefcount(self._lut)
        cdef lut_t[:, :, ::1] lut = self._lut
        rc_after = sys.getrefcount(self._lut)
        cdef bint need_decref = NEED_DECREF and ((rc_after - rc_before) >= 2)

        assert weights.size == size, "weights size"

        if dummy is not None:
            do_dummy = True
            cdummy = <data_t> float(dummy)
            if delta_dummy is None:
                cddummy = zerof
            else:
                cddummy = <data_t> float(delta_dummy)

        if flat is not None:
            do_flat = True
            assert flat.size == size, "flat-field array size"
            cflat = numpy.ascontiguousarray(flat.ravel(), dtype=data_d)
        if dark is not None:
            do_dark = True
            assert dark.size == size, "dark current array size"
            cdark = numpy.ascontiguousarray(dark.ravel(), dtype=data_d)
        if solidAngle is not None:
            do_solidAngle = True
            assert solidAngle.size == size, "Solid angle array size"
            csolidAngle = numpy.ascontiguousarray(solidAngle.ravel(), dtype=data_d)
        if polarization is not None:
            do_polarization = True
            assert polarization.size == size, "polarization array size"
            cpolarization = numpy.ascontiguousarray(polarization.ravel(), dtype=data_d)

        if (do_dark + do_flat + do_polarization + do_solidAngle):
            tdata = numpy.ascontiguousarray(weights.ravel(), dtype=data_d)
            cdata = numpy.zeros(size, dtype=data_d)
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
                cdata = numpy.ascontiguousarray(weights.ravel(), dtype=data_d)

        for i0 in prange(bins0, nogil=True, schedule="guided"):
            for i1 in range(bins1):
                acc_data = 0.0
                acc_count = 0.0
                for j in range(lut_size):
                    idx = lut[i0, i1, j].idx
                    coef = lut[i0, i1, j].coef
                    if idx <= 0 and coef <= 0.0:
                        continue
                    data = cdata[idx]
                    if do_dummy and data == cdummy:
                        continue

                    acc_data = acc_data + coef ** coef_power * data
                    acc_count = acc_count + coef
                sum_data[i0, i1] += acc_data
                sum_count[i0, i1] += acc_count
                if acc_count > epsilon:
                    merged[i0, i1] += <data_t> (acc_data / acc_count / normalization_factor)
                else:
                    merged[i0, i1] += cdummy

        # Ugly against bug #89
        if need_decref and (sys.getrefcount(self._lut) >= rc_before + 2):
            Py_XDECREF(<PyObject *> self._lut)

        return (numpy.asarray(merged).T, 
                self.bin_centers0, self.bin_centers1, 
                numpy.asarray(sum_data).T, 
                numpy.asarray(sum_count).T)

    @property
    @deprecated(replacement="bin_centers0", since_version="0.16", only_once=True)
    def outPos0(self):
        return self.bin_centers0

    @property
    @deprecated(replacement="bin_centers1", since_version="0.16", only_once=True)
    def outPos1(self):
        return self.bin_centers1
