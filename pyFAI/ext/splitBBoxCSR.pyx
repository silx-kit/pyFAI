# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal Integration
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
__date__ = "15/11/2018"
__status__ = "stable"
__license__ = "MIT"

include "regrid_common.pxi"

import cython
import os
import sys
import logging
logger = logging.getLogger(__name__)
from cython.parallel import prange
import numpy
from ..utils import crc32
from ..utils.decorators import deprecated


class HistoBBox1d(object):
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
                 empty=0.0
                 ):
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
        :param empty: value to be assigned to bins without contribution from any pixel

        """
        self.size = pos0.size
        if "size" not in dir(delta_pos0) or delta_pos0.size != self.size:
            logger.warning("Pixel splitting desactivated !")
            delta_pos0 = None
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
        self.data = self.nnz = self.indices = self.indptr = None
        self.pos0Range = pos0Range
        self.pos1Range = pos1Range
        self.cpos0 = numpy.ascontiguousarray(pos0.ravel(), dtype=position_d)
        if delta_pos0 is None:
            self.calc_boundaries_nosplit(pos0Range)
        else:
            self.dpos0 = numpy.ascontiguousarray(delta_pos0.ravel(), dtype=position_d)
            self.cpos0_sup = numpy.empty_like(self.cpos0)  # self.cpos0 + self.dpos0
            self.cpos0_inf = numpy.empty_like(self.cpos0)  # self.cpos0 - self.dpos0
            self.calc_boundaries(pos0Range)

        if pos1Range is not None and len(pos1Range) > 1:
            assert pos1.size == self.size, "pos1 size"
            assert delta_pos1.size == self.size, "delta_pos1.size == self.size"
            self.check_pos1 = True
            self.cpos1_min = numpy.ascontiguousarray((pos1 - delta_pos1).ravel(), dtype=position_d)
            self.cpos1_max = numpy.ascontiguousarray((pos1 + delta_pos1).ravel(), dtype=position_d)
            self.pos1_min = min(pos1Range)
            pos1_maxin = max(pos1Range)
            self.pos1_max = calc_upper_bound(<double> pos1_maxin)
        else:
            self.check_pos1 = False
            self.cpos1_min = None
            self.pos1_max = None

        self.delta = (self.pos0_max - self.pos0_min) / (<position_t> bins)
        if delta_pos0 is not None:
            self.calc_lut()
        else:
            self.calc_lut_nosplit()

        self.bin_centers = numpy.linspace(self.pos0_min + 0.5 * self.delta,
                                          self.pos0_max - 0.5 * self.delta,
                                          self.bins)
        self.lut_checksum = crc32(self.data)
        self.unit = unit
        self.lut = (self.data, self.indices, self.indptr)
        self.lut_nbytes = sum([i.nbytes for i in self.lut])      

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def calc_boundaries(self, pos0Range):
        """
        Calculate self.pos0_min and self.pos0_max

        :param pos0Range: 2-tuple containing the requested range
        """
        cdef:
            int size = self.cpos0.size
            bint check_mask = self.check_mask
            mask_t[::1] cmask
            double[::1] cpos0, dpos0, cpos0_sup, cpos0_inf,
            double upper, lower, pos0_max, pos0_min, c, d
            bint allow_pos0_neg = self.allow_pos0_neg
            int idx

        cpos0_sup = self.cpos0_sup
        cpos0_inf = self.cpos0_inf
        cpos0 = self.cpos0
        dpos0 = self.dpos0
        pos0_min = pos0_max = cpos0[0]
        if not allow_pos0_neg and pos0_min < 0:
                    pos0_min = pos0_max = 0
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
        self.pos0_max = calc_upper_bound(<position_t> self.pos0_maxin)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def calc_boundaries_nosplit(self, pos0Range):
        """
        Calculate self.pos0_min and self.pos0_max when no splitting is requested

        :param pos0Range: 2-tuple containing the requested range
        """
        cdef:
            int size = self.cpos0.size
            bint check_mask = self.check_mask
            mask_t[::1] cmask
            position_t[::1] cpos0
            position_t upper, lower, pos0_max, pos0_min, c, d
            bint allow_pos0_neg = self.allow_pos0_neg
            int idx

        if pos0Range is not None and len(pos0Range) > 1:
            self.pos0_min = min(pos0Range)
            self.pos0_maxin = max(pos0Range)
        else:
            cpos0 = self.cpos0
            pos0_min = pos0_max = cpos0[0]

            if not allow_pos0_neg and pos0_min < 0:
                pos0_min = pos0_max = 0

            if check_mask:
                cmask = self.cmask

            with nogil:
                for idx in range(size):
                    c = cpos0[idx]
                    if not allow_pos0_neg and c < 0:
                        c = 0
                    if not (check_mask and cmask[idx]):
                        if c > pos0_max:
                            pos0_max = c
                        if c < pos0_min:
                            pos0_min = c
            self.pos0_min = pos0_min
            self.pos0_maxin = pos0_max

        if (not allow_pos0_neg) and self.pos0_min < 0:
            self.pos0_min = 0
        self.pos0_max = calc_upper_bound(<position_t> self.pos0_maxin)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def calc_lut(self):
        '''
        calculate the max number of elements in the LUT and populate it
        '''
        cdef:
            position_t delta = self.delta, pos0_min = self.pos0_min, pos1_min, pos1_max, 
            position_t min0, max0, fbin0_min, fbin0_max
            cnumpy.int32_t k, idx, i, j, tmp_index, index_tmp_index, bin0_min, bin0_max, bins = self.bins, size, nnz
            bint check_mask, check_pos1
            cnumpy.int32_t[::1] outmax = numpy.zeros(bins, dtype=numpy.int32)
            cnumpy.int32_t[::1] indptr, indices
            float[::1] data
            position_t[::1] cpos0_sup = self.cpos0_sup, cpos0_inf = self.cpos0_inf, cpos1_min, cpos1_max,
            mask_t[::1] cmask
            acc_t inv_area, delta_left, delta_right

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
                bin0_min = < int > fbin0_min
                bin0_max = < int > fbin0_max

                if (bin0_max < 0) or (bin0_min >= bins):
                    continue
                if bin0_max >= bins:
                    bin0_max = bins - 1
                if bin0_min < 0:
                    bin0_min = 0

                if bin0_min == bin0_max:
                    #  All pixel is within a single bin
                    outmax[bin0_min] += 1

                else:  # We have pixel splitting.
                    for i in range(bin0_min, bin0_max + 1):
                        outmax[i] += 1

        indptr = numpy.concatenate(([numpy.int32(0)], 
                                    numpy.asarray(outmax).cumsum(dtype=numpy.int32)))
        self.indptr = numpy.asarray(indptr)
        self.nnz = nnz = indptr[bins]

        # just recycle the outmax array
        outmax[:] = 0

        lut_nbytes = nnz * (sizeof(cnumpy.int32_t) + sizeof(cnumpy.float32_t))
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
                        raise MemoryError("CSR Lookup-table (%i, %i) is %.3fGB whereas the memory of the system is only %.3fGB" %
                                          (bins, self.nnz, lut_nbytes / 2. ** 30, memsize / 2. ** 30))


        # else hope we have enough memory

        data = numpy.empty(nnz, dtype=numpy.float32)
        indices = numpy.empty(nnz, dtype=numpy.int32)

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
                bin0_min = < int > fbin0_min
                bin0_max = < int > fbin0_max

                if (bin0_max < 0) or (bin0_min >= bins):
                    continue
                if bin0_max >= bins:
                    bin0_max = bins - 1
                if bin0_min < 0:
                    bin0_min = 0

                if bin0_min == bin0_max:
                    # All pixel is within a single bin
                    k = outmax[bin0_min]
                    indices[indptr[bin0_min] + k] = idx
                    data[indptr[bin0_min] + k] = onef
                    outmax[bin0_min] += 1  # k+1
                else:  # we have pixel splitting.
                    inv_area = 1.0 / (fbin0_max - fbin0_min)

                    delta_left = <position_t> (bin0_min + 1) - fbin0_min
                    delta_right = fbin0_max - <position_t> (bin0_max)

                    k = outmax[bin0_min]
                    indices[indptr[bin0_min] + k] = idx
                    data[indptr[bin0_min] + k] = (inv_area * delta_left)
                    outmax[bin0_min] += 1

                    k = outmax[bin0_max]
                    indices[indptr[bin0_max] + k] = idx
                    data[indptr[bin0_max] + k] = (inv_area * delta_right)
                    outmax[bin0_max] += 1

                    if bin0_min + 1 < bin0_max:
                        for i in range(bin0_min + 1, bin0_max):
                            k = outmax[i]
                            indices[indptr[i] + k] = idx
                            data[indptr[i] + k] = (inv_area)
                            outmax[i] += 1

        self.data = numpy.asarray(data)
        self.indices = numpy.asarray(indices)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def calc_lut_nosplit(self):
        '''
        calculate the max number of elements in the LUT and populate it
        '''
        cdef:
            position_t delta = self.delta, pos0_min = self.pos0_min, pos1_min, pos1_max, fbin0, pos0
            cnumpy.int32_t k, idx, i, j, tmp_index, index_tmp_index, bin0, bins = self.bins, size, nnz
            bint check_mask, check_pos1
            cnumpy.int32_t[::1] outmax = numpy.zeros(bins, dtype=numpy.int32)
            cnumpy.int32_t[::1] indptr, indices
            float[::1] data
            position_t[::1] cpos0 = self.cpos0, cpos1_min, cpos1_max,
            mask_t[::1] cmask

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

                pos0 = cpos0[idx]

                if check_pos1 and ((cpos1_max[idx] < pos1_min) or (cpos1_min[idx] > pos1_max)):
                    continue

                fbin0 = get_bin_number(pos0, pos0_min, delta)
                bin0 = < int > fbin0

                if (bin0 >= 0) and (bin0 < bins):
                    outmax[bin0] += 1

        indptr = numpy.concatenate((numpy.zeros(1, dtype=numpy.int32), numpy.asarray(outmax).cumsum(dtype=numpy.int32)))
        self.indptr = numpy.asarray(indptr)
        self.nnz = nnz = indptr[bins]

        # just recycle the outmax array
        outmax[:] = 0

        lut_nbytes = nnz * (sizeof(cnumpy.int32_t) + sizeof(cnumpy.float32_t))
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
                        raise MemoryError("CSR Lookup-table (%i, %i) is %.3fGB whereas the memory of the system is only %.3fGB" %
                                          (bins, self.nnz, lut_nbytes / 2. ** 30, memsize / 2. ** 30))
        # else hope we have enough memory

        data = numpy.empty(nnz, dtype=numpy.float32)
        indices = numpy.empty(nnz, dtype=numpy.int32)

        with nogil:
            for idx in range(size):
                if (check_mask) and (cmask[idx]):
                    continue

                pos0 = cpos0[idx]

                if check_pos1 and ((cpos1_max[idx] < pos1_min) or (cpos1_min[idx] > pos1_max)):
                    continue

                fbin0 = get_bin_number(pos0, pos0_min, delta)
                bin0 = < int > fbin0

                if (bin0 < 0) or (bin0 >= bins):
                    continue
                k = outmax[bin0]
                indices[indptr[bin0] + k] = idx
                data[indptr[bin0] + k] = onef
                outmax[bin0] += 1  # k+1

        self.data = numpy.asarray(data)
        self.indices = numpy.asarray(indices)

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
        :param coef_power: set to 2 for variance propagation, leave to 1 for mean calculation

        :return: positions, pattern, weighted_histogram and unweighted_histogram
        :rtype: 4-tuple of ndarrays

        """
        cdef:
            cnumpy.int32_t i = 0, j = 0, idx = 0, bins = self.bins, size = self.size
            acc_t acc_data = 0.0, acc_count = 0.0, epsilon = 1e-10, coef = 0.0
            data_t data = 0.0, cdummy = 0.0, cddummy = 0.0
            bint do_dummy = False, do_dark = False, do_flat = False, do_polarization = False, do_solidAngle = False
            acc_t[::1] sum_data = numpy.zeros(self.bins, dtype=acc_d)
            acc_t[::1] sum_count = numpy.zeros(self.bins, dtype=acc_d)
            data_t[::1] merged = numpy.zeros(self.bins, dtype=data_d)
            data_t[::1] ccoef = self.data
            data_t[::1] cdata, tdata, cflat, cdark, csolidAngle, cpolarization
            cnumpy.int32_t[::1] indices = self.indices, indptr = self.indptr
        assert weights.size == size, "weights size"

        if dummy is not None:
            do_dummy = True
            cdummy = <data_t> float(dummy)

            if delta_dummy is None:
                cddummy = <data_t> 0.0
            else:
                cddummy = <data_t> float(delta_dummy)
        else:
            do_dummy = False
            cdummy = <data_t> self.empty

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
                tdata = numpy.ascontiguousarray(weights.ravel(), dtype=data_d)
                cdata = numpy.zeros(size, dtype=data_d)
                for i in prange(size, nogil=True, schedule="static"):
                    data = tdata[i]
                    if ((cddummy != 0) and (fabs(data - cdummy) > cddummy)) or ((cddummy == 0) and (data != cdummy)):
                        cdata[i] += data
                    else:
                        cdata[i] += cdummy
            else:
                cdata = numpy.ascontiguousarray(weights.ravel(), dtype=data_d)

        for i in prange(bins, nogil=True, schedule="guided"):
            acc_data = 0.0
            acc_count = 0.0
            for j in range(indptr[i], indptr[i + 1]):
                idx = indices[j]
                coef = ccoef[j]
                if coef == 0.0:
                    continue
                data = cdata[idx]
                if do_dummy and (data == cdummy):
                    continue
                acc_data = acc_data + (coef ** coef_power) * data
                acc_count = acc_count + coef

            sum_data[i] += acc_data
            sum_count[i] += acc_count
            if acc_count > epsilon:
                merged[i] += acc_data / acc_count / normalization_factor
            else:
                merged[i] += cdummy
        return (self.bin_centers, 
                numpy.asarray(merged), 
                numpy.asarray(sum_data), 
                numpy.asarray(sum_count))

    @property
    @deprecated(replacement="bin_centers", since_version="0.16", only_once=True)
    def outPos(self):
        return self.bin_centers


################################################################################
# Bidimensionnal regrouping
################################################################################


class HistoBBox2d(object):
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
                 chiDiscAtPi=True,
                 empty=0.0
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
        """
        cdef: 
            int i, size, bin0, bin1
        self.size = pos0.size
        assert pos1.size == self.size, "pos1 size"

        # Declare a few variables
        self.pos0_min = self.pos0_maxin = self.pos0_max = self.delta0 = None
        self.pos1_min = self.pos1_maxin = self.pos1_max = self.delta1 = None

        if "size" not in dir(delta_pos0) or delta_pos0.size != self.size or\
                "size" not in dir(delta_pos1) or delta_pos1.size != self.size:
            logger.warning("Pixel splitting deactivated !")
            delta_pos0 = None
            delta_pos1 = None

        self.chiDiscAtPi = 1 if chiDiscAtPi else 0
        self.allow_pos0_neg = allow_pos0_neg
        self.empty = 0.0
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

        self.data = self.nnz = self.indices = self.indptr = None
        self.pos0Range = pos0Range
        self.pos1Range = pos1Range

        self.cpos0 = numpy.ascontiguousarray(pos0.ravel(), dtype=position_d)
        self.cpos1 = numpy.ascontiguousarray((pos1).ravel(), dtype=position_d)
        if delta_pos0 is not None:
            self.dpos0 = numpy.ascontiguousarray(delta_pos0.ravel(), dtype=position_d)
            self.cpos0_sup = numpy.empty_like(self.cpos0)  # self.cpos0 + self.dpos0
            self.cpos0_inf = numpy.empty_like(self.cpos0)  # self.cpos0 - self.dpos0
            self.dpos1 = numpy.ascontiguousarray((delta_pos1).ravel(), dtype=position_d)
            self.cpos1_sup = numpy.empty_like(self.cpos1)  # self.cpos1 + self.dpos1
            self.cpos1_inf = numpy.empty_like(self.cpos1)  # self.cpos1 - self.dpos1
            self.calc_boundaries(pos0Range, pos1Range)
        else:
            self.calc_boundaries_nosplit(pos0Range, pos1Range)

        self.delta0 = (self.pos0_max - self.pos0_min) / float(bins0)
        self.delta1 = (self.pos1_max - self.pos1_min) / float(bins1)

        if delta_pos0 is not None:
            self.calc_lut()
        else:
            self.calc_lut_nosplit()

        self.bin_centers0 = numpy.linspace(self.pos0_min + 0.5 * self.delta0, 
                                           self.pos0_max - 0.5 * self.delta0, 
                                           bins0)
        self.bin_centers1 = numpy.linspace(self.pos1_min + 0.5 * self.delta1, 
                                           self.pos1_max - 0.5 * self.delta1, 
                                           bins1)
        self.unit = unit
        self.lut = (self.data, self.indices, self.indptr)
        self.lut_checksum = crc32(self.data)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def calc_boundaries(self, pos0Range, pos1Range):
        """
        Calculate self.pos0_min/max and self.pos1_min/max

        :param pos0Range: 2-tuple containing the requested range
        :param pos1Range: 2-tuple containing the requested range
        """
        cdef:
            int size = self.cpos0.size
            bint check_mask = self.check_mask
            mask_t[::1] cmask
            position_t[::1] cpos0, dpos0, cpos0_sup, cpos0_inf
            position_t[::1] cpos1, dpos1, cpos1_sup, cpos1_inf
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
        pos0_min = pos0_max = cpos0[0]
        pos1_min = pos1_max = cpos1[0]
        if not allow_pos0_neg and pos0_min < 0:
            pos0_min = pos0_max = 0
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

        if pos1Range is not None and len(pos1Range) > 1:
            self.pos1_min = min(pos1Range)
            self.pos1_maxin = max(pos1Range)
        else:
            self.pos1_min = pos1_min
            self.pos1_maxin = pos1_max

        if (not allow_pos0_neg) and self.pos0_min < 0:
            self.pos0_min = 0
        self.pos0_max = calc_upper_bound(<position_t> self.pos0_maxin)
        self.cpos0_sup = cpos0_sup
        self.cpos0_inf = cpos0_inf
        self.pos1_max = calc_upper_bound(<position_t> self.pos1_maxin)
        self.cpos1_sup = cpos1_sup
        self.cpos1_inf = cpos1_inf

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def calc_boundaries_nosplit(self, pos0Range, pos1Range):
        """
        Calculate self.pos0_min/max and self.pos1_min/max

        :param pos0Range: 2-tuple containing the requested range
        :param pos1Range: 2-tuple containing the requested range
        """
        cdef:
            int size = self.cpos0.size
            bint check_mask = self.check_mask
            mask_t[::1] cmask
            double[::1] cpos0
            double[::1] cpos1
            double upper0, lower0, pos0_max, pos0_min, c0, d0
            double upper1, lower1, pos1_max, pos1_min, c1, d1
            bint allow_pos0_neg = self.allow_pos0_neg
            bint chiDiscAtPi = self.chiDiscAtPi

        cpos0 = self.cpos0
        cpos1 = self.cpos1
        pos0_min = pos0_max = cpos0[0]
        pos1_min = pos1_max = cpos1[0]
        if not allow_pos0_neg and pos0_min < 0:
            pos0_min = pos0_max = 0
        if check_mask:
            cmask = self.cmask
        with nogil:
            for idx in range(size):
                c0 = cpos0[idx]
                c1 = cpos1[idx]
                if not allow_pos0_neg and c0 < 0:
                    c0 = 0
                if c1 > (2 - chiDiscAtPi) * pi:
                    c1 = (2 - chiDiscAtPi) * pi
                if c1 < (-chiDiscAtPi) * pi:
                    c1 = (-chiDiscAtPi) * pi
                if not (check_mask and cmask[idx]):
                    if c0 > pos0_max:
                        pos0_max = c0
                    if c0 < pos0_min:
                        pos0_min = c0
                    if c1 > pos1_max:
                        pos1_max = c1
                    if c1 < pos1_min:
                        pos1_min = c1

        if pos0Range is not None and len(pos0Range) > 1:
            self.pos0_min = min(pos0Range)
            self.pos0_maxin = max(pos0Range)
        else:
            self.pos0_min = pos0_min
            self.pos0_maxin = pos0_max

        if pos1Range is not None and len(pos1Range) > 1:
            self.pos1_min = min(pos1Range)
            self.pos1_maxin = max(pos1Range)
        else:
            self.pos1_min = pos1_min
            self.pos1_maxin = pos1_max

        if (not allow_pos0_neg) and self.pos0_min < 0:
            self.pos0_min = 0
        self.pos0_max = calc_upper_bound(<position_t> self.pos0_maxin)
        self.pos1_max = calc_upper_bound(<position_t> self.pos1_maxin)

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
            int k, idx, lut_size, i, j, size = self.size, nnz
            bint check_mask
            position_t[::1] cpos0_sup = self.cpos0_sup
            position_t[::1] cpos0_inf = self.cpos0_inf
            position_t[::1] cpos1_inf = self.cpos1_inf
            position_t[::1] cpos1_sup = self.cpos1_sup
            cnumpy.int32_t[:, ::1] outmax = numpy.zeros((bins0, bins1), dtype=numpy.int32)
            cnumpy.int32_t[::1] indptr = numpy.zeros((bins0 * bins1) + 1, dtype=numpy.int32)
            cnumpy.int32_t[::1] indices
            data_t[::1] data
            mask_t[::1] cmask
            acc_t inv_area, delta_left, delta_right, delta_down, delta_up

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

                bin0_min = < int > get_bin_number(min0, pos0_min, delta0)
                bin0_max = < int > get_bin_number(max0, pos0_min, delta0)

                bin1_min = < int > get_bin_number(min1, pos1_min, delta1)
                bin1_max = < int > get_bin_number(max1, pos1_min, delta1)

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

        indptr = numpy.concatenate(([numpy.int32(0)],
                                    numpy.asarray(outmax).ravel().cumsum(dtype=numpy.int32)))
        self.nnz = nnz = indptr[bins0 * bins1]
        self.indptr = numpy.asarray(indptr)
        # Just recycle the outmax array
        outmax[:, :] = 0
        lut_nbytes = nnz * (sizeof(float) + sizeof(cnumpy.int32_t)) + bins0 * bins1 * sizeof(cnumpy.int32_t)
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
                        raise MemoryError("CSR Matrix is %.3fGB whereas the memory of the system is only %s" %
                                          (lut_nbytes/(2.**30), memsize/(2.**30)))
        # else hope we have enough memory

        data = numpy.zeros(nnz, dtype=numpy.float32)
        indices = numpy.zeros(nnz, dtype=numpy.int32)
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

                bin0_min = < int > fbin0_min
                bin0_max = < int > fbin0_max
                bin1_min = < int > fbin1_min
                bin1_max = < int > fbin1_max

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
                        indices[indptr[bin0_min * bins1 + bin1_min] + k] = idx
                        data[indptr[bin0_min * bins1 + bin1_min] + k] = onef
                        outmax[bin0_min, bin1_min] = k + 1

                    else:
                        # spread on more than 2 bins
                        delta_down = (<acc_t> (bin1_min + 1)) - fbin1_min
                        delta_up = fbin1_max - bin1_max
                        inv_area = 1.0 / (fbin1_max - fbin1_min)

                        k = outmax[bin0_min, bin1_min]
                        indices[indptr[bin0_min * bins1 + bin1_min] + k] = idx
                        data[indptr[bin0_min * bins1 + bin1_min] + k] = inv_area * delta_down
                        outmax[bin0_min, bin1_min] = k + 1

                        k = outmax[bin0_min, bin1_max]
                        indices[indptr[bin0_min * bins1 + bin1_max] + k] = idx
                        data[indptr[bin0_min * bins1 + bin1_max] + k] = inv_area * delta_up
                        outmax[bin0_min, bin1_max] = k + 1

                        for j in range(bin1_min + 1, bin1_max):
                            k = outmax[bin0_min, j]
                            indices[indptr[bin0_min * bins1 + j] + k] = idx
                            data[indptr[bin0_min * bins1 + j] + k] = inv_area
                            outmax[bin0_min, j] = k + 1

                else:  # spread on more than 2 bins in dim 0
                    if bin1_min == bin1_max:
                        # All pixel fall on 1 bins in dim 1
                        inv_area = 1.0 / (fbin0_max - fbin0_min)
                        delta_left = (<acc_t> (bin0_min + 1)) - fbin0_min

                        k = outmax[bin0_min, bin1_min]
                        indices[indptr[bin0_min * bins1 + bin1_min] + k] = idx
                        data[indptr[bin0_min * bins1 + bin1_min] + k] = inv_area * delta_left
                        outmax[bin0_min, bin1_min] = k + 1

                        delta_right = fbin0_max - (<acc_t> bin0_max)

                        k = outmax[bin0_max, bin1_min]
                        indices[indptr[bin0_max * bins1 + bin1_min] + k] = idx
                        data[indptr[bin0_max * bins1 + bin1_min] + k] = inv_area * delta_right
                        outmax[bin0_max, bin1_min] = k + 1

                        for i in range(bin0_min + 1, bin0_max):
                            k = outmax[i, bin1_min]
                            indices[indptr[i * bins1 + bin1_min] + k] = idx
                            data[indptr[i * bins1 + bin1_min] + k] = inv_area
                            outmax[i, bin1_min] = k + 1

                    else:
                        # spread on n pix in dim0 and m pixel in dim1:
                        delta_left = (<acc_t> (bin0_min + 1)) - fbin0_min
                        delta_right = fbin0_max - (<acc_t> bin0_max)
                        delta_down = (<acc_t> (bin1_min + 1)) - fbin1_min
                        delta_up = fbin1_max - (<acc_t> bin1_max)
                        inv_area = 1.0 / ((fbin0_max - fbin0_min) * (fbin1_max - fbin1_min))

                        k = outmax[bin0_min, bin1_min]
                        indices[indptr[bin0_min * bins1 + bin1_min] + k] = idx
                        data[indptr[bin0_min * bins1 + bin1_min] + k] = inv_area * delta_left * delta_down
                        outmax[bin0_min, bin1_min] = k + 1

                        k = outmax[bin0_min, bin1_max]
                        indices[indptr[bin0_min * bins1 + bin1_max] + k] = idx
                        data[indptr[bin0_min * bins1 + bin1_max] + k] = inv_area * delta_left * delta_up
                        outmax[bin0_min, bin1_max] = k + 1

                        k = outmax[bin0_max, bin1_min]
                        indices[indptr[bin0_max * bins1 + bin1_min] + k] = idx
                        data[indptr[bin0_max * bins1 + bin1_min] + k] = inv_area * delta_right * delta_down
                        outmax[bin0_max, bin1_min] = k + 1

                        k = outmax[bin0_max, bin1_max]
                        indices[indptr[bin0_max * bins1 + bin1_max] + k] = idx
                        data[indptr[bin0_max * bins1 + bin1_max] + k] = inv_area * delta_right * delta_up
                        outmax[bin0_max, bin1_max] = k + 1

                        for i in range(bin0_min + 1, bin0_max):
                            k = outmax[i, bin1_min]
                            indices[indptr[i * bins1 + bin1_min] + k] = idx
                            data[indptr[i * bins1 + bin1_min] + k] = inv_area * delta_down
                            outmax[i, bin1_min] = k + 1

                            for j in range(bin1_min + 1, bin1_max):
                                k = outmax[i, j]
                                indices[indptr[i * bins1 + j] + k] = idx
                                data[indptr[i * bins1 + j] + k] = inv_area
                                outmax[i, j] = k + 1

                            k = outmax[i, bin1_max]
                            indices[indptr[i * bins1 + bin1_max] + k] = idx
                            data[indptr[i * bins1 + bin1_max] + k] = inv_area * delta_up
                            outmax[i, bin1_max] = k + 1

                        for j in range(bin1_min + 1, bin1_max):
                            k = outmax[bin0_min, j]
                            indices[indptr[bin0_min * bins1 + j] + k] = idx
                            data[indptr[bin0_min * bins1 + j] + k] = inv_area * delta_left
                            outmax[bin0_min, j] = k + 1

                            k = outmax[bin0_max, j]
                            indices[indptr[bin0_max * bins1 + j] + k] = idx
                            data[indptr[bin0_max * bins1 + j] + k] = inv_area * delta_right
                            outmax[bin0_max, j] = k + 1

        self.data = numpy.asarray(data)
        self.indices = numpy.asarray(indices)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def calc_lut_nosplit(self):
        """
        "calculate the max number of elements in the LUT and populate it

        This is the version which does not split pixels.

        """
        cdef:
            float delta0 = self.delta0, pos0_min = self.pos0_min, c0, fbin0
            float delta1 = self.delta1, pos1_min = self.pos1_min, c1, fbin1, fbin1_max
            int bin0, bins0 = self.bins[0]
            int bin1, bins1 = self.bins[1]
            cnumpy.int32_t k, idx, lut_size, i, j, size = self.size, nnz
            bint check_mask
            double[::1] cpos0 = self.cpos0
            double[::1] cpos1 = self.cpos1
            cnumpy.int32_t[:, ::1] outmax = numpy.zeros((bins0, bins1), dtype=numpy.int32)
            cnumpy.int32_t[::1] indptr, indices
            float[::1] data
            mask_t[::1] cmask

        if self.check_mask:
            cmask = self.cmask
            check_mask = True
        else:
            check_mask = False

        with nogil:
            for idx in range(size):
                if (check_mask) and (cmask[idx]):
                    continue

                c0 = cpos0[idx]
                c1 = cpos1[idx]

                bin0 = < int > get_bin_number(c0, pos0_min, delta0)
                bin1 = < int > get_bin_number(c1, pos1_min, delta1)

                if (bin0 < 0) or (bin0 >= bins0) or (bin1 < 0) or (bin1 >= bins1):
                    continue

                outmax[bin0, bin1] += 1

        indptr = numpy.concatenate(([numpy.int32(0)], 
                                    numpy.asarray(outmax).ravel().cumsum(dtype=numpy.int32)))
        self.nnz = nnz = indptr[bins0 * bins1]
        self.indptr = numpy.asarray(indptr)
        # Just recycle the outmax array
        outmax[:, :] = 0
        lut_nbytes = nnz * (sizeof(float) + sizeof(cnumpy.int32_t)) + bins0 * bins1 * sizeof(cnumpy.int32_t)
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
                        raise MemoryError("CSR Matrix is %.3fGB whereas the memory of the system is only %s" %
                                          (lut_nbytes/2.**30, memsize/2.**30))
        # else hope we have enough memory

        data = numpy.zeros(nnz, dtype=numpy.float32)
        indices = numpy.zeros(nnz, dtype=numpy.int32)
        with nogil:
            for idx in range(size):
                if (check_mask) and cmask[idx]:
                    continue

                c0 = cpos0[idx]
                c1 = cpos1[idx]

                fbin0 = get_bin_number(c0, pos0_min, delta0)
                fbin1 = get_bin_number(c1, pos1_min, delta1)

                bin0 = < int > fbin0
                bin1 = < int > fbin1

                if (bin0 < 0) or (bin0 >= bins0) or (bin1 < 0) or (bin1 >= bins1):
                    continue

                # No pixel splitting: All pixel is within a single bin
                k = outmax[bin0, bin1]
                indices[indptr[bin0 * bins1 + bin1] + k] = idx
                data[indptr[bin0 * bins1 + bin1] + k] = onef
                outmax[bin0, bin1] += 1

        self.data = numpy.asarray(data)
        self.indices = numpy.asarray(indices)

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
                  int coef_power=1
                  ):
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
        :param coef_power: set to 2 for variance propagation, leave to 1 for mean calculation
        :return:  I(2d), bin_centers0(1d), bin_centers1(1d), weighted histogram(2d), unweighted histogram (2d)
        :rtype: 5-tuple of ndarrays

        """
        cdef:
            cnumpy.int32_t i = 0, j = 0, idx = 0, bins = self.bins[0] * self.bins[1], size = self.size
            acc_t acc_data = 0.0, acc_count = 0.0, epsilon = 1e-10, coef = 0.0
            data_t data = 0.0, cdummy = 0.0, cddummy = 0.0
            bint do_dummy = False, do_dark = False, do_flat = False, do_polarization = False, do_solidAngle = False
            acc_t[::1] sum_data = numpy.zeros(bins, dtype=acc_d)
            acc_t[::1] sum_count = numpy.zeros(bins, dtype=acc_d)
            data_t[::1] merged = numpy.zeros(bins, dtype=data_d)
            data_t[::1] ccoef = self.data, 
            data_t[::1] cdata, tdata, cflat, cdark, csolidAngle, cpolarization
            cnumpy.int32_t[::1] indices = self.indices, indptr = self.indptr

        assert weights.size == size, "weights size"

        if dummy is not None:
            do_dummy = True
            cdummy = <data_t> float(dummy)
            if delta_dummy is None:
                cddummy = <data_t> 0.0
            else:
                cddummy = <data_t> float(delta_dummy)
        else:
            do_dummy = False
            cdummy = <data_t> float(self.empty)

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
                tdata = numpy.ascontiguousarray(weights.ravel(), dtype=data_d)
                cdata = numpy.zeros(size, dtype=data_d)
                for i in prange(size, nogil=True, schedule="static"):
                    data = tdata[i]
                    if ((cddummy != 0) and (fabs(data - cdummy) > cddummy)) or ((cddummy == 0) and (data != cdummy)):
                        cdata[i] += data
                    else:
                        cdata[i] += cdummy
            else:
                cdata = numpy.ascontiguousarray(weights.ravel(), dtype=data_d)

        for i in prange(bins, nogil=True, schedule="guided"):
            acc_data = 0.0
            acc_count = 0.0
            for j in range(indptr[i], indptr[i + 1]):
                idx = indices[j]
                coef = ccoef[j]
                if coef == 0.0:
                    continue
                data = cdata[idx]
                if do_dummy and (data == cdummy):
                    continue
                acc_data = acc_data + (coef ** coef_power) * data
                acc_count = acc_count + coef
            sum_data[i] += acc_data
            sum_count[i] += acc_count
            if acc_count > epsilon:
                merged[i] += acc_data / acc_count / normalization_factor
            else:
                merged[i] += cdummy
        return (numpy.asarray(merged).reshape(self.bins).T, 
                self.bin_centers0, 
                self.bin_centers1, 
                numpy.asarray(sum_data).reshape(self.bins).T, 
                numpy.asarray(sum_count).reshape(self.bins).T)

    @property
    @deprecated(replacement="bin_centers0", since_version="0.16", only_once=True)
    def outPos0(self):
        return self.bin_centers0

    @property
    @deprecated(replacement="bin_centers1", since_version="0.16", only_once=True)
    def outPos1(self):
        return self.bin_centers1

