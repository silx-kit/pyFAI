# -*- coding: utf-8 -*-
#cython: embedsignature=True, language_level=3, binding=True
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False,
## This is for developping
## cython: profile=True, warn.undeclared=True, warn.unused=True, warn.unused_result=False, warn.unused_arg=True
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2021-2022 European Synchrotron Radiation Facility, France
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

"""Common code for bounding-box pixel splitting

"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "28/09/2023"
__status__ = "stable"
__license__ = "MIT"

include "regrid_common.pxi"
from ..utils import crc32
from .sparse_builder cimport SparseBuilder
from libc.math cimport INFINITY
import logging
logger = logging.getLogger(__name__)


def calc_boundaries(position_t[::1] pos0,
                    position_t[::1] delta_pos0=None,
                    position_t[::1] pos1=None,
                    position_t[::1] delta_pos1=None,
                    mask_t[::1] cmask=None,
                    pos0_range=None,
                    pos1_range=None,
                    bint allow_pos0_neg=False,
                    bint chiDiscAtPi=True,
                    bint clip_pos1=False):
    """Calculate the boundaries in radial/azimuthal space in bounding-box mode.

    :param pos0: 1D array with pos0: tth or q_vect
    :param delta_pos0: 1D array with delta pos0: max center-corner distance. None to deactivate splitting
    :param pos1: 1D array with pos1: chi
    :param delta_pos1: 1D array with max pos1: max center-corner distance, unused !
    :param cmask: 1d array with mask
    :param pos0_range: minimum and maximum of the radial range (2th, q, ...)
    :param pos1_range: minimum and maximum of the azimuthal range (chi)
    :param allow_pos0_neg: set to allow the radial range to start below 0 (usful for log_q radial range)
    :param chiDiscAtPi: tell if azimuthal discontinuity is at 0 (0° when False) or π (180° when True)
    :param clip_pos1: clip the azimuthal range to [-π π] (or [0 2π] depending on chiDiscAtPi), set to False to deactivate behavior
    :return: Boundaries(pos0_min, pos0_max, pos1_min, pos1_max)
    """
    cdef:
        Py_ssize_t idx, size = pos0.shape[0]
        bint check_mask = False, check_pos1 = True, do_split=True
        position_t pos0_min, pos0_max, pos1_min=-INFINITY, pos1_max=INFINITY
        position_t c0, c1=0.0, d0=0.0, d1=0.0

    if cmask is not None:
        check_mask = True
        assert cmask.size == size, "mask size"
    if pos1 is None:
        check_pos1 = False
    if delta_pos0 is None:
        do_split = False

    if (pos0_range is None or pos1_range is None):
        with nogil:
            for idx in range(size):
                if not (check_mask and cmask[idx]):
                    pos0_max = pos0_min = pos0[idx]
                    if check_pos1:
                        pos1_max = pos1_min = pos1[idx]
                    break
            for idx in range(size):
                if (check_mask) and (cmask[idx]):
                    continue
                c0 = pos0[idx]
                if do_split:
                    d0 = delta_pos0[idx]

                pos0_max = max(pos0_max, c0 + d0)
                pos0_min = min(pos0_min, c0 - d0)

                if check_pos1:
                    c1 = pos1[idx]
                    if do_split:
                        d1 = delta_pos1[idx]

                    pos1_max = max(pos1_max, c1 + d1)
                    pos1_min = min(pos1_min, c1 - d1)

    if (not allow_pos0_neg):
        pos0_min = max(0.0, pos0_min)
        pos0_max = max(0.0, pos0_max)
    if clip_pos1:
        chiDiscAtPi = 1 if chiDiscAtPi else 0
        pos1_max = min(pos1_max, (2 - chiDiscAtPi) * pi)
        pos1_min = max(pos1_min, -chiDiscAtPi * pi)

    if pos0_range is not None and len(pos0_range) > 1:
        pos0_min = min(pos0_range)
        pos0_max = max(pos0_range)

    if pos1_range is not None and len(pos1_range) > 1:
        pos1_min = min(pos1_range)
        pos1_max = max(pos1_range)

    return Boundaries(pos0_min, pos0_max, pos1_min, pos1_max)


class SplitBBoxIntegrator:
    """
    Abstract class which contains the boundary selection and the LUT calculation
    """

    def __init__(self,
                 pos0 not None,
                 delta_pos0=None,
                 pos1=None,
                 delta_pos1=None,
                 bins=(100, 36),
                 pos0_range=None,
                 pos1_range=None,
                 mask=None,
                 mask_checksum=None,
                 bint allow_pos0_neg=False,
                 bint chiDiscAtPi=True,
                 bint clip_pos1=True):
        """Constructor of the class:

        :param pos0: 1D array with pos0: tth or q_vect
        :param delta_pos0: 1D array with delta pos0: max center-corner distance. None to deactivate splitting
        :param pos1: 1D array with pos1: chi
        :param delta_pos1: 1D array with max pos1: max center-corner distance, unused !
        :param bins: number of output bins (tth=100, chi=36 by default)
        :param pos0_range: minimum and maximum  of the 2th range
        :param pos1_range: minimum and maximum  of the chi range
        :param mask: array (of int8) with masked pixels with 1 (0=not masked)
        :param mask_checksum: int with the checksum of the mask
        :param allow_pos0_neg: enforce the q<0 is usually not possible
        :param chiDiscAtPi: tell if azimuthal discontinuity is at 0 (0° when False) or π (180° when True)
        :param clip_pos1: clip the azimuthal range to [-π π] (or [0 2π] depending on chiDiscAtPi), set to False to deactivate behavior
        """
        self.cpos0 = numpy.ascontiguousarray(pos0.ravel(), dtype=position_d)
        self.size = pos0.size
        self.dpos0 = None
        self.cpos1 = None
        self.dpos1 = None

        if delta_pos0 is not None:
            self.dpos0 = numpy.ascontiguousarray(delta_pos0.ravel(), dtype=position_d)
            assert self.dpos0.size == self.size, "dpos0 size"
        if pos1 is not None:
            self.cpos1 = numpy.ascontiguousarray(pos1.ravel(), dtype=position_d)
            assert self.cpos1.size == self.size, "cpos1 size"
        if delta_pos1 is not None:
            self.dpos1 = numpy.ascontiguousarray(delta_pos1.ravel(), dtype=position_d)
            assert self.dpos1.size == self.size, "dpos1 size"

        if "__len__" in dir(bins):
            self.bins = tuple(max(i, 1) for i in bins)
        else:
            self.bins = bins or 1
        self.allow_pos0_neg = allow_pos0_neg
        self.chiDiscAtPi = chiDiscAtPi

        if mask is None:
            self.cmask = None
            self.mask_checksum = None
        else:
            assert mask.size == self.size, "mask size"
            self.cmask = numpy.ascontiguousarray(mask.ravel(), dtype=mask_d)
            self.mask_checksum = mask_checksum if mask_checksum else crc32(mask)

        #keep this unchanged for validation of the range or not
        self.pos0_range = pos0_range
        self.pos1_range = pos1_range
        cdef:
            position_t pos0_max, pos1_max, pos0_maxin, pos1_maxin
        pos0_min, pos0_maxin, pos1_min, pos1_maxin = calc_boundaries(self.cpos0, self.dpos0,
                                                                     self.cpos1, self.dpos1,
                                                                     self.cmask, pos0_range, pos1_range,
                                                                     allow_pos0_neg, chiDiscAtPi, clip_pos1)
        self.pos0_min = pos0_min
        self.pos1_min = pos1_min
        self.pos0_maxin = pos0_maxin
        self.pos1_maxin = pos1_maxin
        self.pos0_max = calc_upper_bound(pos0_maxin)
        self.pos1_max = calc_upper_bound(pos1_maxin)

    def calc_lut_1d(self):
        """Calculate the LUT and return the LUT-builder object
        """
        cdef:
            position_t[::1] cpos0, cpos1, dpos0, dpos1
            mask_t[::1] cmask
            position_t pos0_min = 0.0, pos1_min = 0.0, pos1_maxin=0.0
            position_t delta, inv_area=0.0
            position_t c0, d0=0.0, c1, d1=0.0, min0, max0
            position_t fbin0_min, fbin0_max, delta_left, delta_right
            Py_ssize_t bins, idx=0, bin=0, bin0_max=0, bin0_min=0, size
            bint check_pos1=self.pos1_range, check_mask=False, do_split=True
            SparseBuilder builder

        check_pos1=self.pos1_range is not None
        cpos0 = self.cpos0
        cpos1 = self.cpos1
        dpos0 = self.dpos0
        dpos1 = self.dpos1
        if self.dpos0 is None:
            do_split = False
        pos0_min = self.pos0_min
        pos1_min = self.pos1_min
        pos1_maxin = self.pos1_maxin
        delta = self.delta
        bins=self.bins
        size = self.size
        check_mask = self.check_mask
        cmask = self.cmask
        #heap size is larger than the size of the image and a multiple of 1024 which can be divided by 32
        builder = SparseBuilder(bins, block_size=32, heap_size=(size+1023)&~(1023))

        with nogil:
            for idx in range(size):
                if (check_mask) and (cmask[idx]):
                    continue
                c0 = cpos0[idx]
                if do_split:
                    d0 = dpos0[idx]
                min0 = c0 - d0
                max0 = c0 + d0

                if check_pos1:
                    c1 = cpos1[idx]
                    if do_split:
                        d1 = dpos1[idx]
                    if (c1+d1 < pos1_min) or (c1 - d1 > pos1_maxin):
                        continue

                fbin0_min = get_bin_number(min0, pos0_min, delta)
                fbin0_max = get_bin_number(max0, pos0_min, delta)
                bin0_min = <Py_ssize_t> fbin0_min
                bin0_max = <Py_ssize_t> fbin0_max

                if (bin0_max < 0) or (bin0_min >= bins):
                    continue
                bin0_max = min(bin0_max, bins - 1)
                bin0_min = max(bin0_min, 0)

                if bin0_min == bin0_max:
                    # All pixel is within a single bin
                    builder.cinsert(bin0_min, idx, 1.0)
                else:  # we have pixel splitting.
                    inv_area = 1.0 / (fbin0_max - fbin0_min)

                    delta_left = <position_t> (bin0_min + 1) - fbin0_min
                    builder.cinsert(bin0_min, idx, inv_area * delta_left)

                    delta_right = fbin0_max - <position_t> (bin0_max)
                    builder.cinsert(bin0_max, idx, inv_area * delta_right)

                    if bin0_min + 1 < bin0_max:
                        for bin in range(bin0_min + 1, bin0_max):
                            builder.cinsert(bin, idx, inv_area)
        return builder

    def calc_lut_2d(self):
        """Calculate the LUT and return the LUT-builder object
        """
        cdef:
            Py_ssize_t bins0, bins1, size
            position_t[::1] cpos0, cpos1, dpos0, dpos1
            mask_t[::1] cmask
            bint check_mask=False, do_split=True
            position_t c0, c1, d0=0.0, d1=0.0, min0=0.0, max0=0.0, min1=0.0, max1=0.0, inv_area=0.0
            position_t pos0_min=0.0, pos1_min=0.0
            position_t delta0, delta1, delta_down, delta_up, delta_left, delta_right
            position_t fbin0_min, fbin0_max, fbin1_min, fbin1_max
            Py_ssize_t i = 0, j = 0, idx = 0
            Py_ssize_t bin0_min, bin0_max, bin1_min, bin1_max
            SparseBuilder builder

        bins0=self.bins[0]
        bins1=self.bins[1]
        size = self.size
        cpos0 = self.cpos0
        cpos1 = self.cpos1
        dpos0 = self.dpos0
        dpos1 = self.dpos1
        if self.dpos0 is None:
            do_split = False
        pos0_min = self.pos0_min
        pos1_min = self.pos1_min
        size = self.size
        check_mask = self.check_mask
        cmask = self.cmask
        delta0 = self.delta0
        delta1 = self.delta1

        #heap size is larger than the size of the image and a multiple of 1024 which can be divided by 8
        builder = SparseBuilder(bins1*bins0, block_size=8, heap_size=(size+1023)&~(1023))
        with nogil:
            for idx in range(size):
                if (check_mask) and cmask[idx]:
                    continue
                c0 = cpos0[idx]
                c1 = cpos1[idx]
                if do_split:
                    d0 = dpos0[idx]
                    d1 = dpos1[idx]
                min0 = c0 - d0
                max0 = c0 + d0
                min1 = c1 - d1
                max1 = c1 + d1

                fbin0_min = get_bin_number(min0, pos0_min, delta0)
                fbin0_max = get_bin_number(max0, pos0_min, delta0)
                fbin1_min = get_bin_number(min1, pos1_min, delta1)
                fbin1_max = get_bin_number(max1, pos1_min, delta1)

                bin0_min = <Py_ssize_t> fbin0_min
                bin0_max = <Py_ssize_t> fbin0_max
                bin1_min = <Py_ssize_t> fbin1_min
                bin1_max = <Py_ssize_t> fbin1_max

                if (bin0_max < 0) or (bin0_min >= bins0) or (bin1_max < 0) or (bin1_min >= bins1):
                    continue

                #clip values
                bin0_max = min(bin0_max, bins0 - 1)
                bin0_min = max(bin0_min, 0)
                bin1_max = min(bin1_max, bins1 - 1)
                bin1_min = max(bin1_min, 0)

                if bin0_min == bin0_max:
                    if bin1_min == bin1_max:
                        # All pixel is within a single bin
                        builder.cinsert(bin0_min * bins1 + bin1_min, idx, 1.0)
                    else:
                        # spread on more than 2 bins in dim1
                        delta_down = (<position_t> (bin1_min + 1)) - fbin1_min
                        delta_up = fbin1_max - bin1_max
                        inv_area = 1.0 / (fbin1_max - fbin1_min)

                        builder.cinsert(bin0_min * bins1 + bin1_min, idx, inv_area * delta_down)
                        builder.cinsert(bin0_min * bins1 + bin1_max, idx, inv_area * delta_up)

                        for j in range(bin1_min + 1, bin1_max):
                            builder.cinsert(bin0_min * bins1 + j, idx, inv_area)

                else:  # spread on more than 2 bins in dim 0
                    if bin1_min == bin1_max:
                        # All pixel fall on 1 bins in dim 1
                        inv_area = 1.0 / (fbin0_max - fbin0_min)
                        delta_left = (<position_t> (bin0_min + 1)) - fbin0_min
                        builder.cinsert(bin0_min * bins1 + bin1_min, idx, inv_area * delta_left)

                        delta_right = fbin0_max - (<position_t> bin0_max)
                        builder.cinsert(bin0_max * bins1 + bin1_min, idx, inv_area * delta_right)

                        for i in range(bin0_min + 1, bin0_max):
                            builder.cinsert(i * bins1 + bin1_min, idx, inv_area)

                    else:
                        # spread on n pix in dim0 and m pixel in dim1:
                        delta_left = (<position_t> (bin0_min + 1)) - fbin0_min
                        delta_right = fbin0_max - (<position_t> bin0_max)
                        delta_down = (<position_t> (bin1_min + 1)) - fbin1_min
                        delta_up = fbin1_max - (<position_t> bin1_max)
                        inv_area = 1.0 / ((fbin0_max - fbin0_min) * (fbin1_max - fbin1_min))

                        builder.cinsert(bin0_min * bins1 + bin1_min, idx, inv_area * delta_left * delta_down)
                        builder.cinsert(bin0_min * bins1 + bin1_max, idx, inv_area * delta_left * delta_up)
                        builder.cinsert(bin0_max * bins1 + bin1_min, idx, inv_area * delta_right * delta_down)
                        builder.cinsert(bin0_max * bins1 + bin1_max, idx, inv_area * delta_right * delta_up)

                        for i in range(bin0_min + 1, bin0_max):
                            builder.cinsert(i * bins1 + bin1_min, idx, inv_area * delta_down)
                            for j in range(bin1_min + 1, bin1_max):
                                builder.cinsert(i * bins1 + j, idx, inv_area)
                            builder.cinsert(i * bins1 + bin1_max, idx, inv_area * delta_up)
                        for j in range(bin1_min + 1, bin1_max):
                            builder.cinsert(bin0_min * bins1 + j, idx, inv_area * delta_left)
                            builder.cinsert(bin0_max * bins1 + j, idx, inv_area * delta_right)
        return builder
