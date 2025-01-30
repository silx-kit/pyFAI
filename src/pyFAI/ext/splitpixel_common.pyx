# coding: utf-8
#cython: embedsignature=True, language_level=3, binding=True
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False,
## This is for developping
## cython: profile=True, warn.undeclared=True, warn.unused=True, warn.unused_result=False, warn.unused_arg=True
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2021-2024 European Synchrotron Radiation Facility, France
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

"""Common code for full pixel splitting

"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "26/04/2024"
__status__ = "stable"
__license__ = "MIT"

include "regrid_common.pxi"
from ..utils import crc32
from .sparse_builder cimport SparseBuilder
from libc.math cimport INFINITY
import logging
logger = logging.getLogger(__name__)

if logging.root.level >= logging.ERROR:
    NUM_WARNING = -1
elif logging.root.level >= logging.WARNING:
    NUM_WARNING = 10
elif logging.root.level >= logging.INFO:
    NUM_WARNING = 100
else:
    NUM_WARNING = 10000


def calc_boundaries(position_t[:, :, ::1] cpos,
                    mask_t[::1] cmask=None,
                    pos0_range=None,
                    pos1_range=None,
                    bint allow_pos0_neg=False,
                    bint chiDiscAtPi=True,
                    bint clip_pos1=False):
    """Calculate the boundaries in radial/azimuthal space in fullsplit mode.

    :param cpos: 3D array of position, shape: (size, 4 (corners), 2 (rad/azim))
    :param cmask: 1d array with mask
    :param pos0_range: minimum and maximum of the radial range (2th, q, ...)
    :param pos1_range: minimum and maximum of the azimuthal range (chi)
    :param allow_pos0_neg: set to allow the radial range to start below 0 (usful for log_q radial range)
    :param chiDiscAtPi: tell if azimuthal discontinuity is at 0 (0° when False) or π (180° when True)
    :param clip_pos1: clip the azimuthal range to [-π π] (or [0 2π] depending on chiDiscAtPi), set to False to deactivate behavior
    :return: Boundaries(pos0_min, pos0_max, pos1_min, pos1_max)
    """
    cdef:
        Py_ssize_t idx, size= cpos.shape[0]
        bint check_mask = False
        position_t pos0_min, pos0_max, pos1_min=-INFINITY, pos1_max=INFINITY
        position_t a0, a1, b0, b1, c0, c1, d0, d1
        position_t min0, min1, max0, max1

    if cmask is not None:
        check_mask = True
        assert cmask.size == size, "mask size"

    if (pos0_range is None or pos1_range is None):
        with nogil:
            for idx in range(size):
                if not (check_mask and cmask[idx]):
                    pos0_max = pos0_min = cpos[idx, 0, 0]
                    pos1_max = pos1_min = cpos[idx, 0, 1]
                    break
            for idx in range(size):
                if check_mask and cmask[idx]:
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
                pos0_max = max(pos0_max, max0)
                pos0_min = min(pos0_min, min0)
                min1 = min(a1, b1, c1, d1)
                max1 = max(a1, b1, c1, d1)
                pos1_max = max(pos1_max, max1)
                pos1_min = min(pos1_min, min1)

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


class FullSplitIntegrator:
    """
    Abstract class which contains the boundary selection and the LUT calculation
    """

    def __init__(self,
                 pos not None,
                 bins=(100, 36),
                 pos0_range=None,
                 pos1_range=None,
                 mask=None,
                 mask_checksum=None,
                 bint allow_pos0_neg=False,
                 bint chiDiscAtPi=True,
                 bint clip_pos1=True):
        """Constructor of the class:

        :param pos: 3D or 4D array with the coordinates of each pixel point
        :param bins: number of output bins (tth=100, chi=36 by default)
        :param pos0_range: minimum and maximum  of the 2th range
        :param pos1_range: minimum and maximum  of the chi range
        :param mask: array (of int8) with masked pixels with 1 (0=not masked)
        :param mask_checksum: int with the checksum of the mask
        :param allow_pos0_neg: enforce the q<0 is usually not possible
        :param chiDiscAtPi: tell if azimuthal discontinuity is at 0 (0° when False) or π (180° when True)
        :param clip_pos1: clip the azimuthal range to [-π π] (or [0 2π] depending on chiDiscAtPi), set to False to deactivate behavior
        """
        if pos.ndim > 3:  # create a view
            pos = pos.reshape((-1, 4, 2))
        assert pos.shape[1] == 4, "pos.shape[1] == 4"
        assert pos.shape[2] == 2, "pos.shape[2] == 2"
        assert pos.ndim == 3, "pos.ndim == 3"
        self.pos = numpy.ascontiguousarray(pos, dtype=position_d)
        self.size = pos.shape[0]
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
        pos0_min, pos0_maxin, pos1_min, pos1_maxin = calc_boundaries(self.pos, self.cmask,
                                                                     pos0_range, pos1_range,
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
            position_t[:, :, ::1] cpos = numpy.ascontiguousarray(self.pos, dtype=position_d)
            position_t[:, ::1] v8 = numpy.empty((4,2), dtype=position_d)
            buffer_t[::1] buffer = numpy.zeros(self.bins, dtype=buffer_d)
            mask_t[::1] cmask = None
            position_t pos0_min = 0.0, pos1_min = 0.0, pos1_max = 0.0, pos1_maxin=0.0
            position_t areaPixel = 0, delta = 0, areaPixel2 = 0
            position_t a0, b0, c0, d0, a1, b1, c1, d1
            position_t inv_area, area_pixel, sub_area, sum_area
            position_t min0, max0, min1, max1
            Py_ssize_t bins=self.bins, idx = 0, bin = 0, bin0 = 0, bin0_max = 0, bin0_min = 0, size = 0
            bint check_pos1=self.pos1_range is not None, check_mask = False, chiDiscAtPi=self.chiDiscAtPi
            SparseBuilder builder = SparseBuilder(bins, block_size=32, heap_size=(size+1023)&~(1023))

        pos0_min = self.pos0_min
        pos1_min = self.pos1_min
        pos1_max = self.pos1_max
        pos1_maxin = self.pos1_maxin

        delta = self.delta

        size = self.size
        check_mask = self.check_mask
        if check_mask:
            cmask = self.cmask

        with nogil:
            for idx in range(size):

                if (check_mask) and (cmask[idx]):
                    continue
                # Play with coordinates ...
                v8[:, :] = cpos[idx, :, :]
                area_pixel = - _recenter(v8, chiDiscAtPi) / delta
                a0 = get_bin_number(v8[0, 0], pos0_min, delta)
                a1 = v8[0, 1]
                b0 = get_bin_number(v8[1, 0], pos0_min, delta)
                b1 = v8[1, 1]
                c0 = get_bin_number(v8[2, 0], pos0_min, delta)
                c1 = v8[2, 1]
                d0 = get_bin_number(v8[3, 0], pos0_min, delta)
                d1 = v8[3, 1]

                min0 = min(a0, b0, c0, d0)
                max0 = max(a0, b0, c0, d0)

                if (max0 < 0) or (min0 >= bins):
                    continue
                if check_pos1:
                    min1 = min(a1, b1, c1, d1)
                    max1 = max(a1, b1, c1, d1)
                    if (max1 < pos1_min) or (min1 > pos1_maxin):
                        continue

                bin0_min = < Py_ssize_t > floor(min0)
                bin0_max = < Py_ssize_t > floor(max0)

                if bin0_min == bin0_max:
                    # All pixel is within a single bin
                    builder.cinsert(bin0_min, idx, 1.0)
                else:
                    # else we have pixel spliting.
                    # offseting the min bin of the pixel to be zero to avoid percision problems
                    bin0_min = max(0, bin0_min)
                    bin0_max = min(bins, bin0_max + 1)

                    _integrate1d(buffer, a0, a1, b0, b1)  # A-B
                    _integrate1d(buffer, b0, b1, c0, c1)  # B-C
                    _integrate1d(buffer, c0, c1, d0, d1)  # C-D
                    _integrate1d(buffer, d0, d1, a0, a1)  # D-A

                    sum_area = 0.0
                    for bin in range(bin0_min, bin0_max):
                        sum_area += buffer[bin]
                    inv_area = 1.0 / sum_area
                    for bin in range(bin0_min, bin0_max):
                        builder.cinsert(bin, idx, buffer[bin]*inv_area)
                    # Check the total area:
                    buffer[bin0_min:bin0_max] = 0.0

        return builder

    def calc_lut_2d(self):
        """Calculate the LUT and return the LUT-builder object
        """
        cdef:
            Py_ssize_t bins0=self.bins[0], bins1=self.bins[1], size = self.size
            position_t[:, :, ::1] cpos = numpy.ascontiguousarray(self.pos, dtype=position_d)
            position_t[:, ::1] v8 = numpy.empty((4,2), dtype=position_d)
            mask_t[:] cmask = self.cmask
            bint check_mask = False, chiDiscAtPi = self.chiDiscAtPi
            position_t min0 = 0.0, max0 = 0.0, min1 = 0.0, max1 = 0.0, inv_area = 0.0
            position_t pos0_min = 0.0, pos1_min = 0.0, pos1_max = 0.0, pos0_maxin = 0.0, pos1_maxin = 0.0
            position_t a0 = 0.0, a1 = 0.0, b0 = 0.0, b1 = 0.0, c0 = 0.0, c1 = 0.0, d0 = 0.0, d1 = 0.0
            position_t delta0, delta1
            position_t foffset0, foffset1, sum_area, area
            Py_ssize_t i = 0, j = 0, idx = 0
            Py_ssize_t ioffset0, ioffset1, w0, w1, bw0=15, bw1=15
            buffer_t[::1] linbuffer = numpy.empty(256, dtype=buffer_d)
            buffer_t[:, ::1] buffer = numpy.asarray(linbuffer[:(bw0+1)*(bw1+1)]).reshape((bw0+1,bw1+1))
            SparseBuilder builder = SparseBuilder(bins1*bins0, block_size=8, heap_size=(size+1023)&~(1023))

        if self.cmask is not None:
            check_mask = True
            cmask = self.cmask

        pos0_min = self.pos0_min
        pos0_maxin = self.pos0_maxin
        pos1_maxin = self.pos1_maxin
        pos1_max = self.pos1_max
        pos1_min = self.pos1_min
        delta0 = self.delta0
        delta1 = self.delta1

        with nogil:
            for idx in range(size):

                if (check_mask) and (cmask[idx]):
                    continue

                # Play with coordinates ...
                v8[:, :] = cpos[idx, :, :]
                area = _recenter(v8, chiDiscAtPi) # this is an unprecise measurement of the surface of the pixels
                a0 = v8[0, 0]
                a1 = v8[0, 1]
                b0 = v8[1, 0]
                b1 = v8[1, 1]
                c0 = v8[2, 0]
                c1 = v8[2, 1]
                d0 = v8[3, 0]
                d1 = v8[3, 1]

                min0 = min(a0, b0, c0, d0)
                max0 = max(a0, b0, c0, d0)
                min1 = min(a1, b1, c1, d1)
                max1 = max(a1, b1, c1, d1)

                if (max0 < pos0_min) or (min0 > pos0_maxin) or (max1 < pos1_min) or (min1 >= pos1_max):
                        continue

                # Swith to bin space.
                a0 = get_bin_number(_clip(a0, pos0_min, pos0_maxin), pos0_min, delta0)
                a1 = get_bin_number(_clip(a1, pos1_min, pos1_maxin), pos1_min, delta1)
                b0 = get_bin_number(_clip(b0, pos0_min, pos0_maxin), pos0_min, delta0)
                b1 = get_bin_number(_clip(b1, pos1_min, pos1_maxin), pos1_min, delta1)
                c0 = get_bin_number(_clip(c0, pos0_min, pos0_maxin), pos0_min, delta0)
                c1 = get_bin_number(_clip(c1, pos1_min, pos1_maxin), pos1_min, delta1)
                d0 = get_bin_number(_clip(d0, pos0_min, pos0_maxin), pos0_min, delta0)
                d1 = get_bin_number(_clip(d1, pos1_min, pos1_maxin), pos1_min, delta1)

                # Recalculate here min0, max0, min1, max1 based on the actual area of ABCD and the width/height ratio
                min0 = min(a0, b0, c0, d0)
                max0 = max(a0, b0, c0, d0)
                min1 = min(a1, b1, c1, d1)
                max1 = max(a1, b1, c1, d1)
                foffset0 = floor(min0)
                foffset1 = floor(min1)
                ioffset0 = <Py_ssize_t> foffset0
                ioffset1 = <Py_ssize_t> foffset1
                w0 = <Py_ssize_t>(ceil(max0) - foffset0)
                w1 = <Py_ssize_t>(ceil(max1) - foffset1)
                if (w0>bw0) or (w1>bw1):
                    if (w0+1)*(w1+1)>linbuffer.shape[0]:
                        with gil:
                            linbuffer = numpy.zeros((w0+1)*(w1+1), dtype=buffer_d)
                            buffer = numpy.asarray(linbuffer).reshape((w0+1,w1+1))
                            logger.debug("malloc  %s->%s and %s->%s", w0, bw0, w1, bw1)
                    else:
                        with gil:
                            buffer = numpy.asarray(linbuffer[:(w0+1)*(w1+1)]).reshape((w0+1,w1+1))
                            logger.debug("reshape %s->%s and %s->%s", w0, bw0, w1, bw1)
                    bw0 = w0
                    bw1 = w1

                a0 -= foffset0
                a1 -= foffset1
                b0 -= foffset0
                b1 -= foffset1
                c0 -= foffset0
                c1 -= foffset1
                d0 -= foffset0
                d1 -= foffset1

                # ABCD is anti-trigonometric order: order input position accordingly
                _integrate2d(buffer, a0, a1, b0, b1)
                _integrate2d(buffer, b0, b1, c0, c1)
                _integrate2d(buffer, c0, c1, d0, d1)
                _integrate2d(buffer, d0, d1, a0, a1)

                sum_area = 0.0
                for i in range(w0):
                    for j in range(w1):
                        sum_area += buffer[i, j]
                inv_area = 1.0 / sum_area

                for i in range(w0):
                    for j in range(w1):
                        builder.cinsert((ioffset0 + i)*bins1 + ioffset1 + j, idx, buffer[i, j] * inv_area)
                linbuffer[:] = 0.0 # reset full buffer since it is likely faster than memsetting 2d view
        return builder
