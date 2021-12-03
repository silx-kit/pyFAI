# -*- coding: utf-8 -*-
#cython: embedsignature=True, language_level=3, binding=True
##cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False,
## This is for developping
# cython: profile=True, warn.undeclared=True, warn.unused=True, warn.unused_result=False, warn.unused_arg=True
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2021-2021 European Synchrotron Radiation Facility, France
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

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "03/12/2021"
__status__ = "stable"
__license__ = "MIT"

include "regrid_common.pxi"
from ..utils import crc32
from .sparse_builder cimport SparseBuilder
import logging
logger = logging.getLogger(__name__)
cdef Py_ssize_t NUM_WARNING
if logger.level >= logging.ERROR:
    NUM_WARNING = -1
elif logger.level >= logging.WARNING:
    NUM_WARNING = 10 
elif logger.level >= logging.INFO:
    NUM_WARNING = 100 
else:
    NUM_WARNING = 10000

def calc_boundaries(position_t[:, :, ::1] cpos,
                    mask_t[::1] cmask=None,
                    pos0_range=None,
                    pos1_range=None
                    ):
    """Calculate the boundaries in radial/azimuthal space in fullsplit mode.
    
    :param cpos: 3D array of position, shape: (size, 4 (corners), 2 (rad/azim)) 
    :param cmask: 1d array with mask
     
    :return: (pos0_min, pos0_max, pos1_min, pos1_max)
    """
    cdef:
        Py_ssize_t idx, size= cpos.shape[0]
        bint check_mask = False
        position_t pos0_min, pos1_min, pos0_max, pos1_max
        position_t a0, a1, b0, b1, c0, c1, d0, d1
        position_t min0, min1, max0, max1
          
    if cmask is not None:
        check_mask = True
        assert cmask.size == size, "mask size"

    if (pos0_range is None or pos1_range is None):
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
                if max0 > pos0_max:
                    pos0_max = max0
                if min0 < pos0_min:
                    pos0_min = min0
                min1 = min(a1, b1, c1, d1)
                max1 = max(a1, b1, c1, d1)
                if max1 > pos1_max:
                    pos1_max = max1
                if min1 < pos1_min:
                    pos1_min = min1

    if pos0_range is not None and len(pos0_range) > 1:
        pos0_min = min(pos0_range)
        pos0_max = max(pos0_range)

    if pos1_range is not None and len(pos1_range) > 1:
        pos1_min = min(pos1_range)
        pos1_max = max(pos1_range)

    return (pos0_min, pos0_max, pos1_min, pos1_max)


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
                 bint chiDiscAtPi=True):
        """Constructor of the class:
        
        :param pos: 3D or 4D array with the coordinates of each pixel point
        :param bins: number of output bins (tth=100, chi=36 by default)
        :param pos0_range: minimum and maximum  of the 2th range
        :param pos1_range: minimum and maximum  of the chi range
        :param mask: array (of int8) with masked pixels with 1 (0=not masked)
        :param mask_checksum: int with the checksum of the mask
        :param allow_pos0_neg: enforce the q<0 is usually not possible
        :param chiDiscAtPi: 
        """
        if pos.ndim > 3:  # create a view
            pos = pos.reshape((-1, 4, 2))
        assert pos.shape[1] == 4, "pos.shape[1] == 4"
        assert pos.shape[2] == 2, "pos.shape[2] == 2"
        assert pos.ndim == 3, "pos.ndim == 3"
        self.pos = numpy.ascontinuousarray(pos, dtype=position_d)
        self.size = pos.shape[0]
        if "__len__" in dir(bins): 
            self.bins = tuple(max(i, 1) for i in bins)
        else:
            self.bins = bins or 1 
        self.allow_pos0_neg = allow_pos0_neg
        self.chiDiscAtPi = chiDiscAtPi

        if mask is None:
            self.cmask = None
            self.check_mask = False
            self.mask_checksum = None
        else:
            assert mask.size == self.size, "mask size"
            self.check_mask = True
            self.cmask = numpy.ascontiguousarray(mask.ravel(), dtype=mask_d)
            self.mask_checksum = mask_checksum if mask_checksum else crc32(mask)

        #keep this unchanged for validation of the range or not
        self.pos0_range = pos0_range
        self.pos1_range = pos1_range
        cdef:
            position_t pos0_max, pos1_max, pos0_maxin, pos1_maxin
        pos0_min, pos0_maxin, pos1_min, pos1_maxin = calc_boundaries(self.pos, self.cmask, pos0_range, pos1_range)
        if (not allow_pos0_neg):
            pos0_min = max(0.0, pos0_min)
            pos0_maxin = max(pos0_maxin, 0.0)
        self.pos0_min = pos0_min
        self.pos1_min = pos1_min
        self.pos0_max = calc_upper_bound(pos0_maxin)
        self.pos1_max = calc_upper_bound(pos1_maxin)

    def calc_lut_2d(self):
        """Calculate the LUT and return the LUT-builder object
        """
        cdef:
            Py_ssize_t bins0=self.bins[0], bins1=self.bins[1], size = self.size
            position_t[:, :, ::1] cpos = numpy.ascontiguousarray(self.pos, dtype=position_d)
            position_t[:, ::1] v8 = numpy.empty((4,2), dtype=position_d)
            mask_t[:] cmask = self.cmask
            bint check_mask = False, allow_pos0_neg = self.allow_pos0_neg, chiDiscAtPi = self.chiDiscAtPi
            position_t min0 = 0, max0 = 0, min1 = 0, max1 = 0, inv_area = 0
            position_t pos0_min = 0, pos0_max = 0, pos1_min = 0, pos1_max = 0, pos0_maxin = 0, pos1_maxin = 0
            position_t fbin0_min = 0, fbin0_max = 0, fbin1_min = 0, fbin1_max = 0
            position_t a0 = 0, a1 = 0, b0 = 0, b1 = 0, c0 = 0, c1 = 0, d0 = 0, d1 = 0
            position_t center0 = 0.0, center1 = 0.0, area, width, height,   
            position_t delta0, delta1, new_width, new_height, new_min0, new_max0, new_min1, new_max1
            Py_ssize_t bin0_max = 0, bin0_min = 0, bin1_max = 0, bin1_min = 0, i = 0, j = 0, idx = 0
            Py_ssize_t ioffset0, ioffset1, w0, w1, bw0=15, bw1=15, nwarn=NUM_WARNING
            buffer_t[::1] linbuffer = numpy.empty(256, dtype=buffer_d)
            buffer_t[:, ::1] buffer = numpy.asarray(linbuffer[:(bw0+1)*(bw1+1)]).reshape((bw0+1,bw1+1))
            double foffset0, foffset1, sum_area, loc_area
            SparseBuilder builder = SparseBuilder(bins1*bins0, block_size=8, heap_size=size)
            
        if self.cmask is not None:
            check_mask = True
            cmask = self.cmask

        pos0_max = self.pos0_max
        pos0_min = self.pos0_min
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
                area = _recenter(v8, chiDiscAtPi)
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
                
                if (max0 < pos0_min) or (min0 > pos0_maxin) or (max1 < pos1_min) or (min1 > pos1_maxin):
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
                            linbuffer = numpy.empty((w0+1)*(w1+1), dtype=buffer_d)
                            buffer = numpy.asarray(linbuffer).reshape((w0+1,w1+1))
                            logger.debug("malloc  %s->%s and %s->%s", w0, bw0, w1, bw1) 
                    else:
                        with gil:
                            buffer = numpy.asarray(linbuffer[:(w0+1)*(w1+1)]).reshape((w0+1,w1+1))
                            logger.debug("reshape %s->%s and %s->%s", w0, bw0, w1, bw1)
                    bw0 = w0
                    bw1 = w1
                buffer[:, :] = 0.0
                
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
    
                area = 0.5 * ((c1 - a1) * (d0 - b0) - (c0 - a0) * (d1 - b1))
                if area == 0.0:
                    continue
                inv_area = 1.0 / area
                sum_area = 0.0
                for i in range(w0):
                    for j in range(w1):
                        loc_area = buffer[i, j]
                        sum_area += loc_area
                        builder.cinsert((ioffset0 + i)*bins1 + ioffset1 + j, idx, loc_area * inv_area)

                if fabs(area - sum_area)*inv_area > 1e-3:
                    nwarn -=1
                    if nwarn>0:
                        with gil:            
                            logger.info(f"Invstigate idx {idx}, area {area} {sum_area}, {numpy.asarray(v8)}, {w0}, {w1}")
        if nwarn<NUM_WARNING:
            logger.info(f"Total number of spurious pixels: {NUM_WARNING - nwarn} / {size} total")
      
        return builder

    
    