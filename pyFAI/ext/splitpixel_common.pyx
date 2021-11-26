# -*- coding: utf-8 -*-
#cython: embedsignature=True, language_level=3, binding=True
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False,
## This is for developping
## cython: profile=True, warn.undeclared=True, warn.unused=True, warn.unused_result=False, warn.unused_arg=True
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
__date__ = "26/11/2021"
__status__ = "stable"
__license__ = "MIT"


include "regrid_common.pxi"


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
