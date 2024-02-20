# -*- coding: utf-8 -*-
#cython: embedsignature=True, language_level=3, binding=True
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False,
## This is for developping:
##cython: profile=True, warn.undeclared=True, warn.unused=True, warn.unused_result=False, warn.unused_arg=True
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2020 European Synchrotron Radiation Facility, Grenoble, France
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


"""
Extenstion with Cython implementation of pyFAI.utils.mathutil
"""

__author__ = "Jerome Kieffer"
__license__ = "MIT"
__date__ = "20/02/2024"
__copyright__ = "2024-2024, ESRF"
__contact__ = "jerome.kieffer@esrf.fr"

from libc.stdint cimport int8_t, uint8_t, int16_t, uint16_t, \
                         int32_t, uint32_t, int64_t, uint64_t
import numpy

def is_far_from_group_cython(pt, list lst_pts, double d2):
    """
    Tells if a point is far from a group of points, distance greater than d2 (distance squared)

    :param pt: point of interest
    :param lst_pts: list of points
    :param d2: minimum distance squarred
    :return: True If the point is far from all others.

    """
    cdef:
        double p0, p1, q0, q1, dsq, d0, d1
    p0, p1 = pt
    for (q0, q1) in lst_pts:
        d0 = p0 - q0
        d1 = p1 - q1
        dsq = d0*d0 + d1*d1
        if dsq <= d2:
            return False
    return True


def build_qmask(tth_array,
                tth_min,
                tth_max,
                mask=None):
    """Build a qmask, array with the index of the reflection for each pixel.
    qmask==-1 is for uninteresting pixels
    qmask==-2 is for masked ones
    The count array (same size as tth_min or tth_max) holds the number of pixel per reflection.

    :param tth_array: 2D array with 2th positions
    :param tth_min: 1D array with lower bound for each reflection, same size as tth_max
    :param tth_max: 1D array with upper bound for each reflection, same size as tth_min
    :param mask: marked pixel are invalid
    :return: qmask, count
    """
    cdef Py_ssize_t nref = tth_min.size
    assert tth_max.size == nref, "tth_max.size == tth_min.size"
    cdef:
        int i, r, n = tth_array.size
        double tthv, lb, ub
        double[::1] ctth = numpy.ascontiguousarray(tth_array.ravel(), dtype=numpy.float64)
        double[::1] ctth_min = numpy.ascontiguousarray(tth_min.ravel(), dtype=numpy.float64)
        double[::1] ctth_max = numpy.ascontiguousarray(tth_max.ravel(), dtype=numpy.float64)
        int32_t[::1] qmask = numpy.zeros(n, dtype=numpy.int32)
        int32_t[::1] count = numpy.zeros(nref, dtype=numpy.int32)
        int8_t[::1] cmask
        bint do_mask=False
    if mask is not None:
        assert mask.size == n, "mask.size == tth_array.size"
        cmask = numpy.ascontiguousarray(mask.ravel(), dtype=numpy.int8)
        do_mask = True

    with nogil:
        for i in range(n):
            if do_mask and cmask[i]:
                qmask[i] = -2
                continue
            tthv = ctth[i]
            for r in range(nref):
                lb = ctth_min[r]
                ub = ctth_max[r]
                if tthv < lb:
                    qmask[i] = -1
                    break
                if lb <= tthv < ub:
                    qmask[i] = r
                    count[r] += 1
                    break
            else:
                qmask[i] = -1

    return numpy.asarray(qmask).reshape(tth_array.shape), numpy.asarray(count)
