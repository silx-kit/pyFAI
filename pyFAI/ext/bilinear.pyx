# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
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


__author__ = "Jerome Kieffer"
__license__ = "MIT"
__date__ = "31/05/2016"
__copyright__ = "2011-2015, ESRF"
__contact__ = "jerome.kieffer@esrf.fr"

import cython
import numpy
cimport numpy
from cython cimport floating
from cython.parallel import prange

import logging
logger = logging.getLogger("pyFAI.ext.bilinear")

from ..decorators import timeit


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def calc_cartesian_positions(floating[::1] d1, floating[::1] d2,
                             float[:, :, :, ::1] pos,
                             bint is_flat=True):
    """
    Calculate the Cartesian position for array of position (d1, d2)
    with pixel coordinated stored in array pos
    This is bilinear interpolation

    @param d1: position in dim1
    @param d2: position in dim2
    @param pos: array with position of pixels corners
    @return 3-tuple of position.
    """
    cdef:
        int i, p1, p2, dim1, dim2, size = d1.size
        float delta1, delta2, f1, f2, A0, A1, A2, B0, B1, B2, C1, C0, C2, D0, D1, D2
        numpy.ndarray[numpy.float32_t, ndim = 1] out1 = numpy.zeros(size, dtype=numpy.float32)
        numpy.ndarray[numpy.float32_t, ndim = 1] out2 = numpy.zeros(size, dtype=numpy.float32)
        numpy.ndarray[numpy.float32_t, ndim = 1] out3
    if not is_flat:
        out3 = numpy.zeros(size, dtype=numpy.float32)
    dim1 = pos.shape[0]
    dim2 = pos.shape[1]
    assert size == d2.size

    for i in prange(size, nogil=True, schedule="static"):
        f1 = floor(d1[i])
        f2 = floor(d2[i])

        p1 = <int> f1
        p2 = <int> f2

        delta1 = d1[i] - f1
        delta2 = d2[i] - f2

        if p1 < 0:
            with gil:
                print("f1= %s" % f1)

        if p1 < 0:
            with gil:
                print("f2= %s" % f2)

        if p1 >= dim1:
            if p1 > dim1:
                with gil:
                    print("d1= %s, f1=%s, p1=%s, delta1=%s" % (d1[i], f1, p1, delta1))
            p1 = dim1 - 1
            delta1 = d1[i] - p1

        if p2 >= dim2:
            if p2 > dim2:
                with gil:
                    print("d2= %s, f2=%s, p2=%s, delta2=%s" % (d2[i], f2, p2, delta2))
            p2 = dim2 - 1
            delta2 = d2[i] - p2

        A1 = pos[p1, p2, 0, 1]
        A2 = pos[p1, p2, 0, 2]
        B1 = pos[p1, p2, 1, 1]
        B2 = pos[p1, p2, 1, 2]
        C1 = pos[p1, p2, 2, 1]
        C2 = pos[p1, p2, 2, 2]
        D1 = pos[p1, p2, 3, 1]
        D2 = pos[p1, p2, 3, 2]
        if not is_flat:
            A0 = pos[p1, p2, 0, 0]
            B0 = pos[p1, p2, 1, 0]
            C0 = pos[p1, p2, 2, 0]
            D0 = pos[p1, p2, 3, 0]
            out3[i] += A0 * (1.0 - delta1) * (1.0 - delta2) \
                + B0 * delta1 * (1.0 - delta2) \
                + C0 * delta1 * delta2 \
                + D0 * (1.0 - delta1) * delta2

        # A and D are on the same:  dim1 (Y)
        # A and B are on the same:  dim2 (X)
        # nota: += is needed as well as numpy.zero because of prange: avoid reduction
        out1[i] += A1 * (1.0 - delta1) * (1.0 - delta2) \
            + B1 * delta1 * (1.0 - delta2) \
            + C1 * delta1 * delta2 \
            + D1 * (1.0 - delta1) * delta2
        out2[i] += A2 * (1.0 - delta1) * (1.0 - delta2) \
            + B2 * delta1 * (1.0 - delta2) \
            + C2 * delta1 * delta2 \
            + D2 * (1.0 - delta1) * delta2
    if is_flat:
        return out1, out2, None
    else:
        return out1, out2, out3


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def convert_corner_2D_to_4D(int ndim,
                            floating[:, ::1] d1 not None,
                            floating[:, ::1] d2 not None,
                            floating[:, ::1] d3=None):
    """
    Convert 2 (or 3) arrays of corner position into a 4D array of pixel corner coordinates

    @param ndim: 2d or 3D output
    @param d1: 2D position in dim1 (shape +1)
    @param d2: 2D position in dim2 (shape +1)
    @param d3: 2D position in dim3 (z) (shape +1)
    @return: pos 4D array with position of pixels corners
    """
    cdef int shape0, shape1, i, j
    #  edges position are n+1 compared to number of pixels
    shape0 = d1.shape[0] - 1
    shape1 = d2.shape[1] - 1
    assert d1.shape[0] == d2.shape[0]
    assert d1.shape[1] == d2.shape[1]
    cdef numpy.ndarray[numpy.float32_t, ndim = 4] pos = numpy.zeros((shape0, shape1, 4, ndim), dtype=numpy.float32)
    for i in prange(shape0, nogil=True, schedule="static"):
        for j in range(shape1):
            pos[i, j, 0, ndim - 2] += d1[i, j]
            pos[i, j, 0, ndim - 1] += d2[i, j]
            pos[i, j, 1, ndim - 2] += d1[i + 1, j]
            pos[i, j, 1, ndim - 1] += d2[i + 1, j]
            pos[i, j, 2, ndim - 2] += d1[i + 1, j + 1]
            pos[i, j, 2, ndim - 1] += d2[i + 1, j + 1]
            pos[i, j, 3, ndim - 2] += d1[i, j + 1]
            pos[i, j, 3, ndim - 1] += d2[i, j + 1]
    if (d3 is not None) and (ndim == 3):
        assert d1.shape[0] == d3.shape[0]
        assert d1.shape[1] == d3.shape[1]
        for i in prange(shape0, nogil=True, schedule="static"):
            for j in range(shape1):
                pos[i, j, 0, 0] += d3[i, j]
                pos[i, j, 1, 0] += d3[i + 1, j]
                pos[i, j, 2, 0] += d3[i + 1, j + 1]
                pos[i, j, 3, 0] += d3[i, j + 1]
    return pos

include "bilinear.pxi"
