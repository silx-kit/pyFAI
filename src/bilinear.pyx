# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
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

__author__ = "Jerome Kieffer"
__license__ = "GPLv3+"
__date__ = "07/04/2015"
__copyright__ = "2011-2015, ESRF"
__contact__ = "jerome.kieffer@esrf.fr"

import cython
import numpy
cimport numpy
from cython cimport floating  # float32 or float64
from cython.parallel import prange

import logging
logger = logging.getLogger("pyFAI.bilinear")

from .utils import timeit
include "bilinear.pxi"


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def calc_cartesian_positions(floating[:] d1, floating[:] d2, float[:, :, :, :] pos):
    """
    Calculate the Cartesian position for array of position (d1, d2)
    with pixel coordinated stored in array pos
    This is bilinear interpolation

    @param d1: position in dim1
    @param d2: position in dim2
    @param pos: array with position of pixels corners
    """
    cdef:
        int i, p1, p2, dim1, dim2, size = d1.size
        float delta1, delta2, f1, f2, A1, A2, B1, B2, C1, C2, D1, D2
        numpy.ndarray[numpy.float32_t, ndim = 1] out1 = numpy.zeros(size, dtype=numpy.float32)
        numpy.ndarray[numpy.float32_t, ndim = 1] out2 = numpy.zeros(size, dtype=numpy.float32)

    dim1 = pos.shape[0]
    dim2 = pos.shape[1]
    assert size == d2.size

    for i in prange(size, nogil=True):
        f1 = floor(d1[i])
        f2 = floor(d2[i])

        p1 = <int> f1
        p2 = <int> f2

        delta1 = d1[i] - f1
        delta2 = d2[i] - f2

        if p1 < 0:
            with gil:
                print("f1= %s"%f1)

        if p1 < 0:
            with gil:
                print("f2= %s"%f2)

        if p1 >= dim1:
            if p1>dim1:
                with gil:
                    print("d1= %s, f1=%s, p1=%s, delta1=%s" % (d1[i], f1, p1, delta1))
            p1 = dim1 - 1
            delta1 = d1[i] - p1

        if p2 >= dim2:
            if p2>dim2:
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

    return out1, out2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def convert_corner_2D_to_4D(int ndim, floating[:, :] d1, floating[:, :] d2):
    """
    Convert 2 array of corner position into a 4D array of pixel corner coordinates

    @param ndim: 2d or 3D output
    @param d1: 2D position in dim1 (shape +1)
    @param d2: 2D position in dim2 (shape +1)
    @param pos: 4D array with position of pixels corners
    """
    cdef int shape0, shape1, i, j
    #  edges position are n+1 compared to number of pixels
    shape0 = d1.shape[0] - 1
    shape1 = d1.shape[1] - 1
    cdef numpy.ndarray[numpy.float32_t, ndim = 4] pos = numpy.zeros((shape0, shape1, 4, ndim), dtype=numpy.float32)
#    assert d1.shape == d2.shape
    for i in prange(shape0, nogil=True):
        for j in range(shape1):
            pos[i, j, 0, ndim - 2] += d1[i, j]
            pos[i, j, 0, ndim - 1] += d2[i, j]
            pos[i, j, 1, ndim - 2] += d1[i + 1, j]
            pos[i, j, 1, ndim - 1] += d2[i + 1, j]
            pos[i, j, 2, ndim - 2] += d1[i + 1, j + 1]
            pos[i, j, 2, ndim - 1] += d2[i + 1, j + 1]
            pos[i, j, 3, ndim - 2] += d1[i, j + 1]
            pos[i, j, 3, ndim - 1] += d2[i, j + 1]
    return pos