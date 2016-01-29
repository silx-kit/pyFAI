#!/usr/bin/env python
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

__doc__ = """
Cython module to reconstruct the masked values of an image"""
__author__ = "Jerome Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "20140401"
__status__ = "stable"
__license__ = "GPLv3+"


import cython
import numpy
cimport numpy
from libc.math cimport sqrt
from cython.parallel import prange


@cython.cdivision(True)
cdef float invert_distance(size_t i0, size_t i1, size_t p0, size_t p1) nogil:
    return 1. / sqrt(<float> ((i0 - p0) ** 2 + (i1 - p1) ** 2))


@cython.cdivision(True)
@cython.boundscheck(False)
cdef inline float processPoint(float[:, :] data, 
                               numpy.int8_t[:, :] mask, 
                               size_t p0,
                               size_t p1, 
                               size_t d0, 
                               size_t d1)nogil:
    cdef:
        size_t dist = 0, i = 0
        float sum = 0.0, count = 0.0, invdst = 0.0
        bint found = 0
        size_t start0 = p0, stop0 = p0, start1 = p1, stop1 = p1
    while not found:
        dist += 1
        if start0 > 0:
            start0 = p0 - dist
        else:
            start0 = 0
        if stop0 < d0 - 1:
            stop0 = p0 + dist
        else:
            stop0 = d0 - 1
        if start1 > 0:
            start1 = p1 - dist
        else:
            start1 = 0
        if stop1 < d1 - 1:
            stop1 = p1 + dist
        else:
            stop1 = d1 - 1
        for i in range(start0, stop0 + 1):
            if mask[i, start1] == 0:
                invdst = invert_distance(i, start1, p0, p1)
                count += invdst
                sum += invdst * data[i, start1]
            if mask[i, stop1] == 0:
                invdst = invert_distance(i, stop1, p0, p1)
                count += invdst
                sum += invdst * data[i, stop1]
        for i in range(start1 + 1, stop1):
            if mask[start0, i] == 0:
                invdst = invert_distance(start0, i, p0, p1)
                count += invdst
                sum += invdst * data[start0, i]
            if mask[stop0, i] == 0:
                invdst = invert_distance(stop0, i, p0, p1)
                count += invdst
                sum += invdst * data[stop0, i]
        if count > 0:
            found = 1
    return sum / count


@cython.boundscheck(False)
@cython.wraparound(False)
def reconstruct(numpy.ndarray data not None, numpy.ndarray mask=None, dummy=None, delta_dummy=None):
    """
    reconstruct missing part of an image (tries to be continuous)
    
    @param data: the input image 
    @parma mask: where data should be reconstructed.
    @param dummy: value of the dummy (masked out) data
    @param delta_dummy: precision for dummy values
    
    @return: reconstructed image.
    """
    assert data.ndim == 2
    cdef: 
        ssize_t d0 = data.shape[0]
        ssize_t d1 = data.shape[1]
        float[:, :] cdata
    data = numpy.ascontiguousarray(data, dtype=numpy.float32)
    cdata = data
    if mask is not None:
        mask = numpy.ascontiguousarray(mask, dtype=numpy.int8)
    else:
        mask = numpy.zeros((d0, d1), dtype=numpy.int8)
    if dummy is not None:
        if delta_dummy is None:
            mask += (data == dummy)
        else:
            mask += (abs(data - dummy) <= delta_dummy)
    cdef numpy.int8_t[:, :] cmask = mask.astype(numpy.int8)
    assert d0 == mask.shape[0]
    assert d1 == mask.shape[1]
    cdef numpy.ndarray[numpy.float32_t, ndim = 2]out = numpy.zeros_like(data)
    out += data
    out[mask.astype(bool)] = 0

    cdef ssize_t p0, p1, i, l
    for p0 in prange(d0, nogil=True, schedule="guided"):
        for p1 in range(d1):
            if cmask[p0, p1]:
                out[p0, p1] += processPoint(cdata, cmask, p0, p1, d0, d1)
    return out

