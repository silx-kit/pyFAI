#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) 2012-2016 European Synchrotron Radiation Facility, France
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

__doc__ = """A module to relabel regions"""
__author__ = "Jerome Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "31/05/2016"
__status__ = "stable"
__license__ = "MIT"
import cython
import numpy
cimport numpy


@cython.boundscheck(False)
@cython.wraparound(False)
def countThem(numpy.ndarray label not None, \
              numpy.ndarray data not None, \
              numpy.ndarray blured not None):
    """
    @param label: 2D array containing labeled zones
    @param data: 2D array containing the raw data
    @param blured: 2D array containing the blured data
    @return: 2D arrays containing:
        * count pixels in labelled zone: label == index).sum()
        * max of data in that zone:      data[label == index].max()
        * max of blured in that zone:    blured[label == index].max()
        * data-blured where data is max.
    """
    cdef:
        numpy.uint32_t[:] clabel = numpy.ascontiguousarray(label.ravel(), dtype=numpy.uint32)
        float[:] cdata = numpy.ascontiguousarray(data.ravel(), dtype=numpy.float32)
        float[:] cblured = numpy.ascontiguousarray(blured.ravel(), dtype=numpy.float32)
        size_t maxLabel = label.max()
        numpy.ndarray[numpy.uint_t, ndim = 1] count = numpy.zeros(maxLabel + 1, dtype=numpy.uint)
        numpy.ndarray[numpy.float32_t, ndim = 1] maxData = numpy.zeros(maxLabel + 1, dtype=numpy.float32)
        numpy.ndarray[numpy.float32_t, ndim = 1] maxBlured = numpy.zeros(maxLabel + 1, dtype=numpy.float32)
        numpy.ndarray[numpy.float32_t, ndim = 1] maxDelta = numpy.zeros(maxLabel + 1, dtype=numpy.float32)
        int s, i, idx
        float d, b
    s = label.size
    assert s == cdata.size
    assert s == cblured.size
    with nogil:
        for i in range(s):
            idx = clabel[i]
            d = cdata[i]
            b = cblured[i]
            count[idx] += 1
            if d > maxData[idx]:
                maxData[idx] = d
                maxDelta[idx] = d - b
            if b > maxBlured[idx]:
                maxBlured[idx] = b
    return count, maxData, maxBlured, maxDelta



