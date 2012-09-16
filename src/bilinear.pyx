# -*- coding: utf8 -*-
#
#    Project: Azimuthal integration
#             https://forge.epn-campus.eu/projects/azimuthal
#
#    File: "$Id$"
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

__author__ = "Jerome Kieffer"
__license__ = "GPLv3"
__date__ = "21/12/2011"
__copyright__ = "2011, ESRF"
__contact__ = "jerome.kieffer@esrf.fr"

import cython
import numpy
cimport numpy

from libc.math cimport floor,ceil
#from libc.stdlib cimport  calloc,malloc,memcpy


cdef class bilinear:
    """Bilinear interpolator for finding max"""

    cdef float[:] data
    cdef float maxi, mini
    cdef size_t d0_max, d1_max, r
#
#    def __dealloc__(self):
#        free(self.data)

    def __cinit__(self, numpy.ndarray data not None):
        assert data.ndim == 2
        self.d0_max = data.shape[0] - 1
        self.d1_max = data.shape[1] - 1
        self.r = data.shape[1]
        self.maxi = data.max()
        self.mini = data.min()
        #self.data = < float *> malloc(data.size * sizeof(float))
        #cdef numpy.ndarray[numpy.float32_t, ndim = 2] data2 = numpy.ascontiguousarray(data, dtype=numpy.float32)
        #memcpy(self.data, data2.data, data.size * sizeof(float))
        self.data = numpy.ascontiguousarray(data.ravel(), dtype=numpy.float32)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def f_cy(self, x):
        """
        Function f((y,x)) where f is a continuous function (y,x) are pixel coordinates
        @param x: 2-tuple of float
        @return: Interpolated signal from the image (negative for minimizer)

        """
        cdef float d0 = x[0]
        cdef float d1 = x[1]
        cdef int i0, i1, j0, j1
        cdef float x0, x1, y0, y1, res
        x0 = floor(d0)
        x1 = ceil(d0)
        y0 = floor(d1)
        y1 = ceil(d1)
        i0 = < int > x0
        i1 = < int > x1
        j0 = < int > y0
        j1 = < int > y1
        if d0 < 0:
            res = self.mini + d0
        elif d1 < 0:
            res = self.mini + d1
        elif d0 > self.d0_max:
            res = self.mini - d0 + self.d0_max
        elif d1 > self.d1_max:
            res = self.mini - d1 + self.d1_max
        elif (i0 == i1) and (j0 == j1):
            res = self.data[i0 * self.r + j0]
        elif i0 == i1:
            res = (self.data[i0 * self.r + j0] * (y1 - d1)) + (self.data[i0 * self.r + j1] * (d1 - y0))
        elif j0 == j1:
            res = (self.data[i0 * self.r + j0] * (x1 - d0)) + (self.data[i1 * self.r + j0] * (d0 - x0))
        else:
            res = (self.data[i0 * self.r + j0] * (x1 - d0) * (y1 - d1))  \
                + (self.data[i1 * self.r + j0] * (d0 - x0) * (y1 - d1))  \
                + (self.data[i0 * self.r + j1] * (x1 - d0) * (d1 - y0))  \
                + (self.data[i1 * self.r + j1] * (d0 - x0) * (d1 - y0))
        return - res
