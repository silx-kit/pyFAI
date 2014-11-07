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
__date__ = "21/10/2014"
__copyright__ = "2011-2014, ESRF"
__contact__ = "jerome.kieffer@esrf.fr"

import cython
import numpy
cimport numpy
from cython.parallel import prange
ctypedef fused float32_64:
    cython.float
    cython.double
from libc.math cimport floor,ceil

import logging
logger = logging.getLogger("pyFAI.bilinear")

from .utils import timeit

cdef class Bilinear:
    """Bilinear interpolator for finding max"""

    cdef float[:,:] data
    cdef float maxi, mini
    cdef size_t d0_max, d1_max, r

    def __cinit__(self, data not None):
        assert data.ndim == 2
        self.d0_max = data.shape[0] - 1
        self.d1_max = data.shape[1] - 1
        self.maxi = data.max()
        self.mini = data.min()
        self.data = numpy.ascontiguousarray(data, dtype=numpy.float32)

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
        with nogil:
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
                res = self.data[i0,j0]
            elif i0 == i1:
                res = (self.data[i0,j0] * (y1 - d1)) + (self.data[i0,j1] * (d1 - y0))
            elif j0 == j1:
                res = (self.data[i0,j0] * (x1 - d0)) + (self.data[i1,j0] * (d0 - x0))
            else:
                res = (self.data[i0,j0] * (x1 - d0) * (y1 - d1))  \
                    + (self.data[i1,j0] * (d0 - x0) * (y1 - d1))  \
                    + (self.data[i0,j1] * (x1 - d0) * (d1 - y0))  \
                    + (self.data[i1,j1] * (d0 - x0) * (d1 - y0))
        return - res

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def local_maxi(self, x, int w=1):
        """
        Return the local maximum ... with sub-pixel refinement

        @param x: 2-tuple of integers
        @param w: half with of the window: 1 or 2 are advised
        @return: 2-tuple of int with the nearest local maximum


        Sub-pixel refinement:
        Second order Taylor expansion of the function; first derivative is null
        delta = x-i = -Inverse[Hessian].gradient

        if Hessian is singular or |delta|>1: use a center of mass.

        """
        cdef:
            int current0 = x[0]
            int current1 = x[1]
            int i0, i1, start0, stop0, start1, stop1, new0, new1, cnt=0, width0=w, width1=w
            float tmp, value, current_value, sum0=0,  sum1=0, sum=0
            float a00, a01, a02, a10, a11, a12, a20, a21, a22
            float d00, d11, d01, denom, delta0, delta1

        value = self.data[current0,current1]
        current_value = value-1.0
        new0,new1 = current0,current1
        with nogil:
            while value>current_value:
                current_value=value
                cnt+=1
                if current0 < width0:
                    start0 = 0
                else:
                    start0 = current0 - width0
                if current0 >= self.d0_max - width0:
                    stop0 = self.d0_max
                else:
                    stop0 = current0 + width0
                if current1 < width1:
                    start1 = 0
                else:
                    start1 = current1 - width1
                if current1 >= self.d1_max - width1:
                    stop1=self.d1_max
                else:
                    stop1 = current1 + width1
                for i0 in range(start0, stop0+1):
                    for i1 in range(start1, stop1+1):
                        tmp=self.data[i0,i1]
                        if tmp>current_value:
                            new0,new1=i0,i1
                            value = tmp
                current0,current1=new0,new1

        if (stop0>current0) and (current0>start0) and (stop1>current1) and (current1>start1):
            # Use second order polynomial Taylor expansion
            a00 = self.data[current0-1,current1-1]
            a01 = self.data[current0-1,current1  ]
            a02 = self.data[current0-1,current1+1]
            a10 = self.data[current0  ,current1-1]
            a11 = self.data[current0  ,current1  ]
            a12 = self.data[current0  ,current1+1]
            a20 = self.data[current0+1,current1-1]
            a21 = self.data[current0+1,current1  ]
            a22 = self.data[current0+1,current1-1]
            d00 = a12 - 2.0*a11 + a10
            d11 = a21 - 2.0*a11 + a01
            d01 = (a00 - a02 - a20 + a22)/4.0
            denom = 2.0*(d00*d11-d01*d01)
            if abs(denom)<1e-10:
                logger.debug("Singular determinant, Hessian undefined")
            else:
                delta0 = ((a12 - a10)*d01 + (a01 - a21)*d11)/denom
                delta1 = ((a10 - a12)*d00 + (a21 - a01)*d01)/denom
                if abs(delta0)<=1.0 and abs(delta1)<=1.0: #Result is OK if lower than 0.5.
                    return (delta0 + float(current0), delta1 + float(current1))
                else:
                    logger.debug("Failed to find root using second order expansion")
        #refinement of the position by a simple center of mass of the last valid region used
        for i0 in range(start0, stop0+1):
            for i1 in range(start1, stop1+1):
                tmp = self.data[i0,i1]
                sum0 += tmp * i0
                sum1 += tmp * i1
                sum += tmp
        if sum>0:
           #print current0,current1,sum0/sum,sum1/sum
           return (sum0/sum,sum1/sum)
        else:
            return (current0,current1)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def calc_cartesian_positions(float32_64[:] d1, float32_64[:] d2, float[:,:,:,:] pos):
    """
    Calculate the Cartesian position for array of position (d1, d2)
    with pixel coordinated stored in array pos
    This is bilinear interpolation

    @param d1: position in dim1
    @param d2: position in dim2
    @param pos: array with position of pixels corners
    """
    cdef int i, p1, p2, size = d1.size
    cdef float delta1, delta2, A1, A2, B1, B2, C1, C2, D1, D2
    cdef numpy.ndarray[numpy.float32_t, ndim = 1] out1 = numpy.empty(size, dtype=numpy.float32)
    cdef numpy.ndarray[numpy.float32_t, ndim = 1] out2 = numpy.empty(size, dtype=numpy.float32)
    assert size == d2.size
    for i in prange(size, nogil=True):
        p1 = <int> d1[i]
        delta1 = d1[i]-floor(d1[i])
        p2 = <int> d2[i]
        delta2 = d2[i]-floor(d2[i])
        A1 = pos[p1,p2, 0, 1]
        A2 = pos[p1,p2, 0, 2]
        B1 = pos[p1,p2, 1, 1]
        B2 = pos[p1,p2, 1, 2]
        C1 = pos[p1,p2, 2, 1]
        C2 = pos[p1,p2, 2, 2]
        D1 = pos[p1,p2, 3, 1]
        D2 = pos[p1,p2, 3, 2]

        out1[i] = 0.5 * ((A1 + D1) * (1.0 - delta1) + delta1 * (B1 + C1))
        out2[i] = 0.5 * ((A2 + B2) * (1.0 - delta2) + delta2 * (C2 + D2))
    return out1, out2

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def convert_corner_2D_to_4D(int ndim, float32_64[:,:] d1, float32_64[:,:] d2):
    """
    Convert 2 array of corner position into a 4D array of pixel corner coodinates

    @param ndim: 2d or 3D output
    @param d1: 2D position in dim1 (shape +1)
    @param d2: 2D position in dim2 (shape +1)
    @param pos: 4D array with position of pixels corners
    """
    cdef int shape0, shape1, i, j
    shape0 = d1.shape[0] - 1 #edges position are n+1 compared to number of pixels
    shape1 = d1.shape[1] - 1
    cdef numpy.ndarray[numpy.float32_t, ndim = 4] pos = numpy.zeros((shape0, shape1, 4, ndim), dtype=numpy.float32)
#    assert d1.shape == d2.shape
    for i in prange(shape0, nogil=True):
        for j in range(shape1):
            pos[i,j, 0, ndim-2] = d1[i, j]
            pos[i,j, 0, ndim-1] = d2[i, j]
            pos[i,j, 1, ndim-2] = d1[i+1, j]
            pos[i,j, 1, ndim-1] = d2[i+1, j]
            pos[i,j, 2, ndim-2] = d1[i+1, j+1]
            pos[i,j, 2, ndim-1] = d2[i+1, j+1]
            pos[i,j, 3, ndim-2] = d1[i, j+1]
            pos[i,j, 3, ndim-1] = d2[i, j+1]
    return pos