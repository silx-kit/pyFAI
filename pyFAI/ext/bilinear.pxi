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

from libc.math cimport floor, ceil

cdef class Bilinear:
    """Bilinear interpolator for finding max.
    
    Instance attribute defined in pxd file 
    """
    cdef:
        readonly float[:, ::1] data
        readonly float maxi, mini
        readonly size_t width, height

    cpdef size_t cp_local_maxi(self, size_t)
    cdef size_t c_local_maxi(self, size_t) nogil

    def __cinit__(self, data not None):
        assert data.ndim == 2
        self.height = data.shape[0]
        self.width = data.shape[1]
        self.maxi = data.max()
        self.mini = data.min()
        self.data = numpy.ascontiguousarray(data, dtype=numpy.float32)
    
    def __dealloc__(self):
        self.data = None

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def f_cy(self, x):
        """
        Function f((y,x)) where f is a continuous function (y,x) are pixel coordinates
        @param x: 2-tuple of float
        @return: Interpolated signal from the image (negative for minimizer)

        """
        cdef:
            float d0 = x[0]
            float d1 = x[1]
            int i0, i1, j0, j1
            float x0, x1, y0, y1, res
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
            elif d0 > (self.height - 1):
                res = self.mini - d0 + self.height - 1
            elif d1 > self.width - 1:
                res = self.mini - d1 + self.width - 1
            elif (i0 == i1) and (j0 == j1):
                res = self.data[i0, j0]
            elif i0 == i1:
                res = (self.data[i0, j0] * (y1 - d1)) + (self.data[i0, j1] * (d1 - y0))
            elif j0 == j1:
                res = (self.data[i0, j0] * (x1 - d0)) + (self.data[i1, j0] * (d0 - x0))
            else:
                res = (self.data[i0, j0] * (x1 - d0) * (y1 - d1))  \
                    + (self.data[i1, j0] * (d0 - x0) * (y1 - d1))  \
                    + (self.data[i0, j1] * (x1 - d0) * (d1 - y0))  \
                    + (self.data[i1, j1] * (d0 - x0) * (d1 - y0))
        return - res

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def local_maxi(self, x):
        """
        Return the local maximum ... with sub-pixel refinement

        @param x: 2-tuple of integers
        @param w: half with of the window: 1 or 2 are advised
        @return: 2-tuple of float with the nearest local maximum


        Sub-pixel refinement:
        Second order Taylor expansion of the function; first derivative is null
        delta = x-i = -Inverse[Hessian].gradient

        if Hessian is singular or |delta|>1: use a center of mass.

        """
        cdef:
            int res, current0, current1
            int i0, i1
            float tmp, sum0 = 0, sum1 = 0, sum = 0
            float a00, a01, a02, a10, a11, a12, a20, a21, a22
            float d00, d11, d01, denom, delta0, delta1
            
        res = self.c_local_maxi(round(x[0]) * self.width + round(x[1]))

        current0 = res // self.width
        current1 = res % self.width
        if (current0 > 0) and (current0 < self.height - 1) and (current1 > 0) and (current1 < self.width - 1):
            # Use second order polynomial Taylor expansion
            a00 = self.data[current0 - 1, current1 - 1]
            a01 = self.data[current0 - 1, current1    ]
            a02 = self.data[current0 - 1, current1 + 1]
            a10 = self.data[current0    , current1 - 1]
            a11 = self.data[current0    , current1    ]
            a12 = self.data[current0    , current1 + 1]
            a20 = self.data[current0 + 1, current1 - 1]
            a21 = self.data[current0 + 1, current1    ]
            a22 = self.data[current0 + 1, current1 - 1]
            d00 = a12 - 2.0 * a11 + a10
            d11 = a21 - 2.0 * a11 + a01
            d01 = (a00 - a02 - a20 + a22) / 4.0
            denom = 2.0 * (d00 * d11 - d01 * d01)
            if abs(denom) < 1e-10:
                logger.debug("Singular determinant, Hessian undefined")
            else:
                delta0 = ((a12 - a10) * d01 + (a01 - a21) * d11) / denom
                delta1 = ((a10 - a12) * d00 + (a21 - a01) * d01) / denom
                if abs(delta0) <= 1.0 and abs(delta1) <= 1.0:
                    # Result is OK if lower than 0.5.
                    return (delta0 + float(current0), delta1 + float(current1))
                else:
                    logger.debug("Failed to find root using second order expansion")
            # refinement of the position by a simple center of mass of the last valid region used
            for i0 in range(current0 - 1, current0 + 2):
                for i1 in range(current1 - 1, current1 + 2):
                    tmp = self.data[i0, i1]
                    sum0 += tmp * i0
                    sum1 += tmp * i1
                    sum += tmp
            if sum > 0:
                return (sum0 / sum, sum1 / sum)
                
        return (float(current0), float(current1))

    cpdef size_t cp_local_maxi(self, size_t x):
        return self.c_local_maxi(x)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef size_t c_local_maxi(self, size_t x) nogil:
        """
        Return the local maximum ... without sub-pixel refinement

        @param x: start index
        @param w: half with of the window: 1 or 2 are advised
        @return: local maximum index

        """
        cdef:
            int current0 = x // self.width
            int current1 = x % self.width
            int i0, i1, start0, stop0, start1, stop1, new0, new1
            float tmp, value, old_value

        value = self.data[current0, current1]
        old_value = value - 1.0
        new0, new1 = current0, current1

        while value > old_value:
            old_value = value
            start0 = max(0, current0 - 1)
            stop0 = min(self.height, current0 + 2)
            start1 = max(0, current1 - 1)
            stop1 = min(self.width, current1 + 2)
            for i0 in range(start0, stop0):
                for i1 in range(start1, stop1):
                    tmp = self.data[i0, i1]
                    if tmp > value:
                        new0, new1 = i0, i1
                        value = tmp
            current0, current1 = new0, new1
        return self.width * current0 + current1
