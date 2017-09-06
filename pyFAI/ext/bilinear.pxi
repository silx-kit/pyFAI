# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
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

from libc.math cimport floor, ceil
import logging
logger = logging.getLogger("bilinear")

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

    def f_cy(self, x):
        """
        Function -f((y,x)) where f is a continuous function
        (y,x) are pixel coordinates
        pixels outside the image are given an arbitrary high value to help the minimizer

        :param x: 2-tuple of float
        :return: Interpolated negative signal from the image
                 (negative for using minimizer to search for peaks)
        """
        cdef:
            float d0 = x[0]
            float d1 = x[1]
        if d0 < 0:
            res = self.mini + d0
        elif d1 < 0:
            res = self.mini + d1
        elif d0 > (self.height - 1):
            res = self.mini - d0 + self.height - 1
        elif d1 > self.width - 1:
            res = self.mini - d1 + self.width - 1
        else:
            res = self._f_cy(d0, d1)
        return -res

    def __call__(self, x):
        "Function f((y,x)) where f is a continuous function "
        cdef:
            float d0 = x[0]
            float d1 = x[1]
        return self._f_cy(d0, d1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef float _f_cy(self, cython.floating d0, cython.floating d1) nogil:
        """
        Function f((y,x)) where f is a continuous function (y,x) are pixel coordinates

        :param x: 2-tuple of float
        :return: Interpolated signal from the image
        """

        cdef:
            int i0, i1, j0, j1
            float x0, x1, y0, y1, res
        if d0 < 0:
            d0 = 0
        elif d1 < 0:
            d1 = 0
        elif d0 > (self.height - 1):
            d0 = self.height - 1
        elif d1 > self.width - 1:
            d1 = self.width - 1
        x0 = floor(d0)
        x1 = ceil(d0)
        y0 = floor(d1)
        y1 = ceil(d1)
        i0 = < int > x0
        i1 = < int > x1
        j0 = < int > y0
        j1 = < int > y1
        if (i0 == i1) and (j0 == j1):
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
        return res

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def local_maxi(self, x):
        """
        Return the local maximum with sub-pixel refinement.

        Sub-pixel refinement:
        Second order Taylor expansion of the function; first derivative is null

        .. math:: delta = x-i = -Inverse[Hessian].gradient

        If Hessian is singular or :math:`|delta|>1`: use a center of mass.

        :param x: 2-tuple of integers
        :param w: half with of the window: 1 or 2 are advised
        :return: 2-tuple of float with the nearest local maximum
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

        :param x: start index
        :param w: half with of the window: 1 or 2 are advised
        :return: local maximum index
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
