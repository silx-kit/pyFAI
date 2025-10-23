# -*- coding: utf-8 -*-
#cython: embedsignature=True, language_level=3, binding=True
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False,
## This is for developing:
##cython: profile=True, warn.undeclared=True, warn.unused=True, warn.unused_result=False, warn.unused_arg=True
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2014-2022 European Synchrotron Radiation Facility, Grenoble, France
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
This module provides a couple of binary morphology operations on images.

They are also implemented in ``scipy.ndimage`` in the general case, but not as
fast.
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "27/12/2022"
__status__ = "stable"
__license__ = "MIT"

import cython
import numpy
from libc.stdint cimport int8_t, uint8_t, int16_t, uint16_t, \
                         int32_t, uint32_t, int64_t, uint64_t


def binary_dilation(int8_t[:, ::1] image,
                    float radius=1.0):
    """
    Return fast binary morphological dilation of an image.

    Morphological dilation sets a pixel at (i,j) to the maximum over all pixels in the neighborhood centered at (i,j).
    Dilation enlarges bright regions and shrinks dark regions.

    :param image : ndarray
    :param radius: float
    :return: ndiamge
    """
    cdef:
        Py_ssize_t x, y, i, j, size_x, size_y, px, py,
        Py_ssize_t r_int = int(radius), r2_int = int(radius * radius)
        int8_t val, curr
        int8_t[:, ::1] result
    size_y = image.shape[0]
    size_x = image.shape[1]
    result = numpy.empty(dtype=numpy.int8, shape=(size_y, size_x))

    for y in range(size_y):
        for x in range(size_x):
            val = image[y, x]
            for j in range(-r_int, r_int + 1):
                py = y + j
                if py < 0 or py >= size_y:
                    continue
                for i in range(-r_int, r_int + 1):
                    px = x + i
                    if (px < 0) or (px >= size_x):
                        continue
                    if i * i + j * j <= r2_int:
                        curr = image[py, px]
                        val = max(val, curr)
            result[y, x] = val
    return numpy.asarray(result)


def binary_erosion(int8_t[:, ::1] image,
                   float radius=1.0):
    """Return fast binary morphological erosion of an image.

    Morphological erosion sets a pixel at (i,j) to the minimum over all pixels
    in the neighborhood centered at (i,j).
    Erosion shrinks bright regions and enlarges dark regions.

    :param image : ndarray
    :param radius: float
    :return: ndiamge
    """
    cdef:
        Py_ssize_t x, y, i, j, size_x, size_y, px, py,
        Py_ssize_t r_int = int(radius), r2_int = int(radius * radius)
        int8_t val, curr
        int8_t[:, ::1] result
    size_y = image.shape[0]
    size_x = image.shape[1]
    result = numpy.empty(dtype=numpy.int8, shape=(size_y, size_x))

    for y in range(size_y):
        for x in range(size_x):
            val = image[y, x]
            for j in range(-r_int, r_int + 1):
                py = y + j
                if (py < 0) or (py >= size_y):
                    continue
                for i in range(-r_int, r_int + 1):
                    px = x + i
                    if (px < 0) or (px >= size_x):
                        continue
                    if i * i + j * j <= r2_int:
                        curr = image[py, px]
                        val = min(val, curr)
            result[y, x] = val
    return numpy.asarray(result)
