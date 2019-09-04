# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal Integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2014-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:   Aurore Deschildre <auroredeschildre@gmail.com>
#                        Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
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
Some Cythonized function for blob detection function.

It is used to find peaks in images by performing subsequent blurs.
"""

__authors__ = ["Aurore Deschildre", "Jerome Kieffer"]
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "17/05/2019"
__status__ = "stable"
__license__ = "MIT"
import cython
import numpy
cimport numpy
from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def local_max(float[:, :, ::1] dogs, mask=None, bint n_5=False):
    """Calculate if a point is a maximum in a 3D space: (scale, y, x)

    :param dogs: 3D array of difference of gaussian
    :param mask: mask with invalid pixels
    :param N_5: take a neighborhood of 5x5 pixel in plane
    :return: 3d_array with 1 where is_max
    """
    cdef bint do_mask = mask is not None
    cdef int ns, ny, nx, s, x, y
    cdef numpy.int8_t m
    cdef float c
    cdef numpy.int8_t[:, ::1] cmask
    cdef numpy.int8_t[:, :, ::1] is_max
    ns = dogs.shape[0]
    ny = dogs.shape[1]
    nx = dogs.shape[2]
    if do_mask:
        assert mask.shape[0] == ny, "mask shape 0/y"
        assert mask.shape[1] == nx, "mask shape 1/x"
        cmask = numpy.ascontiguousarray(mask, dtype=numpy.int8)

    is_max = numpy.zeros((ns, ny, nx), dtype=numpy.int8)
    if (ns < 3) or (ny < 3) or (nx < 3):
        return is_max
    for s in range(1, ns - 1):
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                c = dogs[s, y, x]
                if do_mask and cmask[y, x]:
                    m = 0
                else:
                    m = (c > dogs[s, y, x - 1]) and (c > dogs[s, y, x + 1]) and\
                        (c > dogs[s, y - 1, x]) and (c > dogs[s, y + 1, x]) and\
                        (c > dogs[s, y - 1, x - 1]) and (c > dogs[s, y - 1, x + 1]) and\
                        (c > dogs[s, y + 1, x - 1]) and (c > dogs[s, y + 1, x + 1]) and\
                        (c > dogs[s - 1, y, x]) and (c > dogs[s - 1, y, x]) and\
                        (c > dogs[s - 1, y, x - 1]) and (c > dogs[s - 1, y, x + 1]) and\
                        (c > dogs[s - 1, y - 1, x]) and (c > dogs[s - 1, y + 1, x]) and\
                        (c > dogs[s - 1, y - 1, x - 1]) and (c > dogs[s - 1, y - 1, x + 1]) and\
                        (c > dogs[s - 1, y + 1, x - 1]) and (c > dogs[s - 1, y + 1, x + 1]) and\
                        (c > dogs[s + 1, y, x - 1]) and (c > dogs[s + 1, y, x + 1]) and\
                        (c > dogs[s + 1, y - 1, x]) and (c > dogs[s + 1, y + 1, x]) and\
                        (c > dogs[s + 1, y - 1, x - 1]) and (c > dogs[s + 1, y - 1, x + 1]) and\
                        (c > dogs[s + 1, y + 1, x - 1]) and (c > dogs[s + 1, y + 1, x + 1])
                    if not m:
                        continue
                    if n_5:
                        if x > 1:
                            m = (m and (c > dogs[s, y, x - 2]) and (c > dogs[s, y - 1, x - 2]) and (c > dogs[s, y + 1, x - 2]) and
                                 (c > dogs[s - 1, y, x - 2]) and (c > dogs[s - 1, y - 1, x - 2]) and (c > dogs[s - 1, y + 1, x - 2]) and
                                 (c > dogs[s + 1, y, x - 2]) and (c > dogs[s + 1, y - 1, x - 2]) and (c > dogs[s + 1, y + 1, x - 2]))
                            if y > 1:
                                m = m and (c > dogs[s, y - 2, x - 2])and (c > dogs[s - 1, y - 2, x - 2]) and (c > dogs[s, y - 2, x - 2])
                            if y < ny - 2:
                                m = m and (c > dogs[s, y + 2, x - 2])and (c > dogs[s - 1, y + 2, x - 2]) and (c > dogs[s, y + 2, x - 2])
                        if x < (nx - 2):
                            m = (m and (c > dogs[s, y, x + 2]) and (c > dogs[s, y - 1, x + 2]) and (c > dogs[s, y + 1, x + 2]) and
                                 (c > dogs[s - 1, y, x + 2]) and (c > dogs[s - 1, y - 1, x + 2]) and (c > dogs[s - 1, y + 1, x + 2]) and
                                 (c > dogs[s + 1, y, x + 2]) and (c > dogs[s + 1, y - 1, x + 2]) and (c > dogs[s + 1, y + 1, x + 2]))
                            if y > 1:
                                m = m and (c > dogs[s, y - 2, x + 2])and (c > dogs[s - 1, y - 2, x + 2]) and (c > dogs[s, y - 2, x + 2])
                            if y < ny - 2:
                                m = m and (c > dogs[s, y + 2, x + 2])and (c > dogs[s - 1, y + 2, x + 2]) and (c > dogs[s, y + 2, x + 2])

                        if y > 1:
                            m = (m and (c > dogs[s, y - 2, x]) and (c > dogs[s, y - 2, x - 1]) and (c > dogs[s, y - 2, x + 1]) and
                                 (c > dogs[s - 1, y - 2, x]) and (c > dogs[s - 1, y - 2, x - 1]) and (c > dogs[s - 1, y - 2, x + 1]) and
                                 (c > dogs[s + 1, y - 2, x]) and (c > dogs[s + 1, y - 2, x - 1]) and (c > dogs[s + 1, y + 2, x + 1]))

                        if y < (ny - 2):
                            m = (m and (c > dogs[s, y + 2, x]) and (c > dogs[s, y + 2, x - 1]) and (c > dogs[s, y + 2, x + 1]) and
                                 (c > dogs[s - 1, y + 2, x]) and (c > dogs[s - 1, y + 2, x - 1]) and (c > dogs[s - 1, y + 2, x + 1]) and
                                 (c > dogs[s + 1, y + 2, x]) and (c > dogs[s + 1, y + 2, x - 1]) and (c > dogs[s + 1, y + 2, x + 1]))

                is_max[s, y, x] = m
    return numpy.asarray(is_max)
