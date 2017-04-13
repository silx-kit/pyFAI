#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
Simple Cython module for doing CRC32 for checksums, possibly with SSE4 acceleration
"""
__author__ = "Jérôme Kieffer"
__date__ = "13/04/2017"
__contact__ = "Jerome.kieffer@esrf.fr"
__license__ = "MIT"

import cython
cimport numpy
import numpy
from libc import ceil, floor, pi

include "bilinear.pxi"


def largest_width(numpy.int8_t[:, :]image):
    """Calculate the width of the largest part in the binary image 
    Nota: this is along the horizontal direction.
    """   
    cdef: 
        int start, largest, row, col, current
        bint started
    start = 0
    largest = 0

    for row in range(image.shape[0]):
        started = False
        start = 0
        for col in range(image.shape[1]):
            if started:
                if not image[row, col]:
                    started = False
                    current = col - start
                    if current > largest:
                        largest = current
            elif image[row, col]:
                started = True
                start = col
    return largest


def inpaint(img, topaint, mask=None):
    """Relaplce the values flagged in topaint with possible values
    If mask is provided, those values are knows to be invalid and not re-calculated
    """
    pass


def polar_interpolate(data, 
                      numpy.int8_t[:, ::1] mask, 
                      cython.floating[:, ::1] radial, 
                      cython.floating[:, ::1] azimuthal,
                      float[:, ::1] polar, 
                      cython.floating[::1] rad_pos, 
                      cython.floating[::1] azim_pos 
                      ):
    """Perform the bilinear interpolation from polar data into the initial array
    data 
    
    :param data: image with holes, of a given shape
    :param mask: array with the holes marked
    :param radial: 2D array with the radial position 
    :param polar: 2D radial/azimuthal averaged image (continuous). shape is pshape
    :param radial_pos: position of the radial bins (evenly spaced, size = pshape[-1])
    :param azim_pos: position of the azimuthal bins (evenly spaced, size = pshape[0])
    :return: inpainted image 
    """
    cdef:
        float[:, ::1] cdata, cpolar
        int row, col, npt_radial, npt_azim, nb_col, nb_row 
        double azimuthal_min, radial_min, azimuthal_slope, radial_slope, r, a
        Bilinear bili 
        
    npt_radial = rad_pos.ize
    npt_azim = azim_pos.size
    assert polar.shape[0] == npt_azim, "polar.shape[0] == npt_azim"
    assert polar.shape[1] == npt_radial, "polar.shape[1] == npt_radial"

    nb_row = data.shape[0]
    nb_col = data.shape[1]
    
    assert mask.shape == data.shape, "mask.shape == data.shape"
    assert radial == data.shape, "radial == data.shape"
    assert azimuthal == data.shape, "azimuthal == data.shape"
    
    azimuthal_min = azim_pos[0] * pi / 180.
    radial_min = radial_pos[0]
    azimuthal_slope = pi * (azim_pos[npt_azim - 1] - azim_pos[0]) / (npt_azim - 1) / 180.
    radial_slope = (rad_pos[npt_radial - 1] - rad_pos[0]) / (npt_radial - 1)

    bili = Bilinear(polar)

    cdata = numpy.array(data, dtype=numpy.float32) # explicit copy
    cpolar = numpy.ascontiguousarray(polar, dtype=numpy.float32)
    for row in range(nb_row):
        for col in range(nb_col):
            if mask[row, col]:
                r = (radial[row, col] - radial_min) / radial_slope
                a = (chi_center[row, col] - azimuthal_min) / azimuthal_slope
                cdata[row, col] = bili._f_cy(a, r)
    return numpy.asarray(cdata)
