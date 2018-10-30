#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#
#    Copyright (C) 2017-2018 European Synchrotron Radiation Facility, Grenoble, France
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
__date__ = "26/10/2018"
__contact__ = "Jerome.kieffer@esrf.fr"
__license__ = "MIT"

from cython cimport floating 
cimport numpy
import numpy
from libc.math cimport floor, ceil

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


def polar_interpolate(data,
                      numpy.int8_t[:, ::1] mask,
                      floating[:, ::1] radial,
                      floating[:, ::1] azimuthal,
                      polar,
                      floating[::1] rad_pos,
                      floating[::1] azim_pos
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
        float[:, ::1] cdata
        int row, col, npt_radial, npt_azim, nb_col, nb_row
        double azimuthal_min, radial_min, azimuthal_slope, radial_slope, r, a
        Bilinear bili

    npt_radial = rad_pos.size
    npt_azim = azim_pos.size
    assert polar.shape[0] == npt_azim, "polar.shape[0] == npt_azim"
    assert polar.shape[1] == npt_radial, "polar.shape[1] == npt_radial"

    nb_row = data.shape[0]
    nb_col = data.shape[1]

    assert mask.shape[0] == nb_row, "mask.shape == data.shape"
    assert mask.shape[1] == nb_col, "mask.shape == data.shape"
    assert radial.shape[0] == nb_row, "radial == data.shape"
    assert radial.shape[1] == nb_col, "radial == data.shape"
    assert azimuthal.shape[0] == nb_row, "azimuthal == data.shape"
    assert azimuthal.shape[1] == nb_col, "azimuthal == data.shape"

    azimuthal_min = azim_pos[0]
    radial_min = rad_pos[0]
    azimuthal_slope = (azim_pos[npt_azim - 1] - azim_pos[0]) / (npt_azim - 1)
    radial_slope = (rad_pos[npt_radial - 1] - rad_pos[0]) / (npt_radial - 1)

    bili = Bilinear(polar)

    cdata = numpy.array(data, dtype=numpy.float32)  # explicit copy

    for row in range(nb_row):
        for col in range(nb_col):
            if mask[row, col]:
                r = (radial[row, col] - radial_min) / radial_slope
                a = (azimuthal[row, col] - azimuthal_min) / azimuthal_slope
                cdata[row, col] = bili._f_cy(a, r)
    return numpy.asarray(cdata)


def polar_inpaint(floating[:, :] img not None,
                  numpy.int8_t[:, :] topaint not None,
                  numpy.int8_t[:, :] mask=None,
                  empty=None):
    """Relaplce the values flagged in topaint with possible values
    If mask is provided, those values are knows to be invalid and not re-calculated

    :param img: image in polar coordinates
    :param topaint: pixel which deserves inpatining
    :param mask: pixels which are masked and do not need to be inpainted
    :param dummy: value for masked
    :return: image with missing values interpolated from neighbors.
    """
    cdef:
        int row, col, npt_radial, npt_azim, idx_col, idx_row, tar_row, radius
        int start_col, end_col, start_row, end_row
        float[:, ::1] res
        bint do_dummy = empty is not None
        float value, dummy, dist
        double sum, cnt, weight
        list values

    npt_azim = img.shape[0]
    npt_radial = img.shape[1]

    if do_dummy:
        dummy = empty

    assert topaint.shape[0] == npt_azim, "topaint.shape == img.shape"
    assert topaint.shape[1] == npt_radial, "topaint.shape == img.shape"
    assert img.ndim == 2, "img.ndim == 2"

    if mask is None:
        mask = numpy.zeros((npt_azim, npt_radial), numpy.int8)
    else:
        assert mask.shape[0] == npt_azim, "mask.shape == img.shape"
        assert mask.shape[1] == npt_radial, "mask.shape == img.shape"

    assert topaint.ndim == 2, "topaint.ndim == 2"
    assert mask.ndim == 2, "mask.ndim == 2"

    res = numpy.zeros((npt_azim, npt_radial), numpy.float32)

    for row in range(npt_azim):
        for col in range(npt_radial):
            if topaint[row, col]:  # this pixel deserves inpaining
                values = []
                for idx_row in range(row - 1, -1, -1):
                    if (topaint[idx_row, col] == 0) and (mask[idx_row, col] == 0):
                        dist = row - idx_row
                        values.append((img[idx_row, col], dist * dist))
                        if len(values) > 1:
                            break
                if values:
                    tar_row = min(npt_azim, 2 * row - idx_row + 1)
                else:
                    tar_row = npt_azim
                for idx_row in range(row + 1, tar_row, 1):
                    if topaint[idx_row, col] == 0 and mask[idx_row, col] == 0:
                        dist = idx_row - row
                        values.append((img[idx_row, col], dist * dist))
                        if len(values) > 3:
                            break
                if not values:
                    # radial search:
                    radius = 0
                    while not values:
                        radius += 1
                        if radius > max(npt_azim, npt_radial):
                            # Avoid infinit loop
                            break
                        idx_col = max(0, col - radius)
                        for idx_row in range(max(0, row - radius), min(npt_azim, row + radius + 1)):
                            if topaint[idx_row, idx_col] == 0 and mask[idx_row, idx_col] == 0:
                                values.append((img[idx_row, idx_col],
                                               (row - idx_row) ** 2 + (col - idx_col) ** 2))

                        idx_col = min(npt_radial - 1, col + radius)
                        for idx_row in range(max(0, row - radius), min(npt_azim, row + radius + 1)):
                            if topaint[idx_row, idx_col] == 0 and mask[idx_row, idx_col] == 0:
                                values.append((img[idx_row, idx_col],
                                               (row - idx_row) ** 2 + (col - idx_col) ** 2))

                        idx_row = max(0, row - radius)
                        for idx_col in range(max(0, col - radius), min(npt_radial, col + radius + 1)):
                            if topaint[idx_row, idx_col] == 0 and mask[idx_row, idx_col] == 0:
                                values.append((img[idx_row, idx_col],
                                               (row - idx_row) ** 2 + (col - idx_col) ** 2))

                        idx_row = min(npt_azim - 1, row + radius)
                        for idx_col in range(max(0, col - radius), min(npt_radial, col + radius + 1)):
                            if topaint[idx_row, idx_col] == 0 and mask[idx_row, idx_col] == 0:
                                values.append((img[idx_row, idx_col],
                                               (row - idx_row) ** 2 + (col - idx_col) ** 2))
                    if len(values) == 0:
                        zero = numpy.int8(0)
                        if not numpy.logical_and(mask == zero, topaint == zero).any():
                            # There is no available valid pixel on all the image
                            raise RuntimeError("No valid pixel available according to topaint and mask masks. Fix your masks.")
                        raise RuntimeError("No value found for pixel %s,%s (row,col)" % (row, col))
                cnt = 0.0
                sum = 0.0
                for vd in values:
                    weight = 1.0 / vd[1]
                    sum += vd[0] * weight
                    cnt += weight
                value = sum / cnt
            elif do_dummy and mask[row, col]:
                value = dummy
            else:
                value = img[row, col]
            res[row, col] = value

    return numpy.asarray(res)
