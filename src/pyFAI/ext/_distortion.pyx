# -*- coding: utf-8 -*-
#cython: embedsignature=True, language_level=3, binding=True
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False,
## This is for developping:
##cython: profile=True, warn.undeclared=True, warn.unused=True, warn.unused_result=False, warn.unused_arg=True
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2013-2025 European Synchrotron Radiation Facility, Grenoble, France
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
Distortion correction are correction are applied by look-up table (or CSR)
"""

__author__ = "Jerome Kieffer"
__license__ = "MIT"
__date__ = "06/10/2025"
__copyright__ = "2011-2021, ESRF"
__contact__ = "jerome.kieffer@esrf.fr"

include "regrid_common.pxi"
import cython
import numpy
from cython cimport view
from cython.parallel import prange
from cpython.ref cimport PyObject, Py_XDECREF
from libc.string cimport memset, memcpy
from libc.math cimport floor, ceil, fabs, copysign, sqrt
import logging
import threading
import types
import os
import sys
import time
logger = logging.getLogger(__name__)
from ..detectors import detector_factory
from ..utils.mathutil import expand2d
import fabio

from .sparse_builder cimport SparseBuilder


cdef inline float _floor_min4(float a, float b, float c, float d) noexcept nogil:
    "return floor(min(a,b,c,d))"
    cdef float res
    if (b < a):
        res = b
    else:
        res = a
    if (c < res):
        res = c
    if (d < res):
        res = d
    return floor(res)


cdef inline float _ceil_max4(float a, float b, float c, float d) noexcept nogil:
    "return  ceil(max(a,b,c,d))"
    cdef float res
    if (b > a):
        res = b
    else:
        res = a
    if (c > res):
        res = c
    if (d > res):
        res = d
    return ceil(res)



################################################################################
# Functions used in python classes from PyFAI.distortion
################################################################################

def calc_pos(floating[:, :, :, ::1] pixel_corners not None,
             float pixel1, float pixel2, shape_out=None):
    """Calculate the pixel boundary position on the regular grid

    :param pixel_corners: pixel corner coordinate as detector.get_pixel_corner()
    :param shape: requested output shape. If None, it is calculated
    :param pixel1, pixel2: pixel size along row and column coordinates
    :return: pos, delta1, delta2, shape_out, offset
    """
    cdef:
        float32_t[:, :, :, ::1] pos
        int i, j, k, dim0, dim1, nb_corners
        bint do_shape = (shape_out is None)
        float BIG = numpy.finfo(numpy.float32).max
        float min0, min1, max0, max1, delta0, delta1
        float all_min0, all_max0, all_max1, all_min1
        float p0, p1

    if (pixel1 == 0.0) or (pixel2 == 0.0):
        raise RuntimeError("Pixel size cannot be null -> Zero division error")

    dim0 = pixel_corners.shape[0]
    dim1 = pixel_corners.shape[1]
    nb_corners = pixel_corners.shape[2]
    pos = numpy.zeros((dim0, dim1, 4, 2), dtype=numpy.float32)
    with nogil:
        delta0 = -BIG
        delta1 = -BIG
        all_min0 = BIG
        all_min1 = BIG
        all_max0 = -BIG
        all_max1 = -BIG
        for i in range(dim0):
            for j in range(dim1):
                min0 = BIG
                min1 = BIG
                max0 = -BIG
                max1 = -BIG
                for k in range(nb_corners):
                    p0 = pixel_corners[i, j, k, 1] / pixel1
                    p1 = pixel_corners[i, j, k, 2] / pixel2
                    pos[i, j, k, 0] = p0
                    pos[i, j, k, 1] = p1
                    min0 = p0 if p0 < min0 else min0
                    min1 = p1 if p1 < min1 else min1
                    max0 = p0 if p0 > max0 else max0
                    max1 = p1 if p1 > max1 else max1
                delta0 = max(delta0, ceil(max0) - floor(min0))
                delta1 = max(delta1, ceil(max1) - floor(min1))
                if do_shape:
                    all_min0 = min0 if min0 < all_min0 else all_min0
                    all_min1 = min1 if min1 < all_min1 else all_min1
                    all_max0 = max0 if max0 > all_max0 else all_max0
                    all_max1 = max1 if max1 > all_max1 else all_max1

    res = numpy.asarray(pos), int(delta0), int(delta1), \
        (int(ceil(all_max0 - all_min0)), int(ceil(all_max1 - all_min1))) if do_shape else shape_out, \
        (float(all_min0), float(all_min1)) if do_shape else (0.0, 0.0)
    return res


def calc_size(floating[:, :, :, ::1] pos not None,
              shape,
              int8_t[:, ::1] mask=None,
              offset=None):
    """Calculate the number of items per output pixel

    :param pos: 4D array with position in space
    :param shape: shape of the output array
    :param mask: input data mask
    :param offset: 2-tuple of float with the minimal index of
    :return: number of input element per output elements
    """
    cdef:
        int i, j, k, l, shape_out0, shape_out1, shape_in0, shape_in1, min0, min1, max0, max1
        int32_t[:, ::1] lut_size = numpy.zeros(shape, dtype=numpy.int32)
        float A0, A1, B0, B1, C0, C1, D0, D1, offset0=0.0, offset1=0.0
        bint do_mask = mask is not None
        int8_t[:, ::1] cmask
    shape_in0, shape_in1 = pos.shape[0], pos.shape[1]
    shape_out0, shape_out1 = shape

    if do_mask:

        if ((mask.shape[0] != shape_in0) or (mask.shape[1] != shape_in1)):
            err = 'Mismatch between shape of detector (%s, %s) and shape of mask (%s, %s)' % (shape_in0, shape_in1, mask.shape[0], mask.shape[1])
            logger.error(err)
            raise RuntimeError(err)
        else:
            cmask = numpy.ascontiguousarray(mask, dtype=numpy.int8)

    if offset is not None:
        offset0, offset1 = offset

    with nogil:
        for i in range(shape_in0):
            for j in range(shape_in1):
                if do_mask and cmask[i, j]:
                    continue
                A0 = pos[i, j, 0, 0] - offset0
                A1 = pos[i, j, 0, 1] - offset1
                B0 = pos[i, j, 1, 0] - offset0
                B1 = pos[i, j, 1, 1] - offset1
                C0 = pos[i, j, 2, 0] - offset0
                C1 = pos[i, j, 2, 1] - offset1
                D0 = pos[i, j, 3, 0] - offset0
                D1 = pos[i, j, 3, 1] - offset1
                min0 = _clip(<int> _floor_min4(A0, B0, C0, D0), 0, shape_out0)
                min1 = _clip(<int> _floor_min4(A1, B1, C1, D1), 0, shape_out1)
                max0 = _clip(<int> _ceil_max4(A0, B0, C0, D0) + 1, 0, shape_out0)
                max1 = _clip(<int> _ceil_max4(A1, B1, C1, D1) + 1, 0, shape_out1)
                for k in range(min0, max0):
                    for l in range(min1, max1):
                        lut_size[k, l] += 1
    return numpy.asarray(lut_size)


def calc_LUT(float32_t[:, :, :, ::1] pos not None, shape, bin_size, max_pixel_size,
             int8_t[:, :] mask=None, offset=(0,0)):
    """
    :param pos: 4D position array
    :param shape: output shape
    :param bin_size: number of input element per output element (numpy array)
    :param max_pixel_size: (2-tuple of int) size of a buffer covering the largest pixel
    :param mask: arry with bad pixels marked as True
    :param offset: global offset for pixel position
    :return: look-up table
    """
    cdef:
        int i, j, ms, ml, ns, nl, shape0, shape1, delta0, delta1
        int offset0, offset1, box_size0, box_size1, size, k
        int32_t idx = 0
        int err_cnt = 0
        float A0, A1, B0, B1, C0, C1, D0, D1
        float area, inv_area, value, foffset0, foffset1, goffset0, goffset1
        lut_t[:, :, :] lut
        bint do_mask = mask is not None
        buffer_t[:, ::1] buffer
    size = bin_size.max()
    shape0, shape1 = shape
    if do_mask:
        assert shape0 == mask.shape[0], "mask shape dim0"
        assert shape1 == mask.shape[1], "mask shape dim1"
    delta0, delta1 = max_pixel_size
    cdef int32_t[:, :] outMax = numpy.zeros((shape0, shape1), dtype=numpy.int32)
    buffer = numpy.empty((delta0, delta1), dtype=buffer_d)

    if (size == 0):  # fix 271
        raise RuntimeError("The look-up table has dimension 0 which is a non-sense." +
                           " Did you mask out all pixel or is your image out of the geometry range?")
    lut = view.array(shape=(shape0, shape1, size), itemsize=sizeof(lut_t), format="if")
    lut_total_size = shape0 * shape1 * size * sizeof(lut_t)
    memset(&lut[0, 0, 0], 0, lut_total_size)
    logger.info("LUT shape: (%i,%i,%i) %.3f MByte" % (lut.shape[0], lut.shape[1], lut.shape[2], lut_total_size / 1.0e6))
    logger.info("Max pixel size: %ix%i; Max source pixel in target: %i" % (delta1, delta0, size))
    #Manage global pixel offset:
    goffset0 = float(offset[0])
    goffset1 = float(offset[1])
    with nogil:
        # i,j, idx are indexes of the raw image uncorrected
        for i in range(shape0):
            for j in range(shape1):
                if do_mask and mask[i, j]:
                    continue
                # reset buffer
                buffer[:, :] = 0.0

                A0 = pos[i, j, 0, 0] - goffset0
                A1 = pos[i, j, 0, 1] - goffset1
                B0 = pos[i, j, 1, 0] - goffset0
                B1 = pos[i, j, 1, 1] - goffset1
                C0 = pos[i, j, 2, 0] - goffset0
                C1 = pos[i, j, 2, 1] - goffset1
                D0 = pos[i, j, 3, 0] - goffset0
                D1 = pos[i, j, 3, 1] - goffset1
                foffset0 = _floor_min4(A0, B0, C0, D0)
                foffset1 = _floor_min4(A1, B1, C1, D1)
                offset0 = (<int> foffset0)
                offset1 = (<int> foffset1)
                box_size0 = (<int> _ceil_max4(A0, B0, C0, D0)) - offset0
                box_size1 = (<int> _ceil_max4(A1, B1, C1, D1)) - offset1
                if (box_size0 > delta0) or (box_size1 > delta1):
                    # Increase size of the buffer
                    delta0 = max(offset0, delta0)
                    delta1 = max(offset1, delta1)
                    with gil:
                        buffer = numpy.zeros((delta0, delta1), dtype=buffer_d)

                A0 -= foffset0
                A1 -= foffset1
                B0 -= foffset0
                B1 -= foffset1
                C0 -= foffset0
                C1 -= foffset1
                D0 -= foffset0
                D1 -= foffset1

                # ABCD is trigonometric order: order input position accordingly
                _integrate2d(buffer, B0, B1, A0, A1)
                _integrate2d(buffer, C0, C1, B0, B1)
                _integrate2d(buffer, D0, D1, C0, C1)
                _integrate2d(buffer, A0, A1, D0, D1)

                area = 0.5 * ((C0 - A0) * (D1 - B1) - (C1 - A1) * (D0 - B0))
                inv_area = 1.0 / area
                for ms in range(box_size0):
                    ml = ms + offset0
                    if ml < 0 or ml >= shape0:
                        continue
                    for ns in range(box_size1):
                        # ms,ns are indexes of the corrected image in short form, ml & nl are the same
                        nl = ns + offset1
                        if nl < 0 or nl >= shape1:
                            continue
                        value = buffer[ms, ns] * inv_area
                        if value == 0:
                            continue
                        if value < 0 or value > 1.0001:
                            # here we print pathological cases for debugging
                            if err_cnt < 1000:
                                with gil:
                                    print(i, j, ms, box_size0, ns, box_size1, buffer[ms, ns], area, value, buffer[0, 0], buffer[0, 1], buffer[1, 0], buffer[1, 1])
                                    print(" A0=%s; A1=%s; B0=%s; B1=%s; C0=%s; C1=%s; D0=%s; D1=%s" % (A0, A1, B0, B1, C0, C1, D0, D1))
                                err_cnt += 1
                            continue
                        k = outMax[ml, nl]
                        lut[ml, nl, k].idx = idx
                        lut[ml, nl, k].coef = value
                        outMax[ml, nl] = k + 1
                idx += 1

    # Hack to prevent memory leak !!!
    cdef float64_t[:, ::1] tmp_ary = numpy.empty(shape=(shape0 * shape1, size), dtype=numpy.float64)
    memcpy(&tmp_ary[0, 0], &lut[0, 0, 0], tmp_ary.nbytes)
    return numpy.core.records.array(numpy.asarray(tmp_ary).view(dtype=lut_d),
                                    shape=(shape0 * shape1, size), dtype=lut_d,
                                    copy=True)


def calc_CSR(float32_t[:, :, :, :] pos not None, shape, bin_size, max_pixel_size,
             int8_t[:, ::1] mask=None, offset=(0,0)):
    """Calculate the Look-up table as CSR format

    :param pos: 4D position array
    :param shape: output shape
    :param bin_size: number of input element per output element (as numpy array)
    :param max_pixel_size: (2-tuple of int) size of a buffer covering the largest pixel
    :param mask: array with invalid pixels marked
    :param offset: global offset for pixel coordinates
    :return: look-up table in CSR format: 3-tuple of array"""
    cdef:
        int shape0, shape1, delta0, delta1, bins
    shape0, shape1 = shape
    delta0, delta1 = max_pixel_size
    bins = shape0 * shape1
    cdef:
        int i, j, k, ms, ml, ns, nl, idx = 0, tmp_index, err_cnt = 0
        int lut_size, offset0, offset1, box_size0, box_size1
        float A0, A1, B0, B1, C0, C1, D0, D1
        float area, inv_area, value, foffset0, foffset1, goffset0, goffset1
        int32_t[::1] indptr, indices
        float32_t[::1] data
        int32_t[:, ::1] outMax = numpy.zeros((shape0, shape1), dtype=numpy.int32)
        buffer_t[:, ::1] buffer
        bint do_mask = mask is not None
    if do_mask:
        assert shape0 == mask.shape[0], "mask shape dim0"
        assert shape1 == mask.shape[1], "mask shape dim1"

    indptr = numpy.concatenate(([numpy.int32(0)], bin_size.cumsum(dtype=numpy.int32)))
    lut_size = indptr[bins]

    indices = numpy.zeros(shape=lut_size, dtype=numpy.int32)
    data = numpy.zeros(shape=lut_size, dtype=numpy.float32)

    logger.info("CSR matrix: %.3f MByte" % ((indices.nbytes + data.nbytes + indptr.nbytes) / 1.0e6))
    buffer = numpy.empty((delta0, delta1), dtype=buffer_d)
    logger.info("Max pixel size: %ix%i; Max source pixel in target: %i" % (buffer.shape[1], buffer.shape[0], lut_size))
    #global offset (in case the detector is centerred around the origin)
    goffset0 = float(offset[0])
    goffset1 = float(offset[1])

    with nogil:
        # i,j, idx are indices of the raw image uncorrected
        for i in range(shape0):
            for j in range(shape1):
                if do_mask and mask[i, j]:
                    continue
                # reinit of buffer
                buffer[:, :] = 0.0
                A0 = pos[i, j, 0, 0] - goffset0
                A1 = pos[i, j, 0, 1] - goffset1
                B0 = pos[i, j, 1, 0] - goffset0
                B1 = pos[i, j, 1, 1] - goffset1
                C0 = pos[i, j, 2, 0] - goffset0
                C1 = pos[i, j, 2, 1] - goffset1
                D0 = pos[i, j, 3, 0] - goffset0
                D1 = pos[i, j, 3, 1] - goffset1
                foffset0 = _floor_min4(A0, B0, C0, D0)
                foffset1 = _floor_min4(A1, B1, C1, D1)
                offset0 = (<int> foffset0)
                offset1 = (<int> foffset1)
                box_size0 = (<int> _ceil_max4(A0, B0, C0, D0)) - offset0
                box_size1 = (<int> _ceil_max4(A1, B1, C1, D1)) - offset1

                if (box_size0 > delta0) or (box_size1 > delta1):
                    # Increase size of the buffer
                    delta0 = max(offset0, delta0)
                    delta1 = max(offset1, delta1)
                    with gil:
                        buffer = numpy.zeros((delta0, delta1), dtype=buffer_d)

                A0 -= foffset0
                A1 -= foffset1
                B0 -= foffset0
                B1 -= foffset1
                C0 -= foffset0
                C1 -= foffset1
                D0 -= foffset0
                D1 -= foffset1

                _integrate2d(buffer, B0, B1, A0, A1)
                _integrate2d(buffer, C0, C1, B0, B1)
                _integrate2d(buffer, D0, D1, C0, C1)
                _integrate2d(buffer, A0, A1, D0, D1)

                area = 0.5 * ((C0 - A0) * (D1 - B1) - (C1 - A1) * (D0 - B0))
                inv_area = 1.0 / area
                for ms in range(box_size0):
                    ml = ms + offset0
                    if ml < 0 or ml >= shape0:
                        continue
                    for ns in range(box_size1):
                        # ms,ns are indexes of the corrected image in short form, ml & nl are the same
                        nl = ns + offset1
                        if nl < 0 or nl >= shape1:
                            continue
                        value = buffer[ms, ns] * inv_area
                        if value == 0.0:
                            continue
                        if value < 0.0 or value > 1.0001:
                            # here we print pathological cases for debugging
                            if err_cnt < 1000:
                                with gil:
                                    print(i, j, ms, box_size0, ns, box_size1, buffer[ms, ns], area, value, buffer[0, 0], buffer[0, 1], buffer[1, 0], buffer[1, 1])
                                    print(" A0=%s; A1=%s; B0=%s; B1=%s; C0=%s; C1=%s; D0=%s; D1=%s" % (A0, A1, B0, B1, C0, C1, D0, D1))
                                err_cnt += 1
                            continue

                        k = outMax[ml, nl]
                        tmp_index = indptr[ml * shape1 + nl]
                        indices[tmp_index + k] = idx
                        data[tmp_index + k] = value
                        outMax[ml, nl] = k + 1
                idx += 1
    return (numpy.asarray(data), numpy.asarray(indices), numpy.asarray(indptr))


def calc_sparse(float32_t[:, :, :, ::1] pos not None,
                shape,
                max_pixel_size=(8, 8),
                int8_t[:, ::1] mask=None,
                format="csr",
                int bins_per_pixel=8,
                offset=(0,0)):
    """Calculate the look-up table (or CSR) using OpenMP

    :param pos: 4D position array
    :param shape: output shape
    :param max_pixel_size: (2-tuple of int) size of a buffer covering the largest pixel
    :param mask: array with invalid pixels marked (True)
    :param format: can be "CSR" or "LUT"
    :param bins_per_pixel: average splitting factor (number of pixels per bin)
    :param offset: global pixel offset
    :return: look-up table in CSR/LUT format
    """
    cdef:
        int shape_in0, shape_in1, shape_out0, shape_out1, size_in, delta0, delta1, bins, large_size
    format = format.lower()
    shape_out0, shape_out1 = shape
    delta0, delta1 = max_pixel_size
    bins = shape_out0 * shape_out1
    large_size = bins * bins_per_pixel
    shape_in0 = pos.shape[0]
    shape_in1 = pos.shape[1]
    size_in = shape_in0 * shape_in1
    cdef:
        int i, j, ms, ml, ns, nl
        int lut_size, offset0, offset1, box_size0, box_size1
        int counter, bin_number
        int idx, err_cnt = 0
        float A0, A1, B0, B1, C0, C1, D0, D1
        float area, inv_area, value, foffset0, foffset1, goffset0, goffset1
        int32_t[::1] indptr, indices, idx_bin, idx_pixel, pixel_count
        float32_t[::1] data, large_data
        buffer_t[:, ::1] buffer
        bint do_mask = mask is not None
        lut_t[:, :] lut
    if do_mask:
        assert shape_in0 == mask.shape[0], "shape_in0 == mask.shape[0]"
        assert shape_in1 == mask.shape[1], "shape_in1 == mask.shape[1]"

    #  count the number of pixel falling into every single bin
    pixel_count = numpy.zeros(bins, dtype=numpy.int32)
    idx_pixel = numpy.zeros(large_size, dtype=numpy.int32)
    idx_bin = numpy.zeros(large_size, dtype=numpy.int32)
    large_data = numpy.zeros(large_size, dtype=numpy.float32)
    logger.info("Temporary storage: %.3fMB",
                (large_data.nbytes + pixel_count.nbytes + idx_pixel.nbytes + idx_bin.nbytes) / 1e6)

    buffer = numpy.empty((delta0, delta1), dtype=buffer_d)
    counter = -1  # bin index
    #global offset (in case the detector is centerred around the origin)
    goffset0 = float(offset[0])
    goffset1 = float(offset[1])
    with nogil:
        # i, j, idx are indices of the raw image uncorrected
        for idx in range(size_in):
            i = idx // shape_in1
            j = idx % shape_in1
            if do_mask and mask[i, j]:
                continue
            idx = i * shape_in1 + j  # pixel index
            buffer[:, :] = 0.0
            A0 = pos[i, j, 0, 0] - goffset0
            A1 = pos[i, j, 0, 1] - goffset1
            B0 = pos[i, j, 1, 0] - goffset0
            B1 = pos[i, j, 1, 1] - goffset1
            C0 = pos[i, j, 2, 0] - goffset0
            C1 = pos[i, j, 2, 1] - goffset1
            D0 = pos[i, j, 3, 0] - goffset0
            D1 = pos[i, j, 3, 1] - goffset1
            foffset0 = _floor_min4(A0, B0, C0, D0)
            foffset1 = _floor_min4(A1, B1, C1, D1)
            offset0 = <int> foffset0
            offset1 = <int> foffset1
            box_size0 = (<int> _ceil_max4(A0, B0, C0, D0)) - offset0
            box_size1 = (<int> _ceil_max4(A1, B1, C1, D1)) - offset1
            if (box_size0 > delta0) or (box_size1 > delta1):
                # Increase size of the buffer
                delta0 = offset0 if offset0 > delta0 else delta0
                delta1 = offset1 if offset1 > delta1 else delta1
                with gil:
                    buffer = numpy.zeros((delta0, delta1), dtype=buffer_d)

            A0 = A0 - foffset0
            A1 = A1 - foffset1
            B0 = B0 - foffset0
            B1 = B1 - foffset1
            C0 = C0 - foffset0
            C1 = C1 - foffset1
            D0 = D0 - foffset0
            D1 = D1 - foffset1
            _integrate2d(buffer, B0, B1, A0, A1)
            _integrate2d(buffer, C0, C1, B0, B1)
            _integrate2d(buffer, D0, D1, C0, C1)
            _integrate2d(buffer, A0, A1, D0, D1)

            area = 0.5 * ((C0 - A0) * (D1 - B1) - (C1 - A1) * (D0 - B0))
            inv_area = 1.0 / area
            for ms in range(box_size0):
                ml = ms + offset0
                if ml < 0 or ml >= shape_out0:
                    continue
                for ns in range(box_size1):
                    # ms,ns are indexes of the corrected image in short form, ml & nl are the same
                    nl = ns + offset1
                    if nl < 0 or nl >= shape_out1:
                        continue
                    value = buffer[ms, ns] * inv_area
                    if value == 0.0:
                        continue
                    if value < 0.0 or value > 1.0001:
                        # here we print pathological cases for debugging
                        if err_cnt < 1000:
                            with gil:
                                print(i, j, ms, box_size0, ns, box_size1, buffer[ms, ns], area, value, buffer[0, 0], buffer[0, 1], buffer[1, 0], buffer[1, 1])
                                print(" A0=%s; A1=%s; B0=%s; B1=%s; C0=%s; C1=%s; D0=%s; D1=%s" % (A0, A1, B0, B1, C0, C1, D0, D1))
                            err_cnt += 1
                        continue

                    bin_number = ml * shape_out1 + nl
                    # with gil: #Use the gil to perform an atomic operation
                    counter += 1
                    pixel_count[bin_number] += 1
                    if counter >= large_size:
                        with gil:
                            raise RuntimeError("Provided temporary space for storage is not enough. " +
                                               "Please increase bins_per_pixel=%s. " % bins_per_pixel +
                                               "The suggested value is %i or greater." % ceil(1.1 * bins_per_pixel * size_in / idx))
                    idx_pixel[counter] += idx
                    idx_bin[counter] += bin_number
                    large_data[counter] += value
    logger.info("number of elements: %s, average per bin %.3f allocated max: %s",
                counter, counter / size_in, bins_per_pixel)

    if format == "csr":
        indptr = numpy.zeros(bins + 1, dtype=numpy.int32)
        # cumsum
        j = 0
        for i in range(bins):
            indptr[i] = j
            j += pixel_count[i]
        indptr[bins] = j
        # indptr[1:] = numpy.asarray(pixel_count).cumsum(dtype=numpy.int32)
        pixel_count[:] = 0
        lut_size = indptr[bins]
        indices = numpy.zeros(shape=lut_size, dtype=numpy.int32)
        data = numpy.zeros(shape=lut_size, dtype=numpy.float32)

        logger.info("CSR matrix: %.3f MByte; Max source pixel in target: %i, average splitting: %.2f",
                    (indices.nbytes + data.nbytes + indptr.nbytes) / 1.0e6, lut_size, (1.0 * counter / bins))

        for idx in range(counter + 1):
            bin_number = idx_bin[idx]
            i = indptr[bin_number] + pixel_count[bin_number]
            pixel_count[bin_number] += 1
            indices[i] = idx_pixel[idx]
            data[i] = large_data[idx]
        res = (numpy.asarray(data), numpy.asarray(indices), numpy.asarray(indptr))
    elif format == "lut":
        lut_size = numpy.asarray(pixel_count).max()
        lut = numpy.zeros(shape=(bins, lut_size), dtype=lut_d)
        pixel_count[:] = 0
        logger.info("LUT matrix: %.3f MByte; Max source pixel in target: %i, average splitting: %.2f",
                    (lut.nbytes) / 1.0e6, lut_size, (1.0 * counter / bins))
        for idx in range(counter + 1):
            bin_number = idx_bin[idx]
            i = pixel_count[bin_number]
            lut[bin_number, i].idx = idx_pixel[idx]
            lut[bin_number, i].coef = large_data[idx]
            pixel_count[bin_number] += 1
        res = numpy.asarray(lut)
    else:
        raise RuntimeError("Unimplemented sparse matrix format: %s", format)
    return res


def calc_sparse_v2(float32_t[:, :, :, ::1] pos not None,
                   shape,
                   max_pixel_size=(8, 8),
                   int8_t[:, ::1] mask=None,
                   format="csr",
                   int bins_per_pixel=8,
                   builder_config=None):
    """Calculate the look-up table (or CSR) using OpenMP
    :param pos: 4D position array
    :param shape: output shape
    :param max_pixel_size: (2-tuple of int) size of a buffer covering the largest pixel
    :param format: can be "CSR" or "LUT"
    :param bins_per_pixel: average splitting factor (number of pixels per bin) #deprecated
    :return: look-up table in CSR/LUT format
    """
    cdef:
        int shape_in0, shape_in1, shape_out0, shape_out1, size_in, delta0, delta1, bins, large_size
    format = format.lower()
    shape_out0, shape_out1 = shape
    delta0, delta1 = max_pixel_size
    bins = shape_out0 * shape_out1
    large_size = bins * bins_per_pixel
    shape_in0 = pos.shape[0]
    shape_in1 = pos.shape[1]
    size_in = shape_in0 * shape_in1
    cdef:
        int i, j, ms, ml, ns, nl
        int offset0, offset1, box_size0, box_size1
        int counter, bin_number
        int idx, err_cnt = 0
        float A0, A1, B0, B1, C0, C1, D0, D1
        float area, inv_area, value, foffset0, foffset1
        buffer_t[:, ::1] buffer
        bint do_mask = mask is not None

    if do_mask:
        assert shape_in0 == mask.shape[0], "shape_in0 == mask.shape[0]"
        assert shape_in1 == mask.shape[1], "shape_in1 == mask.shape[1]"

    # Here we create a builder:
    if builder_config is None:
        builder = SparseBuilder(bins, block_size=6, heap_size=bins)
    else:
        builder = SparseBuilder(bins, **builder_config)
    buffer = numpy.empty((delta0, delta1), dtype=buffer_d)
    counter = -1  # bin index
    with nogil:
        # i, j, idx are indices of the raw image uncorrected
        for idx in range(size_in):
            i = idx // shape_in1
            j = idx % shape_in1
            if do_mask and mask[i, j]:
                continue
            idx = i * shape_in1 + j  # pixel index
            buffer[:, :] = 0.0
            A0 = pos[i, j, 0, 0]
            A1 = pos[i, j, 0, 1]
            B0 = pos[i, j, 1, 0]
            B1 = pos[i, j, 1, 1]
            C0 = pos[i, j, 2, 0]
            C1 = pos[i, j, 2, 1]
            D0 = pos[i, j, 3, 0]
            D1 = pos[i, j, 3, 1]
            foffset0 = _floor_min4(A0, B0, C0, D0)
            foffset1 = _floor_min4(A1, B1, C1, D1)
            offset0 = <int> foffset0
            offset1 = <int> foffset1
            box_size0 = (<int> _ceil_max4(A0, B0, C0, D0)) - offset0
            box_size1 = (<int> _ceil_max4(A1, B1, C1, D1)) - offset1
            if (box_size0 > delta0) or (box_size1 > delta1):
                # Increase size of the buffer
                delta0 = offset0 if offset0 > delta0 else delta0
                delta1 = offset1 if offset1 > delta1 else delta1
                with gil:
                    buffer = numpy.zeros((delta0, delta1), dtype=buffer_d)

            A0 = A0 - foffset0
            A1 = A1 - foffset1
            B0 = B0 - foffset0
            B1 = B1 - foffset1
            C0 = C0 - foffset0
            C1 = C1 - foffset1
            D0 = D0 - foffset0
            D1 = D1 - foffset1
            _integrate2d(buffer, B0, B1, A0, A1)
            _integrate2d(buffer, C0, C1, B0, B1)
            _integrate2d(buffer, D0, D1, C0, C1)
            _integrate2d(buffer, A0, A1, D0, D1)
            area = 0.5 * ((C0 - A0) * (D1 - B1) - (C1 - A1) * (D0 - B0))
            inv_area = 1.0 / area
            for ms in range(box_size0):
                ml = ms + offset0
                if ml < 0 or ml >= shape_out0:
                    continue
                for ns in range(box_size1):
                    # ms,ns are indexes of the corrected image in short form, ml & nl are the same
                    nl = ns + offset1
                    if nl < 0 or nl >= shape_out1:
                        continue
                    value = buffer[ms, ns] * inv_area
                    if value == 0.0:
                        continue
                    if value < 0.0 or value > 1.0001:
                        # here we print pathological cases for debugging
                        if err_cnt < 1000:
                            with gil:
                                print(i, j, ms, box_size0, ns, box_size1, buffer[ms, ns], area, value, buffer[0, 0], buffer[0, 1], buffer[1, 0], buffer[1, 1])
                                print(" A0=%s; A1=%s; B0=%s; B1=%s; C0=%s; C1=%s; D0=%s; D1=%s" % (A0, A1, B0, B1, C0, C1, D0, D1))
                            err_cnt += 1
                        continue

                    bin_number = ml * shape_out1 + nl
                    # with gil: #Use the gil to perform an atomic operation
                    counter += 1
                    if counter >= large_size:
                        with gil:
                            raise RuntimeError("Provided temporary space for storage is not enough. " +
                                               "Please increase bins_per_pixel=%s. " % bins_per_pixel +
                                               "The suggested value is %i or greater." % ceil(1.1 * bins_per_pixel * size_in / idx))
                    builder.cinsert(bin_number, idx, value)
    logger.info("number of elements: %s, average per bin %.3f allocated max: %s",
                counter, counter / size_in, bins_per_pixel)

    if format == "csr":
        res = builder.to_csr()
    elif format == "lut":
        raise NotImplementedError("")
    else:
        raise RuntimeError("Unimplemented sparse matrix format: %s", format)
    return res


def resize_image_2D(image not None,
                    shape=None):
    """
    Reshape the image in such a way it has the required shape

    :param image: 2D-array with the image
    :param shape: expected shape of input image
    :return: 2D image with the proper shape
    """
    if shape is None:
        return image
    assert image.ndim == 2, "image is 2D"
    shape_in0, shape_in1 = shape
    shape_img0, shape_img1 = image.shape
    if (shape_img0 == shape_in0) and (shape_img1 == shape_in1):
        return image

    new_image = numpy.zeros((shape_in0, shape_in1), dtype=numpy.float32)
    if shape_img0 < shape_in0:
        if shape_img1 < shape_in1:
            new_image[:shape_img0, :shape_img1] = image
        else:
            new_image[:shape_img0, :] = image[:, :shape_in1]
    else:
        if shape_img1 < shape_in1:
            new_image[:, :shape_img1] = image[:shape_in0, :]
        else:
            new_image[:, :] = image[:shape_in0, :shape_in1]
    logger.warning("Patching image of shape %ix%i on expected size of %ix%i",
                   shape_img1, shape_img0, shape_in1, shape_in0)
    return new_image


def resize_image_3D(image not None,
                    shape=None):
    """
    Reshape the image in such a way it has the required shape
    This version is optimized for n-channel images used after preprocesing like:
    nlines * ncolumn * (value, variance, normalization)

    :param image: 3D-array with the preprocessed image
    :param shape: expected shape of input image (2D only)
    :return: 3D image with the proper shape
    """
    if shape is None:
        return image
    assert image.ndim == 3, "image is 3D"
    shape_in0, shape_in1 = shape
    shape_img0, shape_img1, nchan = image.shape
    if (shape_img0 == shape_in0) and (shape_img1 == shape_in1):
        return image

    new_image = numpy.zeros((shape_in0, shape_in1, nchan), dtype=numpy.float32)
    if shape_img0 < shape_in0:
        if shape_img1 < shape_in1:
            new_image[:shape_img0, :shape_img1, :] = image
        else:
            new_image[:shape_img0, :, :] = image[:, :shape_in1, :]
    else:
        if shape_img1 < shape_in1:
            new_image[:, :shape_img1, :] = image[:shape_in0, :, :]
        else:
            new_image[:, :, :] = image[:shape_in0, :shape_in1, :]
    logger.warning("Patching image of shape %ix%i on expected size of %ix%i",
                   shape_img1, shape_img0, shape_in1, shape_in0)
    return new_image


def correct(image, shape_in, shape_out, LUT not None, dummy=None, delta_dummy=None,
            method="double"):
    """Correct an image based on the look-up table calculated ...
    dispatch according to LUT type

    :param image: 2D-array with the image
    :param shape_in: shape of input image
    :param shape_out: shape of output image
    :param LUT: Look up table, here a 2D-array of struct
    :param dummy: value for invalid pixels
    :param delta_dummy: precision for invalid pixels
    :param method: integration method: can be "kahan" using single precision
            compensated for error or "double" in double precision (64 bits)

    :return: corrected 2D image
    """
    if (image.ndim == 3):
        # new generation of processing with (signal, variance, normalization)
        preprocessed_data = True
        image = resize_image_3D(image, shape_in)
    else:
        preprocessed_data = False
        image = resize_image_2D(image, shape_in)

    if len(LUT) == 3:
        # CSR format:
        if preprocessed_data:
            return correct_CSR_preproc_double(image, shape_out, LUT, dummy, delta_dummy)
        else:
            return correct_CSR(image, shape_in, shape_out, LUT, dummy, delta_dummy, method)
    else:
        # LUT format
        if preprocessed_data:
            shape_out0, shape_out1 = shape_out
            assert shape_out0 * shape_out1 == LUT.shape[0], "shape_out0 * shape_out1 == LUT.shape[0]"
#             if method == "kahan":
#                 return correct_LUT_preproc_kahan(image, shape_out, LUT, dummy, delta_dummy)
#             else:
#                 return correct_LUT_preproc_double(image, shape_out, LUT, dummy, delta_dummy)
            return correct_LUT_preproc_double(image, shape_out, LUT, dummy, delta_dummy)
        else:
            return correct_LUT(image, shape_in, shape_out, LUT, dummy, delta_dummy, method)


def correct_LUT(image, shape_in, shape_out, lut_t[:, ::1] LUT not None,
                dummy=None, delta_dummy=None, method="double"):
    """Correct an image based on the look-up table calculated ...
    dispatch between kahan and double

    :param image: 2D-array with the image
    :param shape_in: shape of input image
    :param shape_out: shape of output image
    :param LUT: Look up table, here a 2D-array of struct
    :param dummy: value for invalid pixels
    :param delta_dummy: precision for invalid pixels
    :param method: integration method: can be "kahan" using single precision
            compensated for error or "double" in double precision (64 bits)

    :return: corrected 2D image
    """
    shape_out0, shape_out1 = shape_out
    assert shape_out0 * shape_out1 == LUT.shape[0], "shape_out0 * shape_out1 == LUT.shape[0]"
    image = resize_image_2D(image, shape_in)
    if method == "kahan":
        return correct_LUT_kahan(image, shape_out, LUT, dummy, delta_dummy)
    else:
        return correct_LUT_double(image, shape_out, LUT, dummy, delta_dummy)


def correct_LUT_kahan(image, shape_out, lut_t[:, ::1] LUT not None,
                      dummy=None, delta_dummy=None):
    """Correct an image based on the look-up table calculated ...

    :param image: 2D-array with the image
    :param shape_in: shape of input image
    :param shape_out: shape of output image
    :param LUT: Look up table, here a 2D-array of struct
    :param dummy: value for invalid pixels
    :param delta_dummy: precision for invalid pixels
    :return: corrected 2D image
    """
    cdef:
        int i, j, idx, size
        float coef, sum, error, t, y, value, cdummy, cdelta_dummy
        float32_t[::1] lout, lin
        bint do_dummy = dummy is not None
    if do_dummy:
        cdummy = dummy
        if delta_dummy is None:
            cdelta_dummy = 0.0
    else:
        cdummy = cdelta_dummy = numpy.Nan
    assert numpy.prod(shape_out) == LUT.shape[0], "shape_out0 * shape_out1 == LUT.shape[0]"

    out = numpy.zeros(shape_out, dtype=numpy.float32)
    lout = out.ravel()
    lin = numpy.ascontiguousarray(image.ravel(), dtype=numpy.float32)
    size = lin.size
    for i in prange(LUT.shape[0], nogil=True, schedule="static"):
        sum = 0.0
        error = 0.0  # Implement Kahan summation
        for j in range(LUT.shape[1]):
            idx = LUT[i, j].idx
            coef = LUT[i, j].coef
            if coef <= 0:
                continue
            if idx >= size:
                with gil:
                    logger.warning("Accessing %i >= %i !!!" % (idx, size))
                    continue
            value = lin[idx]
            if do_dummy and fabs(value - cdummy) <= cdelta_dummy:
                continue
            y = value * coef - error
            t = sum + y
            error = (t - sum) - y
            sum = t
        if do_dummy and (sum == 0.0):
            sum = cdummy
        lout[i] += sum  # this += is for Cython's reduction
    return out


def correct_LUT_double(image, shape_out, lut_t[:, ::1] LUT not None,
                       dummy=None, delta_dummy=None):
    """Correct an image based on the look-up table calculated ...
    double precision accumulated

    :param image: 2D-array with the image
    :param shape_in: shape of input image
    :param shape_out: shape of output image
    :param LUT: Look up table, here a 2D-array of struct
    :param dummy: value for invalid pixels
    :param delta_dummy: precision for invalid pixels
    :return: corrected 2D image
    """
    cdef:
        int i, j, idx, size
        float value, cdummy, cdelta_dummy
        double sum, coef
        float32_t[::1] lout, lin
        bint do_dummy = dummy is not None
    if do_dummy:
        cdummy = dummy
        if delta_dummy is None:
            cdelta_dummy = 0.0
    else:
        cdummy = numpy.nan
        cdelta_dummy = 0.0

    assert numpy.prod(shape_out) == LUT.shape[0], "shape_out0 * shape_out1 == LUT.shape[0]"

    out = numpy.zeros(shape_out, dtype=numpy.float32)
    lout = out.ravel()
    lin = numpy.ascontiguousarray(image.ravel(), dtype=numpy.float32)
    size = lin.size
    for i in prange(LUT.shape[0], nogil=True, schedule="static"):
        sum = 0.0
        for j in range(LUT.shape[1]):
            idx = LUT[i, j].idx
            coef = LUT[i, j].coef
            if coef <= 0:
                continue
            if idx >= size:
                with gil:
                    logger.warning("Accessing %i >= %i !!!" % (idx, size))
                    continue
            value = lin[idx]
            if do_dummy and fabs(value - cdummy) <= cdelta_dummy:
                continue
            sum = value * coef + sum
        if do_dummy and (sum == 0.0):
            sum = cdummy
        lout[i] += sum  # this += is for Cython's reduction
    return out


def correct_CSR(image, shape_in, shape_out, LUT, dummy=None, delta_dummy=None,
                variance=None, method="double"):
    """
    Correct an image based on the look-up table calculated ...

    :param image: 2D-array with the image
    :param shape_in: shape of input image
    :param shape_out: shape of output image
    :param LUT: Look up table, here a 3-tuple array of ndarray
    :param dummy: value for invalid pixels
    :param delta_dummy: precision for invalid pixels
    :param variance: unused for now ... TODO: propagate variance.
    :param method: integration method: can be "kahan" using single precision compensated for error or "double" in double precision (64 bits)
    :return: corrected 2D image

    Nota: patch image on proper buffer size if needed.

    """
    image = resize_image_2D(image, shape_in)

    if method == "kahan":
        return correct_CSR_kahan(image, shape_out, LUT, dummy, delta_dummy)
    else:
        return correct_CSR_double(image, shape_out, LUT, dummy, delta_dummy)


def correct_CSR_kahan(image, shape_out, LUT, dummy=None, delta_dummy=None):
    """
    Correct an image based on the look-up table calculated ...
    using kahan's error compensated algorithm

    :param image: 2D-array with the image
    :param shape_in: shape of input image
    :param shape_out: shape of output image
    :param LUT: Look up table, here a 3-tuple array of ndarray
    :param dummy: value for invalid pixels
    :param delta_dummy: precision for invalid pixels
    :return: corrected 2D image
    """
    cdef:
        int i, j, idx, size, bins
        float coef, error, sum, y, t, value, cdummy, cdelta_dummy
        float32_t[::1] lout, lin, data
        int[::1] indices, indptr
        bint do_dummy = dummy is not None

    if do_dummy:
        cdummy = dummy
        if delta_dummy is None:
            cdelta_dummy = 0.0
    else:
        cdummy = numpy.nan
        cdelta_dummy = 0.0

    data, indices, indptr = LUT
    bins = indptr.shape[0] - 1

    out = numpy.zeros(shape_out, dtype=numpy.float32)
    lout = out.ravel()
    lin = numpy.ascontiguousarray(image.ravel(), dtype=numpy.float32)
    size = image.size

    for i in prange(bins, nogil=True, schedule="static"):
        sum = 0.0    # Implement Kahan summation
        error = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            idx = indices[j]
            coef = data[j]
            if coef <= 0:
                continue
            if idx >= size:
                with gil:
                    logger.warning("Accessing %i >= %i !!!" % (idx, size))
                    continue
            value = lin[idx]
            if do_dummy and fabs(value - cdummy) <= cdelta_dummy:
                continue
            y = value * coef - error
            t = sum + y
            error = (t - sum) - y
            sum = t
        if do_dummy and (sum == 0.0):
            sum = cdummy
        lout[i] += sum  # this += is for Cython's reduction
    return out


def correct_CSR_double(image, shape_out, LUT, dummy=None, delta_dummy=None):
    """
    Correct an image based on the look-up table calculated ...
    using double precision accumulator

    :param image: 2D-array with the image
    :param shape_in: shape of input image
    :param shape_out: shape of output image
    :param LUT: Look up table, here a 3-tuple array of ndarray
    :param dummy: value for invalid pixels
    :param delta_dummy: precision for invalid pixels
    :return: corrected 2D image
    """
    cdef:
        int i, j, idx, size, bins
        float value, cdummy, cdelta_dummy
        double coef, sum
        float32_t[::1] lout, lin, data
        int[::1] indices, indptr
        bint do_dummy = dummy is not None

    if do_dummy:
        cdummy = dummy
        if delta_dummy is None:
            cdelta_dummy = 0.0
    else:
        cdummy = numpy.nan
        cdelta_dummy = 0.0

    data, indices, indptr = LUT
    bins = indptr.size - 1
    assert numpy.prod(shape_out) == bins, "shape_out0*shape_out1 == indptr.size-1"
    out = numpy.zeros(shape_out, dtype=numpy.float32)
    lout = out.ravel()
    lin = numpy.ascontiguousarray(image.ravel(), dtype=numpy.float32)
    size = image.size

    for i in prange(bins, nogil=True, schedule="static"):
        sum = 0.0    # double precision
        for j in range(indptr[i], indptr[i + 1]):
            idx = indices[j]
            coef = data[j]
            if coef <= 0.0:
                continue
            if idx >= size:
                with gil:
                    logger.warning("Accessing %i >= %i !!!" % (idx, size))
                    continue
            value = lin[idx]
            if do_dummy and fabs(value - cdummy) <= cdelta_dummy:
                continue
            sum = sum + value * coef  # += operator not allowed in // sections
        if do_dummy and (sum == 0.0):
            sum = cdummy
        lout[i] += sum  # this += is for Cython's reduction
    return out


def correct_LUT_preproc_double(image, shape_out,
                               lut_t[:, ::1] LUT not None,
                               dummy=None, delta_dummy=None,
                               empty=numpy.nan):
    """Correct an image based on the look-up table calculated ...
    implementation using double precision accumulator

    :param image: 2D-array with the image (signal, variance, normalization)
    :param shape_in: shape of input image
    :param shape_out: shape of output image
    :param LUT: Look up table, here a 2D-array of struct
    :param dummy: value for invalid pixels
    :param delta_dummy: precision for invalid pixels
    :param empty: numerical value for empty pixels (if dummy is not provided)
    :param method: integration method: can be "kahan" using single precision
            compensated for error or "double" in double precision (64 bits)

    :return: corrected 2D image + array with (signal, variance, norm)
    """

    cdef:
        int i, j, idx, size, nchan
        float value, cdummy, cdelta_dummy
        double sum_sig, sum_var, sum_norm, coef
        float32_t[::1]  lout, lerr
        float32_t[:, ::1] lin, lprop
        bint do_dummy = dummy is not None

    if do_dummy:
        cdummy = dummy
        if delta_dummy is None:
            cdelta_dummy = 0.0
    else:
        cdummy = empty
    assert numpy.prod(shape_out) == LUT.shape[0], "shape_out0 * shape_out1 == LUT.shape[0]"

    nchan = image.shape[2]
    shape_out0, shape_out1 = shape_out

    prop = numpy.zeros((shape_out0, shape_out1, nchan), dtype=numpy.float32)
    lprop = prop.reshape((-1, nchan))
    out = numpy.zeros((shape_out0, shape_out1), dtype=numpy.float32)
    lout = out.ravel()
    lin = numpy.ascontiguousarray(image, dtype=numpy.float32).reshape((-1, nchan))
    if nchan == 3:
        err = numpy.zeros((shape_out0, shape_out1), dtype=numpy.float32)
        lerr = err.ravel()
    size = lin.shape[0]
    for i in prange(LUT.shape[0], nogil=True, schedule="static"):
        sum_sig = 0.0
        sum_var = 0.0
        sum_norm = 0.0
        for j in range(LUT.shape[1]):
            idx = LUT[i, j].idx
            coef = LUT[i, j].coef
            if coef <= 0:
                continue
            if idx >= size:
                with gil:
                    logger.warning("Accessing %i >= %i !!!" % (idx, size))
                    continue
            value = lin[idx, 0]
            if do_dummy and fabs(value - cdummy) <= cdelta_dummy:
                continue
            sum_sig = value * coef + sum_sig
            if nchan == 2:
                # case (signal, norm)
                sum_norm = coef * lin[idx, 1] + sum_norm
            elif nchan == 3:
                # case (signal, variance,  normalization)
                sum_var = coef * coef * lin[idx, 1] + sum_var
                sum_norm = coef * lin[idx, 2] + sum_norm
            else:
                sum_norm = sum_norm + coef

        if sum_norm == 0.0:  # No contribution to this output pixel
            lout[i] += cdummy  # this += is for Cython's reduction
            if nchan == 3:
                lerr[i] += cdummy
        else:
            lprop[i, 0] += sum_sig
            if nchan == 2:
                # case (signal, norm)
                lprop[i, 1] += sum_norm
                lout[i] += sum_sig / sum_norm
            elif nchan == 3:
                # case (signal, variance,  normalization)
                lprop[i, 1] += sum_var
                lprop[i, 2] += sum_norm
                lout[i] += sum_sig / sum_norm
                lerr[i] += sqrt(sum_var) / sum_norm
            else:
                # Case signal only. No normalization to behave like FIT2D does
                lout[i] += sum_sig

    if nchan == 3:
        return out, err, prop
    else:
        return out, prop


def correct_CSR_preproc_double(image, shape_out,
                               LUT not None,
                               dummy=None, delta_dummy=None,
                               empty=numpy.nan):
    """Correct an image based on the look-up table calculated ...
    implementation using double precision accumulator

    :param image: 2D-array with the image (signal, variance, normalization)
    :param shape_in: shape of input image
    :param shape_out: shape of output image
    :param LUT: Look up table, here a 3-tuple array of ndarray
    :param dummy: value for invalid pixels
    :param delta_dummy: precision for invalid pixels
    :param empty: numerical value for empty pixels (if dummy is not provided)
    :param method: integration method: can be "kahan" using single precision
            compensated for error or "double" in double precision (64 bits)

    :return: corrected 2D image + array with (signal, variance, norm)
    """

    cdef:
        int i, j, idx, size, bins, nchan
        float value, cdummy, cdelta_dummy
        double sum_sig, sum_var, sum_norm, coef
        float32_t[::1]  lout, lerr, data
        float32_t[:, ::1] lin, lprop
        int[::1] indices, indptr
        bint do_dummy = dummy is not None

    if do_dummy:
        cdummy = dummy
        if delta_dummy is None:
            cdelta_dummy = 0.0
    else:
        cdummy = empty
    data, indices, indptr = LUT
    bins = indptr.size - 1
    assert numpy.prod(shape_out) == bins, "shape_out0*shape_out1 == indptr.size-1"

    nchan = image.shape[2]
    shape_out0, shape_out1 = shape_out

    prop = numpy.zeros((shape_out0, shape_out1, nchan), dtype=numpy.float32)
    lprop = prop.reshape((-1, nchan))
    out = numpy.zeros((shape_out0, shape_out1), dtype=numpy.float32)
    lout = out.ravel()
    lin = numpy.ascontiguousarray(image, dtype=numpy.float32).reshape((-1, nchan))
    if nchan == 3:
        err = numpy.zeros((shape_out0, shape_out1), dtype=numpy.float32)
        lerr = err.ravel()
    size = lin.shape[0]

    for i in prange(bins, nogil=True, schedule="static"):
        sum_sig = 0.0
        sum_var = 0.0
        sum_norm = 0.0

        for j in range(indptr[i], indptr[i + 1]):
            idx = indices[j]
            coef = data[j]
            if coef <= 0.0:
                continue
            if idx >= size:
                with gil:
                    logger.warning("Accessing %i >= %i !!!" % (idx, size))
                    continue

            value = lin[idx, 0]
            if do_dummy and fabs(value - cdummy) <= cdelta_dummy:
                continue
            sum_sig = value * coef + sum_sig
            if nchan == 2:
                # case (signal, norm)
                sum_norm = coef * lin[idx, 1] + sum_norm
            elif nchan == 3:
                # case (signal, variance,  normalization)
                sum_var = coef * coef * lin[idx, 1] + sum_var
                sum_norm = coef * lin[idx, 2] + sum_norm
            else:
                sum_norm = sum_norm + coef

        if sum_norm == 0.0:  # No contribution to this output pixel
            lout[i] += cdummy  # this += is for Cython's reduction
            if nchan == 3:
                lerr[i] += cdummy
        else:
            lprop[i, 0] += sum_sig
            if nchan == 2:
                # case (signal, norm)
                lout[i] += sum_sig / sum_norm
                lprop[i, 1] += sum_norm
            elif nchan == 3:
                # case (signal, variance,  normalization)
                lprop[i, 1] += sum_var
                lprop[i, 2] += sum_norm
                lout[i] += sum_sig / sum_norm
                lerr[i] += sqrt(sum_var) / sum_norm
            else:
                # Case signal only. No normalization to behave like FIT2D does
                lout[i] += sum_sig

    if nchan == 3:
        return out, err, prop
    else:
        return out, prop


def uncorrect_LUT(image, shape, lut_t[:, :]LUT):
    """
    Take an image which has been corrected and transform it into it's raw (with loss of information)

    :param image: 2D-array with the image
    :param shape: shape of output image
    :param LUT: Look up table, here a 2D-array of struct
    :return: uncorrected 2D image and a mask (pixels in raw image not existing)
    """
    cdef:
        int idx, j
        float total, coef
        int8_t[::1] lmask
        float32_t[::1] lout, lin

    lin = numpy.ascontiguousarray(image, dtype=numpy.float32).ravel()
    out = numpy.zeros(shape, dtype=numpy.float32)
    mask = numpy.zeros(shape, dtype=numpy.int8)
    lmask = mask.ravel()
    lout = out.ravel()
    for idx in range(LUT.shape[0]):
        total = 0.0
        for j in range(LUT.shape[1]):
            coef = LUT[idx, j].coef
            if coef > 0:
                total += coef
        if total <= 0:
            lmask[idx] = 1
            continue
        val = lin[idx] / total
        for j in range(LUT.shape[1]):
            coef = LUT[idx, j].coef
            if coef > 0:
                lout[LUT[idx, j].idx] += val * coef
    return out, mask


def uncorrect_CSR(image, shape, LUT):
    """Take an image which has been corrected and transform it into it's raw (with loss of information)

    :param image: 2D-array with the image
    :param shape: shape of output image
    :param LUT: Look up table, here a 3-tuple of ndarray
    :return: uncorrected 2D image and a mask (pixels in raw image not existing)
    """
    cdef:
        int idx, j, nbins
        float total, coef
        int8_t[:] lmask
        float32_t[::1] lout, lin, data
        int32_t[::1] indices = LUT[1]
        int32_t[::1] indptr = LUT[2]
    out = numpy.zeros(shape, dtype=numpy.float32)
    lout = out.ravel()
    lin = numpy.ascontiguousarray(image, dtype=numpy.float32).ravel()
    mask = numpy.zeros(shape, dtype=numpy.int8)
    lmask = mask.ravel()
    data = LUT[0]
    nbins = indptr.size - 1
    for idx in range(nbins):
        total = 0.0
        for j in range(indptr[idx], indptr[idx + 1]):
            coef = data[j]
            if coef > 0:
                total += coef
        if total <= 0:
            lmask[idx] = 1
            continue
        val = lin[idx] / total
        for j in range(indptr[idx], indptr[idx + 1]):
            coef = data[j]
            if coef > 0:
                lout[indices[j]] += val * coef
    return out, mask

###########################################################################
# Deprecated but used to give correct results in the case of spline
###########################################################################


class Distortion(object):
    """

    This class applies a distortion correction on an image.

    It is also able to apply an inversion of the correction.

    """
    def __init__(self, detector="detector", shape=None):
        """
        :param detector: detector instance or detector name
        """
        if isinstance(detector, str):
            self.detector = detector_factory(detector)
        else:  # we assume it is a Detector instance
            self.detector = detector
        if shape:
            self.shape = shape
        elif "max_shape" in dir(self.detector):
            self.shape = self.detector.max_shape
        self.shape = tuple([int(i) for i in self.shape])
        self._sem = threading.Semaphore()
        self.lut_size = None
        self.pos = None
        self.LUT = None
        self.delta0 = self.delta1 = None  # max size of an pixel on a regular grid ...

    def __repr__(self):
        return os.linesep.join(["Distortion correction for detector:",
                                self.detector.__repr__()])

    def calc_pos(self):
        if self.pos is None:
            with self._sem:
                if self.pos is None:
                    pos_corners = numpy.empty((self.shape[0] + 1, self.shape[1] + 1, 2), dtype=numpy.float64)
                    d1 = expand2d(numpy.arange(self.shape[0] + 1.0), self.shape[1] + 1, False) - 0.5
                    d2 = expand2d(numpy.arange(self.shape[1] + 1.0), self.shape[0] + 1, True) - 0.5
                    p = self.detector.calc_cartesian_positions(d1, d2)
                    if p[2] is not None:
                        logger.warning("makes little sense to correct for distortion non-flat detectors: %s",
                                       self.detector)
                    pos_corners[:, :, 0], pos_corners[:, :, 1] = p[:2]
                    pos_corners[:, :, 0] /= self.detector.pixel1
                    pos_corners[:, :, 1] /= self.detector.pixel2
                    pos = numpy.empty((self.shape[0], self.shape[1], 4, 2), dtype=numpy.float32)
                    pos[:, :, 0, :] = pos_corners[:-1, :-1]
                    pos[:, :, 1, :] = pos_corners[:-1, 1:]
                    pos[:, :, 2, :] = pos_corners[1:, 1:]
                    pos[:, :, 3, :] = pos_corners[1:, :-1]
                    self.pos = pos
                    self.delta0 = int((numpy.ceil(pos_corners[1:, :, 0]) - numpy.floor(pos_corners[:-1, :, 0])).max())
                    self.delta1 = int((numpy.ceil(pos_corners[:, 1:, 1]) - numpy.floor(pos_corners[:, :-1, 1])).max())
        return self.pos

    def calc_LUT_size(self):
        """
        Considering the "half-CCD" spline from ID11 which describes a (1025,2048) detector,
        the physical location of pixels should go from:
        [-17.48634 : 1027.0543, -22.768829 : 2028.3689]
        We chose to discard pixels falling outside the [0:1025,0:2048] range with a lose of intensity

        We keep self.pos: pos_corners will not be compatible with systems showing non adjacent pixels (like some xpads)

        """
        cdef int i, j, k, l, shape0, shape1
        cdef int[:, ::1] pos0min, pos1min, pos0max, pos1max
        cdef int32_t[:, ::1] lut_size
        if self.pos is None:
            pos = self.calc_pos()
        else:
            pos = self.pos
        if self.lut_size is None:
            with self._sem:
                if self.lut_size is None:
                    shape0, shape1 = self.shape
                    pos0min = numpy.floor(pos[:, :, :, 0].min(axis=-1)).astype(numpy.int32).clip(0, self.shape[0])
                    pos1min = numpy.floor(pos[:, :, :, 1].min(axis=-1)).astype(numpy.int32).clip(0, self.shape[1])
                    pos0max = (numpy.ceil(pos[:, :, :, 0].max(axis=-1)).astype(numpy.int32) + 1).clip(0, self.shape[0])
                    pos1max = (numpy.ceil(pos[:, :, :, 1].max(axis=-1)).astype(numpy.int32) + 1).clip(0, self.shape[1])
                    lut_size = numpy.zeros(self.shape, dtype=numpy.int32)
                    with nogil:
                        for i in range(shape0):
                            for j in range(shape1):
                                for k in range(pos0min[i, j], pos0max[i, j]):
                                    for l in range(pos1min[i, j], pos1max[i, j]):
                                        lut_size[k, l] += 1
                    np_lut_size = numpy.asarray(lut_size)
                    self.lut_size = np_lut_size.max()
                    return np_lut_size

    def calc_LUT(self):
        cdef:
            int i, j, ms, ml, ns, nl, shape0, shape1, size
            int offset0, offset1, box_size0, box_size1
            int32_t k, idx = 0
            float A0, A1, B0, B1, C0, C1, D0, D1, area, inv_area, value
            float32_t[:, :, :, ::1] pos
            lut_t[:, :, ::1] lut
            int32_t[:, ::1] outMax = numpy.zeros(self.shape, dtype=numpy.int32)
            buffer_t[:, ::1] buffer
        shape0, shape1 = self.shape

        if self.lut_size is None:
            self.calc_LUT_size()
        if self.LUT is None:
            with self._sem:
                if self.LUT is None:
                    pos = self.pos
                    lut = numpy.recarray(shape=(self.shape[0], self.shape[1], self.lut_size), dtype=lut_d)
                    size = self.shape[0] * self.shape[1] * self.lut_size * sizeof(lut_t)
                    memset(&lut[0, 0, 0], 0, size)
                    logger.info("LUT shape: (%i,%i,%i) %.3f MByte" % (lut.shape[0], lut.shape[1], lut.shape[2], size / 1.0e6))
                    buffer = numpy.empty((self.delta0, self.delta1), dtype=buffer_d)
                    #buffer_size = self.delta0 * self.delta1 * sizeof(float)
                    logger.info("Max pixel size: %ix%i; Max source pixel in target: %i" % (buffer.shape[1], buffer.shape[0], self.lut_size))
                    with nogil:
                        # i,j, idx are indexes of the raw image uncorrected
                        for i in range(shape0):
                            for j in range(shape1):
                                # reinit of buffer
                                buffer[:, :] = 0
                                A0 = pos[i, j, 0, 0]
                                A1 = pos[i, j, 0, 1]
                                B0 = pos[i, j, 1, 0]
                                B1 = pos[i, j, 1, 1]
                                C0 = pos[i, j, 2, 0]
                                C1 = pos[i, j, 2, 1]
                                D0 = pos[i, j, 3, 0]
                                D1 = pos[i, j, 3, 1]
                                offset0 = (<int> floor(min(A0, B0, C0, D0)))
                                offset1 = (<int> floor(min(A1, B1, C1, D1)))
                                box_size0 = (<int> ceil(max(A0, B0, C0, D0))) - offset0
                                box_size1 = (<int> ceil(max(A1, B1, C1, D1))) - offset1
                                A0 -= <float> offset0
                                A1 -= <float> offset1
                                B0 -= <float> offset0
                                B1 -= <float> offset1
                                C0 -= <float> offset0
                                C1 -= <float> offset1
                                D0 -= <float> offset0
                                D1 -= <float> offset1
                                # ABCD is ANTI-trigonometric order: order input position accordingly
                                _integrate2d(buffer, B0, B1, A0, A1)
                                _integrate2d(buffer, C0, C1, B0, B1)
                                _integrate2d(buffer, D0, D1, C0, C1)
                                _integrate2d(buffer, A0, A1, D0, D1)
                                area = 0.5 * ((C0 - A0) * (D1 - B1) - (C1 - A1) * (D0 - B0))
                                inv_area = 1.0 / area
                                for ms in range(box_size0):
                                    ml = ms + offset0
                                    if ml < 0 or ml >= shape0:
                                        continue
                                    for ns in range(box_size1):
                                        # ms,ns are indexes of the corrected image in short form, ml & nl are the same
                                        nl = ns + offset1
                                        if nl < 0 or nl >= shape1:
                                            continue
                                        value = buffer[ms, ns] * inv_area
                                        if value <= 0:
                                            continue
                                        k = outMax[ml, nl]
                                        lut[ml, nl, k].idx = idx
                                        lut[ml, nl, k].coef = value
                                        outMax[ml, nl] = k + 1
                                idx += 1
                    self.LUT = numpy.asarray(lut).reshape(self.shape[0] * self.shape[1], self.lut_size)
        return self.LUT

################################################################################
# TODO: profile for select between ArrayBuilder and SparseBuilder
################################################################################

#     def demo_ArrayBuilder(self, int n=10):
#         "this just ensures the shared C-library works"
#         cdef:
#             ArrayBuilder ab
#             int i
#
#         ab = ArrayBuilder(n)
#         for i in range(n):
#             ab._append(i, i, 1.0)
#         return ab

    def correct(self, image):
        """
        Correct an image based on the look-up table calculated ...

        :param image: 2D-array with the image
        :return: corrected 2D image
        """
        cdef:
            int i, j, idx, size
            float coef
            lut_t[:, ::1] LUT
            float32_t[::1] lout, lin
        if self.LUT is None:
            self.calc_LUT()
        LUT = self.LUT
        img_shape = image.shape
        if (img_shape[0] < self.shape[0]) or (img_shape[1] < self.shape[1]):
            new_image = numpy.zeros(self.shape, dtype=numpy.float32)
            new_image[:img_shape[0], :img_shape[1]] = image
            image = new_image
            logger.warning("Patching image as image is %ix%i and spline is %ix%i" % (img_shape[1], img_shape[0], self.shape[1], self.shape[0]))

        out = numpy.zeros(self.shape, dtype=numpy.float32)
        lout = out.ravel()
        lin = numpy.ascontiguousarray(image.ravel(), dtype=numpy.float32)
        size = lin.size
        for i in prange(LUT.shape[0], nogil=True, schedule="static"):
            for j in range(LUT.shape[1]):
                idx = LUT[i, j].idx
                coef = LUT[i, j].coef
                if coef <= 0:
                    continue
                if idx >= size:
                    with gil:
                        logger.warning("Accessing %i >= %i !!!" % (idx, size))
                        continue
                lout[i] += lin[idx] * coef
        return out[:img_shape[0], :img_shape[1]]

    def uncorrect(self, image):
        """
        Take an image which has been corrected and transform it into it's raw (with loss of information)

        :param image: 2D-array with the image
        :return: uncorrected 2D image and a mask (pixels in raw image
        """
        if self.LUT is None:
            self.calc_LUT()
        out = numpy.zeros(self.shape, dtype=numpy.float32)
        mask = numpy.zeros(self.shape, dtype=numpy.int8)
        lmask = mask.ravel()
        lout = out.ravel()
        lin = image.ravel()
        tot = self.LUT.coef.sum(axis=-1)
        for idx in range(self.LUT.shape[0]):
            t = tot[idx]
            if t <= 0:
                lmask[idx] = 1
                continue
            val = lin[idx] / t
            lout[self.LUT[idx].idx] += val * self.LUT[idx].coef
        return out, mask
