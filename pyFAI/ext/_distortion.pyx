#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) 2013-2016 European Synchrotron Radiation Facility, Grenoble, France
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

__author__ = "Jerome Kieffer"
__license__ = "MIT"
__date__ = "13/05/2016"
__copyright__ = "2011-2016, ESRF"
__contact__ = "jerome.kieffer@esrf.fr"

import cython
cimport numpy
import numpy
from cython cimport view, floating
from cython.parallel import prange#, threadlocal
from cpython.ref cimport PyObject, Py_XDECREF
from libc.string cimport memset, memcpy
from libc.math cimport floor, ceil, fabs, copysign
import logging
import threading
import types
import os
import sys
import time
logger = logging.getLogger("pyFAI._distortion")
from ..detectors import detector_factory
from ..utils import expand2d
from ..decorators import timeit
try:
    from ..third_party import six
except ImportError:
    import six
import fabio

include "sparse_common.pxi"

cdef bint NEED_DECREF = sys.version_info < (2, 7) and numpy.version.version < "1.5"


cpdef inline float calc_area(float I1, float I2, float slope, float intercept) nogil:
    "Calculate the area between I1 and I2 of a line with a given slope & intercept"
    return 0.5 * (I2 - I1) * (slope * (I2 + I1) + 2 * intercept)


cpdef inline int clip(int value, int min_val, int max_val) nogil:
    "Limits the value to bounds"
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    else:
        return value


cdef inline float _floor_min4(float a, float b, float c, float d) nogil:
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


cdef inline float _ceil_max4(float a, float b, float c, float d) nogil:
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


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline void integrate(float[:, ::1] box, float start, float stop, float slope, float intercept) nogil:
    """Integrate in a box a line between start and stop, line defined by its slope & intercept

    @param box: buffer
    """
    cdef:
        int i, h = 0
        float P, dP, segment_area, abs_area, dA
        #, sign
    if start < stop:  # positive contribution
        P = ceil(start)
        dP = P - start
        if P > stop:  # start and stop are in the same unit
            segment_area = calc_area(start, stop, slope, intercept)
            if segment_area != 0.0:
                abs_area = fabs(segment_area)
                dA = (stop - start)  # always positive
                h = 0
                while abs_area > 0:
                    if dA > abs_area:
                        dA = abs_area
                        abs_area = -1
                    box[(<int> start), h] += copysign(dA, segment_area)
                    abs_area -= dA
                    h += 1
        else:
            if dP > 0:
                segment_area = calc_area(start, P, slope, intercept)
                if segment_area != 0.0:
                    abs_area = fabs(segment_area)
                    h = 0
                    dA = dP
                    while abs_area > 0:
                        if dA > abs_area:
                            dA = abs_area
                            abs_area = -1
                        box[(<int> P) - 1, h] += copysign(dA, segment_area)
                        abs_area -= dA
                        h += 1
            # subsection P1->Pn
            for i in range((<int> floor(P)), (<int> floor(stop))):
                segment_area = calc_area(i, i + 1, slope, intercept)
                if segment_area != 0:
                    abs_area = fabs(segment_area)
                    h = 0
                    dA = 1.0
                    while abs_area > 0:
                        if dA > abs_area:
                            dA = abs_area
                            abs_area = -1
                        box[i , h] += copysign(dA, segment_area)
                        abs_area -= dA
                        h += 1
            # Section Pn->B
            P = floor(stop)
            dP = stop - P
            if dP > 0:
                segment_area = calc_area(P, stop, slope, intercept)
                if segment_area != 0:
                    abs_area = fabs(segment_area)
                    h = 0
                    dA = fabs(dP)
                    while abs_area > 0:
                        if dA > abs_area:
                            dA = abs_area
                            abs_area = -1
                        box[(<int> P), h] += copysign(dA, segment_area)
                        abs_area -= dA
                        h += 1
    elif start > stop:  # negative contribution. Nota if start==stop: no contribution
        P = floor(start)
        if stop > P:  # start and stop are in the same unit
            segment_area = calc_area(start, stop, slope, intercept)
            if segment_area != 0:
                abs_area = fabs(segment_area)
#                 sign = segment_area / abs_area
                dA = (start - stop)  # always positive
                h = 0
                while abs_area > 0:
                    if dA > abs_area:
                        dA = abs_area
                        abs_area = -1
                    box[(<int> start), h] += copysign(dA, segment_area)
                    abs_area -= dA
                    h += 1
        else:
            dP = P - start
            if dP < 0:
                segment_area = calc_area(start, P, slope, intercept)
                if segment_area != 0:
                    abs_area = fabs(segment_area)
                    h = 0
                    dA = fabs(dP)
                    while abs_area > 0:
                        if dA > abs_area:
                            dA = abs_area
                            abs_area = -1
                        box[(<int> P) , h] += copysign(dA, segment_area)
                        abs_area -= dA
                        h += 1
            # subsection P1->Pn
            for i in range((<int> start), (<int> ceil(stop)), -1):
                segment_area = calc_area(i, i - 1, slope, intercept)
                if segment_area != 0:
                    abs_area = fabs(segment_area)
                    h = 0
                    dA = 1
                    while abs_area > 0:
                        if dA > abs_area:
                            dA = abs_area
                            abs_area = -1
                        box[i - 1, h] += copysign(dA, segment_area)
                        abs_area -= dA
                        h += 1
            # Section Pn->B
            P = ceil(stop)
            dP = stop - P
            if dP < 0:
                segment_area = calc_area(P, stop, slope, intercept)
                if segment_area != 0:
                    abs_area = fabs(segment_area)
                    h = 0
                    dA = fabs(dP)
                    while abs_area > 0:
                        if dA > abs_area:
                            dA = abs_area
                            abs_area = -1
                        box[(<int> stop), h] += copysign(dA, segment_area)
                        abs_area -= dA
                        h += 1


################################################################################
# Functions used in python classes from PyFAI.distortion
################################################################################

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def calc_pos(floating[:, :, :, ::1] pixel_corners not None,
             float pixel1, float pixel2, shape_out=None):
    """Calculate the pixel boundary position on the regular grid

    @param pixel_corners: pixel corner coordinate as detector.get_pixel_corner()
    @param shape: requested output shape. If None, it is calculated
    @param pixel1, pixel2: pixel size along row and column coordinates
    @return: pos, delta1, delta2, shape_out, offset
    """
    cdef:
        float[:, :, :, ::1] pos
        int i, j, k, dim0, dim1, nb_corners
        bint do_shape = (shape_out is None)
        float BIG = <float> sys.maxsize
        float min0, min1, max0, max1, delta0, delta1
        float all_min0, all_max0, all_max1, all_min1
        float p0, p1
    
    if (pixel1 == 0.0) or (pixel2 == 0):
        raise RuntimeError("Pixel size cannot be null -> Zero division error") 

    dim0 = pixel_corners.shape[0]
    dim1 = pixel_corners.shape[1]
    nb_corners = pixel_corners.shape[2]
    pos = numpy.zeros((dim0, dim1, 4, 2), dtype=numpy.float32)
    with nogil:
        delta0 = -BIG
        delta1 = -BIG
        all_min0 = BIG
        all_min0 = BIG
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


@cython.wraparound(False)
@cython.boundscheck(False)
def calc_size(floating[:, :, :, ::1] pos not None,
              shape,
              numpy.int8_t[:, ::1] mask=None,
              offset=None):
    """Calculate the number of items per output pixel

    @param pos: 4D array with position in space
    @param shape: shape of the output array
    @param mask: input data mask
    @param offset: 2-tuple of float with the minimal index of
    @return: number of input element per output elements
    """
    cdef:
        int i, j, k, l, shape_out0, shape_out1, shape_in0, shape_in1, min0, min1, max0, max1
        numpy.ndarray[numpy.int32_t, ndim = 2] lut_size = numpy.zeros(shape, dtype=numpy.int32)
        float A0, A1, B0, B1, C0, C1, D0, D1, offset0, offset1
        bint do_mask = mask is not None
        numpy.int8_t[:, ::1] cmask
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
                min0 = clip(<int> _floor_min4(A0, B0, C0, D0), 0, shape_out0)
                min1 = clip(<int> _floor_min4(A1, B1, C1, D1), 0, shape_out1)
                max0 = clip(<int> _ceil_max4(A0, B0, C0, D0) + 1, 0, shape_out0)
                max1 = clip(<int> _ceil_max4(A1, B1, C1, D1) + 1, 0, shape_out1)
                for k in range(min0, max0):
                    for l in range(min1, max1):
                        lut_size[k, l] += 1
    return lut_size


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def calc_LUT(float[:, :, :, ::1] pos not None, shape, bin_size, max_pixel_size,
             numpy.int8_t[:, :] mask=None):
    """
    @param pos: 4D position array
    @param shape: output shape
    @param bin_size: number of input element per output element (numpy array)
    @param max_pixel_size: (2-tuple of int) size of a buffer covering the largest pixel
    @param mask: arry with bad pixels marked as True
    @return: look-up table
    """
    cdef:
        int i, j, ms, ml, ns, nl, shape0, shape1, delta0, delta1
        int offset0, offset1, box_size0, box_size1, size, k
        numpy.int32_t idx = 0
        int err_cnt = 0
        float A0, A1, B0, B1, C0, C1, D0, D1, pAB, pBC, pCD, pDA, cAB, cBC, cCD, cDA, 
        float area, value, foffset0, foffset1
        lut_point[:, :, :] lut
        bint do_mask = mask is not None
        float[:, ::1] buffer
    size = bin_size.max()
    shape0, shape1 = shape
    if do_mask:
        assert shape0 == mask.shape[0]
        assert shape1 == mask.shape[1]
    delta0, delta1 = max_pixel_size
    cdef int[:, :] outMax = view.array(shape=(shape0, shape1), itemsize=sizeof(int), format="i")
    outMax[:, :] = 0
    buffer = numpy.empty((delta0, delta1), dtype=numpy.float32)    
    buffer_nbytes = buffer.nbytes
    if (size == 0): # fix 271
        raise RuntimeError("The look-up table has dimension 0 which is a non-sense."
                           + "Did you mask out all pixel or is your image out of the geometry range ?")
    lut = view.array(shape=(shape0, shape1, size), itemsize=sizeof(lut_point), format="if")
    lut_total_size = shape0 * shape1 * size * sizeof(lut_point)
    memset(&lut[0, 0, 0], 0, lut_total_size)
    logger.info("LUT shape: (%i,%i,%i) %.3f MByte" % (lut.shape[0], lut.shape[1], lut.shape[2], lut_total_size / 1.0e6))
    logger.info("Max pixel size: %ix%i; Max source pixel in target: %i" % (delta1, delta0, size))
    with nogil:
        # i,j, idx are indexes of the raw image uncorrected
        for i in range(shape0):
            for j in range(shape1):
                if do_mask and mask[i, j]:
                    continue
                #reset buffer
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
                offset0 = (<int> foffset0)
                offset1 = (<int> foffset1)
                box_size0 = (<int> _ceil_max4(A0, B0, C0, D0)) - offset0
                box_size1 = (<int> _ceil_max4(A1, B1, C1, D1)) - offset1
                if (box_size0 > delta0) or (box_size1 > delta1):
                    # Increase size of the buffer
                    delta0 = offset0 if offset0 > delta0 else delta0
                    delta1 = offset1 if offset1 > delta1 else delta1
                    with gil: 
                        buffer = numpy.zeros((delta0, delta1), dtype=numpy.float32)    

                A0 -= foffset0
                A1 -= foffset1
                B0 -= foffset0
                B1 -= foffset1
                C0 -= foffset0
                C1 -= foffset1
                D0 -= foffset0
                D1 -= foffset1
                if B0 != A0:
                    pAB = (B1 - A1) / (B0 - A0)
                    cAB = A1 - pAB * A0
                else:
                    pAB = cAB = 0.0
                if C0 != B0:
                    pBC = (C1 - B1) / (C0 - B0)
                    cBC = B1 - pBC * B0
                else:
                    pBC = cBC = 0.0
                if D0 != C0:
                    pCD = (D1 - C1) / (D0 - C0)
                    cCD = C1 - pCD * C0
                else:
                    pCD = cCD = 0.0
                if A0 != D0:
                    pDA = (A1 - D1) / (A0 - D0)
                    cDA = D1 - pDA * D0
                else:
                    pDA = cDA = 0.0

                # ABCD is trigonometric order: order input position accordingly
                integrate(buffer, B0, A0, pAB, cAB)
                integrate(buffer, C0, B0, pBC, cBC)
                integrate(buffer, D0, C0, pCD, cCD)
                integrate(buffer, A0, D0, pDA, cDA)

                area = 0.5 * ((C0 - A0) * (D1 - B1) - (C1 - A1) * (D0 - B0))

                for ms in range(box_size0):
                    ml = ms + offset0
                    if ml < 0 or ml >= shape0:
                        continue
                    for ns in range(box_size1):
                        # ms,ns are indexes of the corrected image in short form, ml & nl are the same
                        nl = ns + offset1
                        if nl < 0 or nl >= shape1:
                            continue
                        value = buffer[ms, ns] / area
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
    cdef numpy.ndarray[numpy.float64_t, ndim = 2] tmp_ary = numpy.empty(shape=(shape0 * shape1, size), dtype=numpy.float64)
    memcpy(&tmp_ary[0, 0], &lut[0, 0, 0], tmp_ary.nbytes)
    return numpy.core.records.array(tmp_ary.view(dtype=dtype_lut),
                                    shape=(shape0 * shape1, size), dtype=dtype_lut,
                                    copy=True)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def calc_CSR(float[:, :, :, :] pos not None, shape, bin_size, max_pixel_size,
             numpy.int8_t[:, :] mask=None):
    """Calculate the Look-up table as CSR format

    @param pos: 4D position array
    @param shape: output shape
    @param bin_size: number of input element per output element (as numpy array)
    @param max_pixel_size: (2-tuple of int) size of a buffer covering the largest pixel
    @return: look-up table in CSR format: 3-tuple of array"""
    cdef: 
        int shape0, shape1, delta0, delta1, bins
    shape0, shape1 = shape
    delta0, delta1 = max_pixel_size
    bins = shape0 * shape1
    cdef:
        int i, j, k, ms, ml, ns, nl, idx = 0, tmp_index, err_cnt=0
        int lut_size, offset0, offset1, box_size0, box_size1
        float A0, A1, B0, B1, C0, C1, D0, D1, pAB, pBC, pCD, pDA, cAB, cBC, cCD, cDA, 
        float area, value, foffset0, foffset1
        numpy.ndarray[numpy.int32_t, ndim = 1] indptr, indices
        numpy.ndarray[numpy.float32_t, ndim = 1] data
        int[:, :] outMax = view.array(shape=(shape0, shape1), itemsize=sizeof(int), format="i")
        float[:, ::1] buffer
        bint do_mask = mask is not None
    if do_mask:
        assert shape0 == mask.shape[0]
        assert shape1 == mask.shape[1]

    outMax[:, :] = 0
    indptr = numpy.empty(bins + 1, dtype=numpy.int32)
    indptr[0] = 0
    indptr[1:] = bin_size.cumsum(dtype=numpy.int32)
    lut_size = indptr[bins]

    indices = numpy.zeros(shape=lut_size, dtype=numpy.int32)
    data = numpy.zeros(shape=lut_size, dtype=numpy.float32)

    indptr[1:] = bin_size.cumsum(dtype=numpy.int32)

    logger.info("CSR matrix: %.3f MByte" % ((indices.nbytes + data.nbytes + indptr.nbytes) / 1.0e6))
    buffer = numpy.empty((delta0, delta1), dtype=numpy.float32)
    logger.info("Max pixel size: %ix%i; Max source pixel in target: %i" % (buffer.shape[1], buffer.shape[0], lut_size))
    with nogil:
        # i,j, idx are indices of the raw image uncorrected
        for i in range(shape0):
            for j in range(shape1):
                if do_mask and mask[i, j]:
                    continue
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
                foffset0 = _floor_min4(A0, B0, C0, D0)
                foffset1 = _floor_min4(A1, B1, C1, D1)
                offset0 = (<int> foffset0)
                offset1 = (<int> foffset1)
                box_size0 = (<int> _ceil_max4(A0, B0, C0, D0)) - offset0
                box_size1 = (<int> _ceil_max4(A1, B1, C1, D1)) - offset1

                if (box_size0 > delta0) or (box_size1 > delta1):
                    # Increase size of the buffer
                    delta0 = offset0 if offset0 > delta0 else delta0
                    delta1 = offset1 if offset1 > delta1 else delta1
                    with gil: 
                        buffer = numpy.zeros((delta0, delta1), dtype=numpy.float32)    

                A0 -= foffset0
                A1 -= foffset1
                B0 -= foffset0
                B1 -= foffset1
                C0 -= foffset0
                C1 -= foffset1
                D0 -= foffset0
                D1 -= foffset1
                if B0 != A0:
                    pAB = (B1 - A1) / (B0 - A0)
                    cAB = A1 - pAB * A0
                else:
                    pAB = cAB = 0.0
                if C0 != B0:
                    pBC = (C1 - B1) / (C0 - B0)
                    cBC = B1 - pBC * B0
                else:
                    pBC = cBC = 0.0
                if D0 != C0:
                    pCD = (D1 - C1) / (D0 - C0)
                    cCD = C1 - pCD * C0
                else:
                    pCD = cCD = 0.0
                if A0 != D0:
                    pDA = (A1 - D1) / (A0 - D0)
                    cDA = D1 - pDA * D0
                else:
                    pDA = cDA = 0.0
                integrate(buffer, B0, A0, pAB, cAB)
                integrate(buffer, A0, D0, pDA, cDA)
                integrate(buffer, D0, C0, pCD, cCD)
                integrate(buffer, C0, B0, pBC, cBC)
                area = 0.5 * ((C0 - A0) * (D1 - B1) - (C1 - A1) * (D0 - B0))
                for ms in range(box_size0):
                    ml = ms + offset0
                    if ml < 0 or ml >= shape0:
                        continue
                    for ns in range(box_size1):
                        # ms,ns are indexes of the corrected image in short form, ml & nl are the same
                        nl = ns + offset1
                        if nl < 0 or nl >= shape1:
                            continue
                        value = buffer[ms, ns] / area
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
    return (data, indices, indptr)


@cython.boundscheck(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def calc_openmp(float[:, :, :, ::1] pos not None,
                shape,
                max_pixel_size=(8, 8),
                numpy.int8_t[:, :] mask=None,
                format="csr",
                int bins_per_pixel=8):
    """Calculate the look-up table (or CSR) using OpenMP

    @param pos: 4D position array
    @param shape: output shape
    @param max_pixel_size: (2-tuple of int) size of a buffer covering the largest pixel
    @param format: can be "CSR" or "LUT"
    @param bins_per_pixel: average splitting factor (number of pixels per bin)
    @return: look-up table in CSR/LUT format
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
        int i, j, k, ms, ml, ns, nl
        int i0, i1, lut_size, offset0, offset1, box_size0, box_size1
        int counter, bin_number
        int idx, err_cnt=0
        float A0, A1, B0, B1, C0, C1, D0, D1, pAB, pBC, pCD, pDA, cAB, cBC, cCD, cDA, 
        float area, value, foffset0, foffset1
        int[::1] indptr, indices, idx_bin, idx_pixel, pixel_count
        float[::1] data, large_data
        float[:, ::1] buffer
        bint do_mask = mask is not None
        lut_point[:, :] lut
    if do_mask:
        assert shape_in0 == mask.shape[0]
        assert shape_in1 == mask.shape[1]

    #count the number of pixel falling into every single bin
    pixel_count = numpy.zeros(bins, dtype=numpy.int32)
    idx_pixel = numpy.zeros(large_size, dtype=numpy.int32)
    idx_bin = numpy.zeros(large_size, dtype=numpy.int32)
    large_data = numpy.zeros(large_size, dtype=numpy.float32)
    logger.info("Temporary storage: %.3fMB",
                (large_data.nbytes + pixel_count.nbytes + idx_pixel.nbytes + idx_bin.nbytes) / 1e6)

    buffer = numpy.empty((delta0, delta1), dtype=numpy.float32)
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
                    buffer = numpy.zeros((delta0, delta1), dtype=numpy.float32)    

            A0 = A0 - foffset0
            A1 = A1 - foffset1
            B0 = B0 - foffset0
            B1 = B1 - foffset1
            C0 = C0 - foffset0
            C1 = C1 - foffset1
            D0 = D0 - foffset0
            D1 = D1 - foffset1
            if B0 != A0:
                pAB = (B1 - A1) / (B0 - A0)
                cAB = A1 - pAB * A0
            else:
                pAB = cAB = 0.0
            if C0 != B0:
                pBC = (C1 - B1) / (C0 - B0)
                cBC = B1 - pBC * B0
            else:
                pBC = cBC = 0.0
            if D0 != C0:
                pCD = (D1 - C1) / (D0 - C0)
                cCD = C1 - pCD * C0
            else:
                pCD = cCD = 0.0
            if A0 != D0:
                pDA = (A1 - D1) / (A0 - D0)
                cDA = D1 - pDA * D0
            else:
                pDA = cDA = 0.0
            
            integrate(buffer, B0, A0, pAB, cAB)
            integrate(buffer, A0, D0, pDA, cDA)
            integrate(buffer, D0, C0, pCD, cCD)
            integrate(buffer, C0, B0, pBC, cBC)
            area = 0.5 * ((C0 - A0) * (D1 - B1) - (C1 - A1) * (D0 - B0))
            for ms in range(box_size0):
                ml = ms + offset0
                if ml < 0 or ml >= shape_out0:
                    continue
                for ns in range(box_size1):
                    # ms,ns are indexes of the corrected image in short form, ml & nl are the same
                    nl = ns + offset1
                    if nl < 0 or nl >= shape_out1:
                        continue
                    value = buffer[ms, ns] / area
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
#                     with gil: #Use the gil to perform an atomic operation
                    counter += 1
                    pixel_count[bin_number] += 1
                    if counter >= large_size:
                        with gil:
                            raise RuntimeError("Provided temporary space for storage is not enough. " +
                                               "Please increase bins_per_pixel=%s. "%bins_per_pixel +
                                               "The suggested value is %i or greater."%ceil(1.1*bins_per_pixel*size_in/idx))
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
        lut = numpy.zeros(shape=(bins, lut_size), dtype=dtype_lut)
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


@cython.wraparound(False)
@cython.boundscheck(False)
def correct_LUT(image, shape_in, shape_out, lut_point[:, ::1] LUT not None, dummy=None, delta_dummy=None):
    """Correct an image based on the look-up table calculated ...

    @param image: 2D-array with the image
    @param shape_in: shape of input image
    @param shape_out: shape of output image
    @param LUT: Look up table, here a 2D-array of struct
    @param dummy: value for invalid pixels
    @param delta_dummy: precision for invalid pixels
    @return: corrected 2D image
    """
    cdef:
        int i, j, lshape0, lshape1, idx, size
        int shape_in0, shape_in1, shape_out0, shape_out1, shape_img0, shape_img1
        float coef, sum, error, t, y, value, cdummy, cdelta_dummy
        float[::1] lout, lin
        bint do_dummy = dummy is not None
    if do_dummy:
        cdummy = dummy
        if delta_dummy is None:
            cdelta_dummy = 0.0
    shape_in0, shape_in1 = shape_in
    shape_out0, shape_out1 = shape_out
    lshape0 = LUT.shape[0]
    lshape1 = LUT.shape[1]
    assert shape_out0 * shape_out1 == LUT.shape[0]
    shape_img0, shape_img1 = image.shape
    if (shape_img0 != shape_in0) or (shape_img1 != shape_in1):
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
        logger.warning("Patching image as image is %ix%i and expected input is %ix%i and output is %ix%i",
                       shape_img1, shape_img0, shape_in1, shape_in0, shape_out1, shape_out0)
        image = new_image

    out = numpy.zeros((shape_out0, shape_out1), dtype=numpy.float32)
    lout = out.ravel()
    lin = numpy.ascontiguousarray(image.ravel(), dtype=numpy.float32)
    size = lin.size
    assert size == shape_in0 * shape_in1
    for i in prange(lshape0, nogil=True, schedule="static"):
        sum = 0.0
        error = 0.0  # Implement Kahan summation
        for j in range(lshape1):
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


@cython.wraparound(False)
@cython.boundscheck(False)
def correct_CSR(image, shape_in, shape_out, LUT, dummy=None, delta_dummy=None):
    """
    Correct an image based on the look-up table calculated ...

    @param image: 2D-array with the image
    @param shape_in: shape of input image
    @param shape_out: shape of output image
    @param LUT: Look up table, here a 3-tuple array of ndarray
    @param dummy: value for invalid pixels
    @param delta_dummy: precision for invalid pixels
    @return: corrected 2D image
    """
    cdef:
        int i, j, idx, size, bins
        int shape_in0, shape_in1, shape_out0, shape_out1, shape_img0, shape_img1
        float coef, tmp, error, sum, y, t, value, cdummy, cdelta_dummy
        float[::1] lout, lin, data
        int[::1] indices, indptr
        bint do_dummy = dummy is not None

    if do_dummy:
        cdummy = dummy
        if delta_dummy is None:
            cdelta_dummy = 0.0

    data, indices, indptr = LUT
    bins = indptr.size - 1
    shape_in0, shape_in1 = shape_in
    shape_out0, shape_out1 = shape_out
    shape_img0, shape_img1 = image.shape
    if (shape_img0 != shape_in0) or (shape_img1 != shape_in1):
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
        logger.warning("Patching image as image is %ix%i and expected input is %ix%i and output is %ix%i",
                       shape_img1, shape_img0, shape_in1, shape_in0, shape_out1, shape_out0)
        image = new_image

    out = numpy.zeros((shape_out0, shape_out1), dtype=numpy.float32)
    lout = out.ravel()
    lin = numpy.ascontiguousarray(image.ravel(), dtype=numpy.float32)
    size = lin.size
    assert size == shape_in0 * shape_in1

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


def uncorrect_LUT(image, shape, lut_point[:, :]LUT):
    """
    Take an image which has been corrected and transform it into it's raw (with loss of information)
    @param image: 2D-array with the image
    @param shape: shape of output image
    @param LUT: Look up table, here a 2D-array of struct
    @return: uncorrected 2D image and a mask (pixels in raw image not existing)
    """
    cdef int idx, j
    cdef float total, coef
    out = numpy.zeros(shape, dtype=numpy.float32)
    mask = numpy.zeros(shape, dtype=numpy.int8)
    cdef numpy.int8_t[:] lmask = mask.ravel()
    cdef float[:] lout = out.ravel()
    cdef float[:] lin = numpy.ascontiguousarray(image, dtype=numpy.float32).ravel()

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
    """
    Take an image which has been corrected and transform it into it's raw (with loss of information)
    @param image: 2D-array with the image
    @param shape: shape of output image
    @param LUT: Look up table, here a 3-tuple of ndarray
    @return: uncorrected 2D image and a mask (pixels in raw image not existing)
    """
    cdef:
        int idx, j, nbins
        float total, coef
        numpy.int8_t[:] lmask
        float[:] lout, lin, data
        numpy.int32_t[:] indices = LUT[1]
        numpy.int32_t[:] indptr = LUT[2]
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
        @param detector: detector instance or detector name
        """
        if isinstance(detector, six.string_types):
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
                    if p[-1] is not None:
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

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def calc_LUT_size(self):
        """
        Considering the "half-CCD" spline from ID11 which describes a (1025,2048) detector,
        the physical location of pixels should go from:
        [-17.48634 : 1027.0543, -22.768829 : 2028.3689]
        We chose to discard pixels falling outside the [0:1025,0:2048] range with a lose of intensity

        We keep self.pos: pos_corners will not be compatible with systems showing non adjacent pixels (like some xpads)

        """
        cdef int i, j, k, l, shape0, shape1
        cdef numpy.ndarray[numpy.float32_t, ndim = 4] pos
        cdef int[:, :] pos0min, pos1min, pos0max, pos1max
        cdef numpy.ndarray[numpy.int32_t, ndim = 2] lut_size
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
                    self.lut_size = lut_size.max()
                    return lut_size

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    def calc_LUT(self):
        cdef:
            int i, j, ms, ml, ns, nl, shape0, shape1, delta0, delta1, buffer_size, i0, i1, size
            int offset0, offset1, box_size0, box_size1
            numpy.int32_t k, idx = 0
            float A0, A1, B0, B1, C0, C1, D0, D1, pAB, pBC, pCD, pDA, cAB, cBC, cCD, cDA, area, value
            float[:, :, :, :] pos
            numpy.ndarray[lut_point, ndim = 3] lut
            numpy.ndarray[numpy.int32_t, ndim = 2] outMax = numpy.zeros(self.shape, dtype=numpy.int32)
            float[:, ::1] buffer
        shape0, shape1 = self.shape

        if self.lut_size is None:
            self.calc_LUT_size()
        if self.LUT is None:
            with self._sem:
                if self.LUT is None:
                    pos = self.pos
                    lut = numpy.recarray(shape=(self.shape[0], self.shape[1], self.lut_size), dtype=[("idx", numpy.int32), ("coef", numpy.float32)])
                    size = self.shape[0] * self.shape[1] * self.lut_size * sizeof(lut_point)
                    memset(&lut[0, 0, 0], 0, size)
                    logger.info("LUT shape: (%i,%i,%i) %.3f MByte" % (lut.shape[0], lut.shape[1], lut.shape[2], size / 1.0e6))
                    buffer = numpy.empty((self.delta0, self.delta1), dtype=numpy.float32)
                    buffer_size = self.delta0 * self.delta1 * sizeof(float)
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
                                if B0 != A0:
                                    pAB = (B1 - A1) / (B0 - A0)
                                    cAB = A1 - pAB * A0
                                else:
                                    pAB = cAB = 0.0
                                if C0 != B0:
                                    pBC = (C1 - B1) / (C0 - B0)
                                    cBC = B1 - pBC * B0
                                else:
                                    pBC = cBC = 0.0
                                if D0 != C0:
                                    pCD = (D1 - C1) / (D0 - C0)
                                    cCD = C1 - pCD * C0
                                else:
                                    pCD = cCD = 0.0
                                if A0 != D0:
                                    pDA = (A1 - D1) / (A0 - D0)
                                    cDA = D1 - pDA * D0
                                else:
                                    pDA = cDA = 0.0
                                # ABCD is ANTI-trigonometric order: order input position accordingly
                                integrate(buffer, B0, A0, pAB, cAB)
                                integrate(buffer, A0, D0, pDA, cDA)
                                integrate(buffer, D0, C0, pCD, cCD)
                                integrate(buffer, C0, B0, pBC, cBC)
                                area = 0.5 * ((C0 - A0) * (D1 - B1) - (C1 - A1) * (D0 - B0))
                                for ms in range(box_size0):
                                    ml = ms + offset0
                                    if ml < 0 or ml >= shape0:
                                        continue
                                    for ns in range(box_size1):
                                        # ms,ns are indexes of the corrected image in short form, ml & nl are the same
                                        nl = ns + offset1
                                        if nl < 0 or nl >= shape1:
                                            continue
                                        value = buffer[ms, ns] / area
                                        if value <= 0:
                                            continue
                                        k = outMax[ml, nl]
                                        lut[ml, nl, k].idx = idx
                                        lut[ml, nl, k].coef = value
                                        outMax[ml, nl] = k + 1
                                idx += 1
                    self.LUT = lut.reshape(self.shape[0] * self.shape[1], self.lut_size)
        return self.LUT

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def correct(self, image):
        """
        Correct an image based on the look-up table calculated ...

        @param image: 2D-array with the image
        @return: corrected 2D image
        """
        cdef:
            int i, j, lshape0, lshape1, idx, size
            float coef
            lut_point[:, :] LUT
            float[:] lout, lin
        if self.LUT is None:
            self.calc_LUT()
        LUT = self.LUT
        lshape0 = LUT.shape[0]
        lshape1 = LUT.shape[1]
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
        for i in prange(lshape0, nogil=True, schedule="static"):
            for j in range(lshape1):
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

    @timeit
    def uncorrect(self, image):
        """
        Take an image which has been corrected and transform it into it's raw (with loss of information)

        @param image: 2D-array with the image
        @return: uncorrected 2D image and a mask (pixels in raw image
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
