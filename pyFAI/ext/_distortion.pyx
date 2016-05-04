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
__date__ = "04/05/2016"
__copyright__ = "2011-2016, ESRF"
__contact__ = "jerome.kieffer@esrf.fr"

import cython
cimport numpy
import numpy
from cython cimport view, floating
from cython.parallel import prange
from cpython.ref cimport PyObject, Py_XDECREF
from libc.string cimport memset, memcpy
from libc.math cimport floor, ceil, fabs
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

cdef struct lut_point:
    numpy.int32_t idx
    numpy.float32_t coef

dtype_lut = numpy.dtype([("idx", numpy.int32), ("coef", numpy.float32)])
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


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline void integrate(float[:, :] box, float start, float stop, float slope, float intercept) nogil:
    "Integrate in a box a line between start and stop, line defined by its slope & intercept "
    cdef:
        int i, h = 0
        float P, dP, A, AA, dA, sign
    if start < stop:  # positive contribution
        P = ceil(start)
        dP = P - start
        if P > stop:  # start and stop are in the same unit
            A = calc_area(start, stop, slope, intercept)
            if A != 0:
                AA = fabs(A)
                sign = A / AA
                dA = (stop - start)  # always positive
                h = 0
                while AA > 0:
                    if dA > AA:
                        dA = AA
                        AA = -1
                    box[(<int> floor(start)), h] += sign * dA
                    AA -= dA
                    h += 1
        else:
            if dP > 0:
                A = calc_area(start, P, slope, intercept)
                if A != 0:
                    AA = fabs(A)
                    sign = A / AA
                    h = 0
                    dA = dP
                    while AA > 0:
                        if dA > AA:
                            dA = AA
                            AA = -1
                        box[(<int> floor(P)) - 1, h] += sign * dA
                        AA -= dA
                        h += 1
            # subsection P1->Pn
            for i in range((<int> floor(P)), (<int> floor(stop))):
                A = calc_area(i, i + 1, slope, intercept)
                if A != 0:
                    AA = fabs(A)
                    sign = A / AA

                    h = 0
                    dA = 1.0
                    while AA > 0:
                        if dA > AA:
                            dA = AA
                            AA = -1
                        box[i , h] += sign * dA
                        AA -= dA
                        h += 1
            # Section Pn->B
            P = floor(stop)
            dP = stop - P
            if dP > 0:
                A = calc_area(P, stop, slope, intercept)
                if A != 0:
                    AA = fabs(A)
                    sign = A / AA
                    h = 0
                    dA = fabs(dP)
                    while AA > 0:
                        if dA > AA:
                            dA = AA
                            AA = -1
                        box[(<int> floor(P)), h] += sign * dA
                        AA -= dA
                        h += 1
    elif start > stop:  # negative contribution. Nota is start=stop: no contribution
        P = floor(start)
        if stop > P:  # start and stop are in the same unit
            A = calc_area(start, stop, slope, intercept)
            if A != 0:
                AA = fabs(A)
                sign = A / AA
                dA = (start - stop)  # always positive
                h = 0
                while AA > 0:
                    if dA > AA:
                        dA = AA
                        AA = -1
                    box[(<int> floor(start)), h] += sign * dA
                    AA -= dA
                    h += 1
        else:
            dP = P - start
            if dP < 0:
                A = calc_area(start, P, slope, intercept)
                if A != 0:
                    AA = fabs(A)
                    sign = A / AA
                    h = 0
                    dA = fabs(dP)
                    while AA > 0:
                        if dA > AA:
                            dA = AA
                            AA = -1
                        box[(<int> floor(P)) , h] += sign * dA
                        AA -= dA
                        h += 1
            # subsection P1->Pn
            for i in range((<int> start), (<int> ceil(stop)), -1):
                A = calc_area(i, i - 1, slope, intercept)
                if A != 0:
                    AA = fabs(A)
                    sign = A / AA
                    h = 0
                    dA = 1
                    while AA > 0:
                        if dA > AA:
                            dA = AA
                            AA = -1
                        box[i - 1, h] += sign * dA
                        AA -= dA
                        h += 1
            # Section Pn->B
            P = ceil(stop)
            dP = stop - P
            if dP < 0:
                A = calc_area(P, stop, slope, intercept)
                if A != 0:
                    AA = fabs(A)
                    sign = A / AA
                    h = 0
                    dA = fabs(dP)
                    while AA > 0:
                        if dA > AA:
                            dA = AA
                            AA = -1
                        box[(<int> floor(stop)), h] += sign * dA
                        AA -= dA
                        h += 1


################################################################################
# Functions used in python classes from PyFAI.distortion
################################################################################

def calc_pos(floating[:, :, :, ::1] pixel_corners not None, 
             shape_out=None, 
             float pixel1, float pixel2,
             offset=None):
    """Calculate the pixel boundary position on the regular grid
    
    @param pixel_corners: pixel corner coordinate as detector.get_pixel_corner
    @param shape: requested output shape. If None, it is calculated
    @param pixel1, pixel2: pixel size along row and column coordinates 
    @return: pos, delta1, delta2. shape_out
    """
    cdef: 
        numpy.ndarray[numpy.float32_t, ndim = 4] pos
        int i, j, k, dim0, dim1
        bint do_shape = (shape_out is None)
        float BIG = <float> sys.maxsize
        float min0, min1, max0, max1, delta0, delta1
        float all_min0, all_max0, all_max1
    shape_in = pixel_corners[:2]
    pos = numpy.zeros((dim0, dim1, 4, 2), dtype=numpy.float32)
    
    delat0 = 0.0
    delta1 = 0.0
    all_min0 = BIG
    all_min0 = BIG
    all_max0 = 0.0
    all_max1 = 0.0
    for i in range(dim0):
        for j in range(dim1):
            min0 = BIG
            min1 = BIG
            max0 = 0
            max1 = 0
            for k range(4):
                p0 = pixel_corners[i, j, k, 1] / pixel1
                p1 = pixel_corners[i, j, k, 2] / pixel2
                pos[i, j, k, 0] += p0
                pos[i, j, k, 1] += p1
                min0 = min(min0, p0)
                min1 = min(min1, p1)
                max0 = max(max0, p0)
                max1 = max(max1, p1)
            delta0 = max(delta0, max0 - min0)
            delta1 = max(delta1, max1 - min1)   
            if do_shape:
                all_min0 = min(all_min0, min0)
                all_min1 = min(all_min1, min1)
                all_max0 = max(all_max0, max0)
                all_max1 = max(all_max1, max1) 
                    
    return pos, delta0, delta1, (all_max0 - all_min0, all_max1 - all_min1) if do_shape else shape_out
 
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
    
    shape_in0, shape_in1 = pos.shape[0], pos.shape[1]
    shape_out0, shape_out1 = shape
    
    if do_mask and ((mask.shape[0] != shape_in0) or (mask.shape[1] != shape_in1)): 
        err = 'Mismatch between shape of detector (%s, %s) and shape of mask (%s, %s)' % (shape_in0, shape_in1, mask.shape[0], mask.shape[1])
        logger.error(err)
        raise RuntimeError(err)
    
    if offset is not None:
        offset0, offset1 = offset
    
    with nogil:
        for i in range(shape_in0):
            for j in range(shape_in1):
                if do_mask and mask[i, j]:
                    continue
                A0 = pos[i, j, 0, 0] - offset0
                A1 = pos[i, j, 0, 1] - offset1
                B0 = pos[i, j, 1, 0] - offset0
                B1 = pos[i, j, 1, 1] - offset1
                C0 = pos[i, j, 2, 0] - offset0
                C1 = pos[i, j, 2, 1] - offset1
                D0 = pos[i, j, 3, 0] - offset0
                D1 = pos[i, j, 3, 1] - offset1
                min0 = clip(<int> floor(min(A0, B0, C0, D0)), 0, shape_out0)
                min1 = clip(<int> floor(min(A1, B1, C1, D1)), 0, shape_out1)
                max0 = clip(<int> ceil(max(A0, B0, C0, D0)) + 1, 0, shape_out0)
                max1 = clip(<int> ceil(max(A1, B1, C1, D1)) + 1, 0, shape_out1)
                for k in range(min0, max0):
                    for l in range(min1, max1):
                        lut_size[k, l] += 1
    return lut_size


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def calc_LUT(float[:, :, :, :] pos not None, shape, bin_size, max_pixel_size,
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
        int i, j, ms, ml, ns, nl, shape0, shape1, delta0, delta1, buffer_size, i0, i1
        int offset0, offset1, box_size0, box_size1, size, k
        numpy.int32_t idx = 0
        float A0, A1, B0, B1, C0, C1, D0, D1, pAB, pBC, pCD, pDA, cAB, cBC, cCD, cDA, area, value
        lut_point[:, :, :] lut
        bint do_mask = mask is not None
    size = bin_size.max()
    shape0, shape1 = shape
    if do_mask:
        assert shape0 == mask.shape[0]
        assert shape1 == mask.shape[1]
    delta0, delta1 = max_pixel_size
    cdef int[:, :] outMax = view.array(shape=(shape0, shape1), itemsize=sizeof(int), format="i")
    outMax[:, :] = 0
    cdef float[:, :] buffer = view.array(shape=(delta0, delta1), itemsize=sizeof(float), format="f")
    if (size == 0): #fix 271
            raise RuntimeError("The look-up table has dimension 0 which is a non-sense."
                               + "Did you mask out all pixel or is your image out of the geometry range ?")
    lut = view.array(shape=(shape0, shape1, size), itemsize=sizeof(lut_point), format="if")
    lut_total_size = shape0 * shape1 * size * sizeof(lut_point)
    memset(&lut[0, 0, 0], 0, lut_total_size)
    logger.info("LUT shape: (%i,%i,%i) %.3f MByte" % (lut.shape[0], lut.shape[1], lut.shape[2], lut_total_size / 1.0e6))
    buffer_size = delta0 * delta1 * sizeof(float)
    logger.info("Max pixel size: %ix%i; Max source pixel in target: %i" % (delta1, delta0, size))
    with nogil:
        # i,j, idx are indexes of the raw image uncorrected
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

    # Hack to prevent memory leak !!!
    cdef numpy.ndarray[numpy.float64_t, ndim = 2] tmp_ary = numpy.empty(shape=(shape0*shape1, size), dtype=numpy.float64)
    memcpy(&tmp_ary[0, 0], &lut[0, 0, 0], tmp_ary.nbytes)
    return numpy.core.records.array(tmp_ary.view(dtype=dtype_lut),
                                    shape=(shape0 * shape1, size), dtype=dtype_lut,
                                    copy=True)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def calc_CSR(float[:, :, :, :] pos not None, shape, bin_size, max_pixel_size,
             numpy.int8_t[:, :] mask=None):
    """
    @param pos: 4D position array
    @param shape: output shape
    @param bin_size: number of input element per output element (as numpy array)
    @param max_pixel_size: (2-tuple of int) size of a buffer covering the largest pixel
    @return: look-up table in CSR format: 3-tuple of array"""
    cdef int i, j, k, ms, ml, ns, nl, shape0, shape1, delta0, delta1, buffer_size, i0, i1, bins, lut_size, offset0, offset1, box_size0, box_size1
    shape0, shape1 = shape
    delta0, delta1 = max_pixel_size
    bins = shape0 * shape1
    cdef:
        numpy.int32_t idx = 0, tmp_index
        float A0, A1, B0, B1, C0, C1, D0, D1, pAB, pBC, pCD, pDA, cAB, cBC, cCD, cDA, area, value
        numpy.ndarray[numpy.int32_t, ndim = 1] indptr, indices
        numpy.ndarray[numpy.float32_t, ndim = 1] data
        int[:, :] outMax = view.array(shape=(shape0, shape1), itemsize=sizeof(int), format="i")
        float[:, :] buffer
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

    indices_size = lut_size * sizeof(numpy.int32)
    data_size = lut_size * sizeof(numpy.float32)
    indptr_size = bins * sizeof(numpy.int32)

    logger.info("CSR matrix: %.3f MByte" % ((indices_size + data_size + indptr_size) / 1.0e6))
    buffer = view.array(shape=(delta0, delta1), itemsize=sizeof(float), format="f")
    buffer_size = delta0 * delta1 * sizeof(float)
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
                        tmp_index = indptr[ml * shape1 + nl]
                        indices[tmp_index + k] = idx
                        data[tmp_index + k] = value
                        outMax[ml, nl] = k + 1
                idx += 1
    return (data, indices, indptr)


@cython.wraparound(False)
@cython.boundscheck(False)
def correct_LUT(image, shape, lut_point[:, :] LUT not None, dummy=None, delta_dummy=None):
    """
    Correct an image based on the look-up table calculated ...

    @param image: 2D-array with the image
    @param shape: shape of output image
    @param LUT: Look up table, here a 2D-array of struct
    @param dummy: value for invalid pixels
    @param delta_dummy: precision for invalid pixels
    @return: corrected 2D image
    """
    cdef:
        int i, j, lshape0, lshape1, idx, size, shape0, shape1
        float coef, sum, error, t, y, value, cdummy, cdelta_dummy
        float[:] lout, lin
        bint do_dummy = dummy is not None
    if do_dummy:
        cdummy = dummy
        if delta_dummy is None:
            cdelta_dummy = 0.0
    shape0, shape1 = shape
    lshape0 = LUT.shape[0]
    lshape1 = LUT.shape[1]
    img_shape = image.shape
    if (img_shape[0] < shape0) or (img_shape[1] < shape1):
        new_image = numpy.zeros((shape0, shape1), dtype=numpy.float32)
        new_image[:img_shape[0], :img_shape[1]] = image
        image = new_image
        logger.warning("Patching image as image is %ix%i and spline is %ix%i" % (img_shape[1], img_shape[0], shape1, shape0))

    out = numpy.zeros(shape, dtype=numpy.float32)
    lout = out.ravel()
    lin = numpy.ascontiguousarray(image.ravel(), dtype=numpy.float32)
    size = lin.size
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
    return out[:img_shape[0], :img_shape[1]]


@cython.wraparound(False)
@cython.boundscheck(False)
def correct_CSR(image, shape, LUT, dummy=None, delta_dummy=None):
    """
    Correct an image based on the look-up table calculated ...

    @param image: 2D-array with the image
    @param shape: shape of output image
    @param LUT: Look up table, here a 3-tuple array of ndarray
    @param dummy: value for invalid pixels
    @param delta_dummy: precision for invalid pixels
    @return: corrected 2D image
    """
    cdef:
        int i, j, idx, size, bins
        float coef, tmp, error, sum, y, t, value, cdummy, cdelta_dummy
        float[:] lout, lin, data
        numpy.int32_t[:] indices, indptr
        bint do_dummy = dummy is not None

    if do_dummy:
        cdummy = dummy
        if delta_dummy is None:
            cdelta_dummy = 0.0

    data, indices, indptr = LUT
    shape0, shape1 = shape
    bins = indptr.size - 1
    img_shape = image.shape
    if (img_shape[0] < shape0) or (img_shape[1] < shape1):
        new_image = numpy.zeros(shape, dtype=numpy.float32)
        new_image[:img_shape[0], :img_shape[1]] = image
        image = new_image
        logger.warning("Patching image as image is %ix%i and spline is %ix%i" % (img_shape[1], img_shape[0], shape1, shape0))

    out = numpy.zeros(shape, dtype=numpy.float32)
    lout = out.ravel()
    lin = numpy.ascontiguousarray(image.ravel(), dtype=numpy.float32)
    size = lin.size

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
    return out[:img_shape[0], :img_shape[1]]


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
