# -*- coding: utf-8 -*-
#cython: embedsignature=True, language_level=3, binding=True
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False,
## This is for developping
## cython: profile=True, warn.undeclared=True, warn.unused=True, warn.unused_result=False, warn.unused_arg=True
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2014-2021 European Synchrotron Radiation Facility,  France
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

"""Cython module to reconstruct the masked values of an image.

It's a simple inpainting module for reconstructing the missing part of an
image (masked) to be able to use more common algorithms.
"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "07/01/2025"
__status__ = "stable"
__license__ = "MIT"


import cython
import numpy
from libc.math cimport sqrtf, fabs, NAN
from cython.parallel import prange
from libc.stdint cimport int8_t, uint8_t, int16_t, uint16_t, \
                         int32_t, uint32_t, int64_t, uint64_t


cdef inline float invert_distance(Py_ssize_t i0, Py_ssize_t i1, Py_ssize_t p0, Py_ssize_t p1) noexcept nogil:
    """Neither d0, nor d1 can be null !"""
    cdef Py_ssize_t d0, d1
    d0 = (i0 - p0)
    d1 = (i1 - p1)
    return (<float> 1.0) / sqrtf(<float> (d0*d0 + d1*d1))

cdef inline float processPoint(float[:, ::1] data,
                               int8_t[:, ::1] mask,
                               size_t p0,
                               size_t p1,
                               size_t d0,
                               size_t d1) noexcept nogil:
    cdef:
        size_t dist = 0, i = 0
        float sum = 0.0, count = 0.0, invdst = 0.0
        bint found = 0
        size_t start0 = p0, stop0 = p0, start1 = p1, stop1 = p1
    while not found:
        dist += 1
        if start0 > 0:
            start0 = p0 - dist
        else:
            start0 = 0
        if stop0 < d0 - 1:
            stop0 = p0 + dist
        else:
            stop0 = d0 - 1
        if start1 > 0:
            start1 = p1 - dist
        else:
            start1 = 0
        if stop1 < d1 - 1:
            stop1 = p1 + dist
        else:
            stop1 = d1 - 1
        for i in range(start0, stop0 + 1):
            if mask[i, start1] == 0:
                invdst = invert_distance(i, start1, p0, p1)
                count += invdst
                sum += invdst * data[i, start1]
            if mask[i, stop1] == 0:
                invdst = invert_distance(i, stop1, p0, p1)
                count += invdst
                sum += invdst * data[i, stop1]
        for i in range(start1 + 1, stop1):
            if mask[start0, i] == 0:
                invdst = invert_distance(start0, i, p0, p1)
                count += invdst
                sum += invdst * data[start0, i]
            if mask[stop0, i] == 0:
                invdst = invert_distance(stop0, i, p0, p1)
                count += invdst
                sum += invdst * data[stop0, i]
        if count > 0:
            found = 1
    return sum / count


def reconstruct(data,
                mask=None,
                dummy=None,
                delta_dummy=None):
    """
    reconstruct missing part of an image (tries to be continuous)

    :param data: the input image
    :param mask: where data should be reconstructed.
    :param dummy: value of the dummy (masked out) data
    :param delta_dummy: precision for dummy values

    :return: reconstructed image.
    """
    assert data.ndim == 2, "data.ndim == 2"
    cdef:
        ssize_t d0 = data.shape[0]
        ssize_t d1 = data.shape[1]
        ssize_t p0, p1
        float[:, ::1] cdata
        int8_t[:, ::1] cmask
        bint is_masked, do_dummy
        float cdummy=0.0, cddummy=0.0, value
        float[:, ::1] out = numpy.zeros_like(data)

    cdata = numpy.ascontiguousarray(data, dtype=numpy.float32)
    if mask is not None:
        cmask = numpy.ascontiguousarray(mask, dtype=numpy.int8)
    else:
        cmask = numpy.zeros((d0, d1), dtype=numpy.int8)
    assert d0 == mask.shape[0], "mask.shape[0]"
    assert d1 == mask.shape[1], "mask.shape[1]"

    if dummy is not None:
        do_dummy = True
        cdummy = <float> dummy
        if delta_dummy is None:
            cddummy = 0.0
        else:
            cddummy = <float> delta_dummy

    #  Nota: this has to go in 2 passes, one to mark, one to reconstruct
    for p0 in prange(d0, nogil=True, schedule="guided"):
        for p1 in range(d1):
            is_masked = cmask[p0, p1]
            if not is_masked and do_dummy:
                value = cdata[p0, p1]
                if cddummy == 0.0:
                    cmask[p0, p1] += (value == cdummy)
                elif (fabs(value - cdummy) <= cddummy):
                    cmask[p0, p1] += 1
    # Reconstruction phase
    for p0 in prange(d0, nogil=True, schedule="guided"):
        for p1 in range(d1):
            if cmask[p0, p1]:
                out[p0, p1] += processPoint(cdata, cmask, p0, p1, d0, d1)
            else:
                out[p0, p1] += cdata[p0, p1]
    return numpy.asarray(out)
