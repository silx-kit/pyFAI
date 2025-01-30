# -*- coding: utf-8 -*-
#cython: embedsignature=True, language_level=3, binding=True
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False,
## This is for developping:
##cython: profile=True, warn.undeclared=True, warn.unused=True, warn.unused_result=False, warn.unused_arg=True
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2020 European Synchrotron Radiation Facility, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
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

"""Common Look-Up table/CSR object creation tools and conversion"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "19/11/2024"
__status__ = "stable"
__license__ = "MIT"


from ..utils.decorators import deprecated_warning
include "regrid_common.pxi"
include "CSR_common.pxi"
include "LUT_common.pxi"


def LUT_to_CSR(lut):
    """Conversion between sparse matrix representations

    :param lut: Look-up table as 2D array of (int idx, float coef)
    :return: the same matrix as CSR representation
    :rtype: 3-tuple of numpy array (data, indices, indptr)
    """
    cdef:
        int nrow, ncol

    ncol = lut.shape[1]
    nrow = lut.shape[0]

    cdef:
        lut_t[:, ::1] lut_ = numpy.ascontiguousarray(lut, lut_d)
        data_t[::1] data = numpy.zeros(nrow * ncol, data_d)
        index_t[::1]  indices = numpy.zeros(nrow * ncol, index_d)
        index_t[::1] indptr = numpy.zeros(nrow + 1, index_d)
        int i, j, nelt
        lut_t point
    with nogil:
        nelt = 0
        for i in range(nrow):
            indptr[i] = nelt
            for j in range(ncol):
                point = lut_[i, j]
                if point.coef <= 0.0:
                    continue
                else:
                    data[nelt] = point.coef
                    indices[nelt] = point.idx
                    nelt += 1
        indptr[nrow] = nelt
    return numpy.asarray(data[:nelt]), numpy.asarray(indices[:nelt]), numpy.asarray(indptr)


def CSR_to_LUT(data, indices, indptr):
    """Conversion between sparse matrix representations

    :param data: coef of the sparse matrix as 1D array
    :param indices: index of the col position in input array as 1D array
    :param indptr: index of the start of the row in the indices array
    :return: the same matrix as LUT representation
    :rtype: record array of (int idx, float coef)
    """
    cdef:
        int nrow, ncol

    nrow = indptr.shape[0] - 1
    ncol = (indptr[1:] - indptr[:nrow]).max()
    assert nrow > 0, "nrow >0"
    assert ncol > 0, "ncol >0"

    cdef:
        data_t[::1] data_ = numpy.ascontiguousarray(data, dtype=data_d)
        index_t[::1]  indices_ = numpy.ascontiguousarray(indices, dtype=index_d)
        index_t[::1] indptr_ = numpy.ascontiguousarray(indptr, dtype=index_d)
        lut_t[:, ::1] lut = numpy.zeros((nrow, ncol), dtype=lut_d)
        lut_t point
        int i, j, nelt
        float coef
    with nogil:
        for i in range(nrow):
            nelt = 0
            for j in range(indptr_[i], indptr_[i + 1]):
                coef = data_[j]
                if coef <= 0.0:
                    continue
                point.coef = coef
                point.idx = indices_[j]
                lut[i, nelt] = point
                nelt += 1
    return numpy.asarray(lut)
