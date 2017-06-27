# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2017 European Synchrotron Radiation Facility, France
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
__date__ = "22/06/2017"
__status__ = "stable"
__license__ = "MIT"

include "sparse_common.pxi"


@cython.boundscheck(False)
def LUT_to_CSR(lut):
    """Conversion between sparse matrix representations

    :param lut: Look-up table as 2D array of (int idx, float coef)
    :return: the same matrix as CSR representation
    :rtype: 3-tuple of numpy array (data, indices, indptr)
    """
    cdef:
        int nrow, ncol

    ncol = lut.shape[-1]
    nrow = lut.shape[0]

    cdef:
        lut_point[:, ::1] lut_ = numpy.ascontiguousarray(lut, dtype_lut)
        float[::1] data = numpy.zeros(nrow * ncol, numpy.float32)
        int[::1]  indices = numpy.zeros(nrow * ncol, numpy.int32)
        int[::1] indptr = numpy.zeros(nrow + 1, numpy.int32)
        int i, j, nelt
        lut_point point
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


@cython.boundscheck(False)
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

    nrow = indptr.size - 1
    ncol = (indptr[1:] - indptr[:-1]).max()
    assert nrow > 0, "nrow >0"
    assert ncol > 0, "ncol >0"

    cdef:
        float[::1] data_ = numpy.ascontiguousarray(data, dtype=numpy.float32)
        int[::1]  indices_ = numpy.ascontiguousarray(indices, dtype=numpy.int32)
        int[::1] indptr_ = numpy.ascontiguousarray(indptr, dtype=numpy.int32)
        lut_point[:, ::1] lut = numpy.zeros((nrow, ncol), dtype=dtype_lut)
        lut_point point
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


cdef class Vector:
    """Variable size vector"""
# --> see the associated PXD file
#     cdef:
#         float[:] coef
#         int[:] idx
#         int size, allocated

    def __cinit__(self, int min_size=4):
        self.allocated = min_size
        self.coef = numpy.empty(self.allocated, dtype=numpy.float32)
        self.idx = numpy.empty(self.allocated, dtype=numpy.int32)
        self.size = 0

    def __dealloc__(self):
        self.coef = self.idx = None

    def __len__(self):
        return self.size

    def __repr__(self):
        return "Vector of size %i (%i elements allocated)" % (self.size, self.allocated)

    @property
    def nbytes(self):
        "Calculate the actual size of the object (in bytes)"
        return (self.allocated + 1) * 8

    def get_data(self):
        return numpy.asarray(self.idx[:self.size]), numpy.asarray(self.coef[:self.size])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef inline void _append(self, int idx, float coef):
        cdef:
            int pos, new_allocated 
            int[:] newidx
            float[:] newcoef
        pos = self.size
        self.size = pos + 1
        if pos >= self.allocated - 1:
            new_allocated = self.allocated * 2
            newcoef = numpy.empty(new_allocated, dtype=numpy.float32)
            newcoef[:pos] = self.coef[:pos]
            self.coef = newcoef
            newidx = numpy.empty(new_allocated, dtype=numpy.int32)
            newidx[:pos] = self.idx[:pos]
            self.idx = newidx
            self.allocated = new_allocated
            
        self.coef[pos] = coef
        self.idx[pos] = idx
        
    def append(self, idx, coef):
        "Python implementation of _append in cython"
        self._append(<int> idx, <float> coef)


cdef class ArrayBuilder:
# --> see the associated PXD file
#     cdef:
#         int size 
#         Vector[:] lines
        
    def __cinit__(self, int nlines, min_size=4):
        cdef int i
        self.size = nlines
        nullarray = numpy.array([None] * nlines)
        self.lines = nullarray
        for i in range(nlines):
            self.lines[i] = Vector(min_size=min_size)
            
    def __dealloc__(self):
        for i in range(self.size):
            self.lines[i] = None
        self.lines = None
        
    def __len__(self):
        return self.size

    def __repr__(self):
        cdef int i, max_line = 0
        for i in range(self.size):
            max_line = max(max_line, self.lines[i].size)
        return "ArrayBuilder of %i lines, the longest is %i" % (self.size, max_line)

    @property
    def nbytes(self):
        "Calculate the actual size of the object (in bytes)"
        cdef int i, sum = 0
        for i in range(self.size):
            sum += self.lines[i].nbytes
        return sum
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef inline void _append(self, int line, int col, float value):
        cdef: 
            Vector vector
        vector = self.lines[line]
        vector._append(col, value)
    
    def append(self, line, col, value):
        'Python wrapper for _append in cython'
        self._append(<int> line, <int> col, <float> value)

    def as_LUT(self):
        cdef:
            int i, max_size = 0
            int[:] local_idx
            float[:] local_coef
            lut_point[:, :] lut
            Vector vector
        for i in range(len(self.lines)):
            if len(self.lines[i]) > max_size:
                max_size = len(self.lines[i])
        lut = numpy.zeros((len(self.lines), max_size), dtype=dtype_lut)
        for i in range(len(self.lines)):
            vector = self.lines[i]
            local_idx, local_coef = vector.get_data()
            for j in range(len(vector)):
                lut[i, j] = lut_point(local_idx[j], local_coef[j])
        return numpy.asarray(lut, dtype=dtype_lut)

    def as_CSR(self):
        cdef:
            int i, val, start, end, total_size = 0 
            Vector vector
            lut_point[:, :] lut
            lut_point[:] data
            int[:] idptr, idx, local_idx
            float[:] coef, local_coef
        idptr = numpy.zeros(len(self.lines) + 1, dtype=numpy.int32)
        for i in range(len(self.lines)):
            total_size += len(self.lines[i])
            idptr[i + 1] = total_size
        coef = numpy.zeros(total_size, dtype=numpy.float32)
        idx = numpy.zeros(total_size, dtype=numpy.int32)
        for i in range(len(self.lines)):
            vector = self.lines[i]
            local_idx, local_coef = vector.get_data()
            start = idptr[i]
            end = start + len(vector)
            idx[start:end] = local_idx
            coef[start:end] = local_coef
        return numpy.asarray(idptr), numpy.asarray(idx), numpy.asarray(coef)
