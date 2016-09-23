# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) 2015-2016 European Synchrotron Radiation Facility, France
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
__date__ = "23/09/2016"
__status__ = "stable"
__license__ = "MIT"

import cython
import numpy

dtype_lut = numpy.dtype([("idx", numpy.int32), ("coef", numpy.float32)])


cdef class Vector2:
#     cdef:
#         float[:] coef
#         int[:] idx
#         int size, allocated
    
    def __cinit__(self, int min_size=10):
        """This vector (resizable array) takes a minimum size as input
        
        This vector stores "together" an index (int) and a value (float) as 
        a struct of array. t has been benchmarked faster than an array of struct.  
        """
        self.allocated = min_size
        self.coef = numpy.empty(self.allocated, dtype=numpy.float32)
        self.idx = numpy.empty(self.allocated, dtype=numpy.int32)
        self.size = 0
    
    def __dealloc__(self):
        self.coef = self.idx = None
    
    def __len__(self):
        return self.size
    
    def get_data(self):
        """Return the array of indices and the array of values"""
        return numpy.asarray(self.idx[:self.size]), numpy.asarray(self.coef[:self.size])
    
    #cdef void append(self, int idx, float coef):
#     cdef append(self, int idx, float coef):
#         cdef:
#             int new_allocated 
#             int[:] newidx
#             float[:] newcoef
#         if self.size >= self.allocated - 1:
#                 new_allocated = self.allocated * 2
#                 newcoef = numpy.empty(new_allocated, dtype=numpy.float32)
#                 newcoef[:self.size] = self.coef[:self.size]
#                 self.coef = newcoef
#                 newidx = numpy.empty(new_allocated, dtype=numpy.int32)
#                 newidx[:self.size] = self.idx[:self.size]
#                 self.idx = newidx
#                 self.allocated = new_allocated
#         self.coef[self.size] = coef
#         self.idx[self.size] = idx
#         self.size = self.size + 1


cdef class ArrayBuilder:
    """Container object representing a sparse matrix for look-up table creation"""
#     cdef:
#         int size 
#         readonly list lines
        
    def __cinit__(self, int nlines, min_size=10):
        """
        :param nlines: number of lines of the sparse matrix. mandatrory, cannot be extended
        :param min_size: minimul size of every line, each of them may be extended as needed.
        """
        cdef int i
        self.lines = [Vector2(min_size=min_size) for i in range(nlines)]
        self.size = nlines
            
    def __dealloc__(self):
        while self.lines.__len__():
            self.lines.pop()
        self.lines = None
        
    def __len__(self):
        return len(self.lines)
    
    #cdef void append(self, int line, int col, float value):
#     cdef append(self, int line, int col, float value):
#         """append an element to a line of the array:
#         
#         :param line: line number 
#         :param col: theoretical column value (will be stored in a more efficient way)
#         :param value: the value to be stored at the given place 
#         """
#         cdef: 
#             Vector2 vector
#         vector = self.lines[line]
#         vector.append(col, value)

    def as_LUT(self):
        """Export container in LUT format"""
        cdef:
            int max_size, i
            int[:] local_idx
            float[:] local_coef
            lut_point[:, :] lut
            Vector2 vector
        max_size = 0
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
        """Export container in CSR format"""
        cdef:
            int total_size, i, val, start, end 
            Vector2 vector
            lut_point[:, :] lut
            lut_point[:] data
            int[:] idptr, idx, local_idx
            float[:] coef, local_coef
        total_size = 0
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


@cython.boundscheck(False)
def LUT_to_CSR(lut):
    """Conversion between sparse matrix representations

    @param lut: Look-up table as 2D array of (int idx, float coef)
    @return: the same matrix as CSR representation
    @rtype: 3-tuple of numpy array (data, indices, indptr)
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

    @param data: coef of the sparse matrix as 1D array
    @param indices: index of the col position in input array as 1D array
    @param indptr: index of the start of the row in the indices array
    @return: the same matrix as LUT representation
    @rtype: record array of (int idx, float coef)
    """
    cdef:
        int nrow, ncol

    nrow = indptr.size - 1
    ncol = (indptr[1:] - indptr[:-1]).max()
    assert nrow > 0
    assert ncol > 0

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
