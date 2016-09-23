# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2016 European Synchrotron Radiation Facility, Grenoble, France
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

"""Common Look-Up table/CSR object creaton tools """
__author__ = "Jerome Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "22/09/2016"
__status__ = "stable"
__license__ = "MIT"

import cython
import numpy

cdef struct lut_point:
    int idx
    float coef

dtype_lut = numpy.dtype([("idx", numpy.int32), ("coef", numpy.float32)])


# cdef class Vector2:
#     cdef:
#         float[:] coef
#         int[:] idx
#         int size, allocated
#     
#     def __cinit__(self, int min_size=10):
#         """This vector (resizable array) takes a minimum size as input
#         
#         This vector stores "together" an index (int) and a value (float) as 
#         a struct of array. t has been benchmarked faster than an array of struct.  
#         """
#         self.allocated = min_size
#         self.coef = np.empty(self.allocated, dtype=numpy.float32)
#         self.idx = np.empty(self.allocated, dtype=numpy.int32)
#         self.size = 0
#     
#     def __dealloc__(self):
#         self.coef = self.idx = None
#     
#     def __len__(self):
#         return self.size
#     
#     def get_data(self):
#         """Return the array of indices and the array of values"""
#         return np.asarray(self.idx[:self.size]), np.asarray(self.coef[:self.size])
#     
#     cpdef void append(self, int idx, float coef):
#         cdef:
#             int new_allocated 
#             int[:] newidx
#             float[:] newcoef
#         if self.size >= self.allocated - 1:
#                 new_allocated = self.allocated * 2
#                 newcoef = np.empty(new_allocated, dtype=numpy.float32)
#                 newcoef[:self.size] = self.coef[:self.size]
#                 self.coef = newcoef
#                 newidx = np.empty(new_allocated, dtype=numpy.int32)
#                 newidx[:self.size] = self.idx[:self.size]
#                 self.idx = newidx
#                 self.allocated = new_allocated
#         self.coef[self.size] = coef
#         self.idx[self.size] = idx
#         self.size = self.size + 1
# 
# 
# cdef class ArrayBuilder:
#     """Container object representing a sparse matrix for look-up table creation"""
#     cdef:
#         int size 
#         readonly list lines
#         
#     def __cinit__(self, int nlines, min_size=10):
#         """
#         :param nlines: number of lines of the sparse matrix. mandatrory, cannot be extended
#         :param min_size: minimul size of every line, each of them may be extended as needed.
#         """
#         cdef int i
#         self.lines = [Vector2(min_size=min_size) for i in range(nlines)]
#         self.size = nlines
#             
#     def __dealloc__(self):
#         while self.lines.__len__():
#             self.lines.pop()
#         self.lines = None
#         
#     def __len__(self):
#         return len(self.lines)
#     
#     cpdef void append(self, int line, int col, float value):
#         """append an element to a line of the array:
#         
#         :param line: line number 
#         :param col: theoretical column value (will be stored in a more eficient way)
#         :param value: the value to be stored at the given place 
#         """
#         cdef: 
#             Vector2 vector
#         vector = self.lines[line]
#         vector.append(col, value)
# 
#     def as_LUT(self):
#         """Export container in LUT format"""
#         cdef:
#             int max_size, i
#             int[:] local_idx
#             float[:] local_coef
#             lut_point[:, :] lut
#             Vector2 vector
#         max_size = 0
#         for i in range(len(self.lines)):
#             if len(self.lines[i]) > max_size:
#                 max_size = len(self.lines[i])
#         lut = np.zeros((len(self.lines), max_size), dtype=dtype_lut)
#         for i in range(len(self.lines)):
#             vector = self.lines[i]
#             local_idx, local_coef = vector.get_data()
#             for j in range(len(vector)):
#                 lut[i, j] = lut_point(local_idx[j], local_coef[j])
#         return np.asarray(lut, dtype=dtype_lut)
# 
#     def as_CSR(self):
#         """Export container in CSR format"""
#         cdef:
#             int total_size, i, val, start, end 
#             Vector2 vector
#             lut_point[:, :] lut
#             lut_point[:] data
#             int[:] idptr, idx, local_idx
#             float[:] coef, local_coef
#         total_size = 0
#         idptr = np.zeros(len(self.lines) + 1, dtype=np.int32)
#         for i in range(len(self.lines)):
#             total_size += len(self.lines[i])
#             idptr[i + 1] = total_size
#         coef = np.zeros(total_size, dtype=np.float32)
#         idx = np.zeros(total_size, dtype=np.int32)
#         for i in range(len(self.lines)):
#             vector = self.lines[i]
#             local_idx, local_coef = vector.get_data()
#             start = idptr[i]
#             end = start + len(vector)
#             idx[start:end] = local_idx
#             coef[start:end] = local_coef
#         return np.asarray(idptr), np.asarray(idx), np.asarray(coef)
