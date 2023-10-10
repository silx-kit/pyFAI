# coding: utf-8
#cython: embedsignature=True, language_level=3
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2018 European Synchrotron Radiation Facility, Grenoble, France
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

from libcpp cimport bool
from libcpp.list cimport list as clist
from .shared_types cimport int32_t, float32_t

cdef packed struct pixel_t:
    int32_t index
    float32_t coef


cdef struct sparse_builder_private_t:
    void **_bins
    void *_compact_bins
    void *_heap


cdef class SparseBuilder(object):

    cdef sparse_builder_private_t _data

    cdef int _nbin
    cdef int _block_size
    cdef int *_sizes
    cdef bool _use_linked_list
    cdef bool _use_blocks
    cdef bool _use_heap_linked_list
    cdef bool _use_packed_list
    cdef object _mode

    cdef void *_create_bin(self) noexcept nogil
    cdef void _copy_bin_indexes_to(self, int bin_id, int32_t *dest) noexcept nogil
    cdef void _copy_bin_coefs_to(self, int bin_id, float32_t *dest) noexcept nogil
    cdef void _copy_bin_data_to(self, int bin_id, pixel_t *dest) noexcept nogil

    cdef int cget_bin_size(self, int bin_id) noexcept nogil
    cdef void cinsert(self, int bin_id, int index, float32_t coef) noexcept nogil
