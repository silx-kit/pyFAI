#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2018 European Synchrotron Radiation Facility, Grenoble, France
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

__author__ = "Valentin Valls"
__license__ = "MIT"
__date__ = "16/07/2018"
__copyright__ = "2018, ESRF"

import numpy
cimport numpy as cnumpy

from libcpp.vector cimport vector
from libcpp.list cimport list as clist
from libcpp cimport bool
from libc.math cimport fabs
cimport libc.stdlib
cimport libc.string

from cython.operator cimport dereference
from cython.operator cimport preincrement
cimport cython

from .sparse_utils import lut_d


cdef packed struct pixel_t:
    cnumpy.int32_t index
    cnumpy.float32_t coef


cdef struct chained_pixel_t:
    pixel_t data
    chained_pixel_t *next


cdef struct compact_bin_t:
    int size
    chained_pixel_t *front_ptr
    chained_pixel_t *back_ptr


cdef packed struct packed_data_t:
    int bin_id
    pixel_t data


cdef cppclass Heap:
    clist[cnumpy.int32_t *] _indexes
    clist[cnumpy.float32_t *] _coefs
    clist[chained_pixel_t *] _pixels
    clist[packed_data_t *] _packed_data

    cnumpy.int32_t *_current_index_block
    cnumpy.float32_t *_current_coef_block
    chained_pixel_t *_current_pixel_block
    packed_data_t *_current_packed_block

    int _index_pos
    int _coef_pos
    int _pixel_pos
    int _packed_pos
    int _block_size

    Heap(int block_size) nogil:
        this._block_size = block_size
        this._index_pos = 0
        this._coef_pos = 0
        this._packed_pos = 0
        this._current_index_block = NULL
        this._current_coef_block = NULL
        this._current_pixel_block = NULL
        this._current_packed_block = NULL

    __dealloc__() nogil:
        cdef:
            clist[cnumpy.int32_t *].iterator it_indexes
            clist[cnumpy.float32_t *].iterator it_coefs
            clist[chained_pixel_t *].iterator it_pixels
            clist[packed_data_t *].iterator it_packed
            cnumpy.int32_t *indexes
            cnumpy.float32_t *coefs
            chained_pixel_t *pixels
            packed_data_t *packed

        it_indexes = this._indexes.begin()
        while it_indexes != this._indexes.end():
            indexes = dereference(it_indexes)
            libc.stdlib.free(indexes)
            preincrement(it_indexes)

        it_coefs = this._coefs.begin()
        while it_coefs != this._coefs.end():
            coefs = dereference(it_coefs)
            libc.stdlib.free(coefs)
            preincrement(it_coefs)

        it_pixels = this._pixels.begin()
        while it_pixels != this._pixels.end():
            pixels = dereference(it_pixels)
            libc.stdlib.free(pixels)
            preincrement(it_pixels)

        it_packed = this._packed_data.begin()
        while it_packed != this._packed_data.end():
            packed = dereference(it_packed)
            libc.stdlib.free(packed)
            preincrement(it_packed)

    cnumpy.int32_t * alloc_indexes(int size) nogil:
        cdef:
            cnumpy.int32_t *data
        if this._current_index_block == NULL or this._index_pos + size > this._block_size:
            data = <cnumpy.int32_t *>libc.stdlib.malloc(this._block_size * sizeof(cnumpy.int32_t))
            this._current_index_block = data
            this._indexes.push_back(data)
            this._index_pos = 0
        data = this._current_index_block + this._index_pos
        this._index_pos += size
        return data

    cnumpy.float32_t * alloc_coefs(int size) nogil:
        cdef:
            cnumpy.float32_t *data
        if this._current_coef_block == NULL or this._coef_pos + size > this._block_size:
            data = <cnumpy.float32_t *>libc.stdlib.malloc(this._block_size * sizeof(cnumpy.float32_t))
            this._current_coef_block = data
            this._coefs.push_back(data)
            this._coef_pos = 0
        data = this._current_coef_block + this._coef_pos
        this._coef_pos += size
        return data

    chained_pixel_t* alloc_pixel() nogil:
        cdef:
            chained_pixel_t *data
            int foo
        if this._current_pixel_block == NULL or this._pixel_pos + 1 > this._block_size:
            data = <chained_pixel_t *>libc.stdlib.malloc(this._block_size * sizeof(chained_pixel_t))
            this._current_pixel_block = data
            this._pixels.push_back(data)
            this._pixel_pos = 0
        data = this._current_pixel_block + this._pixel_pos
        this._pixel_pos += 1
        return data

    packed_data_t* alloc_packed_data() nogil:
        cdef:
            packed_data_t *data
            int foo
        if this._current_packed_block == NULL or this._packed_pos + 1 > this._block_size:
            data = <packed_data_t *>libc.stdlib.malloc(this._block_size * sizeof(packed_data_t))
            this._current_packed_block = data
            this._packed_data.push_back(data)
            this._packed_pos = 0
        data = this._current_packed_block + this._packed_pos
        this._packed_pos += 1
        return data


cdef cppclass PixelElementaryBlock:
    cnumpy.int32_t *_indexes
    cnumpy.float32_t *_coefs
    int _size
    int _max_size
    bool _allocated

    PixelElementaryBlock(int size, Heap *heap) nogil:
        if heap == NULL:
            this._indexes = <cnumpy.int32_t *>libc.stdlib.malloc(size * sizeof(cnumpy.int32_t))
            this._coefs = <cnumpy.float32_t *>libc.stdlib.malloc(size * sizeof(cnumpy.float32_t))
            this._allocated = True
        else:
            this._indexes = heap.alloc_indexes(size)
            this._coefs = heap.alloc_coefs(size)
            this._allocated = False
        this._size = 0
        this._max_size = size

    __dealloc__() nogil:
        if this._allocated:
            libc.stdlib.free(this._indexes)
            libc.stdlib.free(this._coefs)

    void push(pixel_t &pixel) nogil:
        this._indexes[this._size] = pixel.index
        this._coefs[this._size] = pixel.coef
        this._size += 1

    int size() nogil:
        return this._size

    bool is_full() nogil:
        return this._size >= this._max_size

    bool has_space(int size) nogil:
        return this._size + size <= this._max_size


cdef cppclass PixelBlock:
    clist[PixelElementaryBlock*] _blocks
    int _block_size
    Heap *_heap
    PixelElementaryBlock* _current_block

    PixelBlock(int block_size, Heap *heap) nogil:
        this._block_size = block_size
        this._heap = heap
        this._current_block = NULL

    __dealloc__() nogil:
        cdef:
            PixelElementaryBlock* element
            int i = 0
            clist[PixelElementaryBlock*].iterator it
        it = this._blocks.begin()
        while it != this._blocks.end():
            element = dereference(it)
            del element
            preincrement(it)
        this._blocks.clear()

    void push(pixel_t &pixel) nogil:
        cdef:
            PixelElementaryBlock *block
        if this._current_block == NULL or this._current_block.is_full():
            block = new PixelElementaryBlock(this._block_size, this._heap)
            this._blocks.push_back(block)
            this._current_block = block
        block = this._current_block
        block.push(pixel)

    int size() nogil:
        cdef:
            int size = 0
            clist[PixelElementaryBlock*].iterator it
        it = this._blocks.begin()
        while it != this._blocks.end():
            size += dereference(it).size()
            preincrement(it)
        return size

    void copy_indexes_to(cnumpy.int32_t *dest) nogil:
        cdef:
            clist[PixelElementaryBlock*].iterator it
            PixelElementaryBlock* block
        it = this._blocks.begin()
        while it != this._blocks.end():
            block = dereference(it)
            if block.size() != 0:
                libc.string.memcpy(dest, block._indexes, block.size() * sizeof(cnumpy.int32_t))
                dest += block.size()
            preincrement(it)

    void copy_coefs_to(cnumpy.float32_t *dest) nogil:
        cdef:
            clist[PixelElementaryBlock*].iterator it
            PixelElementaryBlock* block
        it = this._blocks.begin()
        while it != this._blocks.end():
            block = dereference(it)
            if block.size() != 0:
                libc.string.memcpy(dest, block._coefs, block.size() * sizeof(cnumpy.float32_t))
                dest += block.size()
            preincrement(it)

    void copy_data_to(pixel_t *dest) nogil:
        cdef:
            clist[PixelElementaryBlock*].iterator it
            PixelElementaryBlock* block
            int i
        it = this._blocks.begin()
        while it != this._blocks.end():
            block = dereference(it)
            for i in range(block.size()):
                dest.index = block._indexes[i]
                dest.coef = block._coefs[i]
                dest += 1
            preincrement(it)


cdef cppclass PixelBin:
    clist[pixel_t] _pixels
    PixelBlock *_pixels_in_block

    PixelBin(int block_size, Heap *heap) nogil:
        if block_size > 0:
            this._pixels_in_block = new PixelBlock(block_size, heap)
        else:
            this._pixels_in_block = NULL

    __dealloc__() nogil:
        if this._pixels_in_block != NULL:
            del this._pixels_in_block
            this._pixels_in_block = NULL
        else:
            this._pixels.clear()

    void push(pixel_t &pixel) nogil:
        if this._pixels_in_block != NULL:
            this._pixels_in_block.push(pixel)
        else:
            this._pixels.push_back(pixel)

    int size() nogil:
        if this._pixels_in_block != NULL:
            return this._pixels_in_block.size()
        else:
            return this._pixels.size()

    void copy_indexes_to(cnumpy.int32_t *dest) nogil:
        cdef:
            clist[pixel_t].iterator it_points

        if this._pixels_in_block != NULL:
            this._pixels_in_block.copy_indexes_to(dest)

        it_points = this._pixels.begin()
        while it_points != this._pixels.end():
            dest[0] = dereference(it_points).index
            preincrement(it_points)
            dest += 1

    void copy_coefs_to(cnumpy.float32_t *dest) nogil:
        cdef:
            clist[pixel_t].iterator it_points

        if this._pixels_in_block != NULL:
            this._pixels_in_block.copy_coefs_to(dest)

        it_points = this._pixels.begin()
        while it_points != this._pixels.end():
            dest[0] = dereference(it_points).coef
            preincrement(it_points)
            dest += 1

    void copy_data_to(pixel_t *dest) nogil:
        cdef:
            clist[pixel_t].iterator it_points

        if this._pixels_in_block != NULL:
            this._pixels_in_block.copy_data_to(dest)

        it_points = this._pixels.begin()
        while it_points != this._pixels.end():
            dest[0] = dereference(it_points)
            preincrement(it_points)
            dest += 1


cdef struct sparse_builder_internal_t:
    PixelBin **_bins
    compact_bin_t *_compact_bins
    Heap *_heap


cdef inline sparse_builder_internal_t *get_internal_data(sparse_builder_private_t* data) nogil:
    return <sparse_builder_internal_t*> data


cdef class SparseBuilder(object):
    """
    This class provade an API to build a sparse matrix from bin data

    It provides different internal structure to be able to use it in different
    context. It can boost a fast insert, or speed up fast convertion to CSR
    format.

    :param: int nbin: Number of bin to store
    :param str mode: Internal structure used to store the data:

        - "pack": Alloc a `heap_size` and feed it with tuple (bin, indice, value).
            The insert is very fast, conversion to CSR is done using sequencial
            read and a random write.
        - "heaplist": Alloc a `heap_size` and feed it with a linked list per bins
            containing (indice, value, next).
            The insert is very fast, conversion to CSR is done using random read
            and a sequencial write.
        - "block": Alloc `block_size` per bins and feed it with values and indices.
            The conversion to CSR is done sequencially using block copy.
            The `heap_size` should be a multiple of the `block_size`. If the
            `heap_size` is zero, block are allocated one by one without management.
        - "stdlist": Use standard C++ list. It is head as reference for testing.
    :param Union[None|int] block_size: Number of element in a block if used. If more
        space is needed another block are allocated on the fly.
    :param Union[None|int] heap_size: Number of element in the global memory
        managment. This system allocation a single time memory for many needs.
        It reduce the overhead of memory allocation. If set to `None` or `0`,
        this management is disabled.
    """

    def __init__(self, nbin, mode="block", block_size=512, heap_size=0):

        modes = ["pack", "heaplist", "block", "stdlist"]
        if mode not in modes:
            raise ValueError("Mode %s unsupported. Supported modes are: %s" % (mode, ", ".join(modes)))

        self._use_linked_list = False
        self._use_blocks = False
        self._use_heap_linked_list = False
        self._use_packed_list = False

        if mode == "block":
            self._use_blocks = True
            if heap_size != 0:
                if heap_size < block_size:
                    raise ValueError("Heap size is supposed to be bigger than block size")
        elif mode == "heaplist":
            self._use_heap_linked_list = True
            if heap_size in [0, None]:
                raise ValueError("A heap size is expected for this mode")
        elif mode == "stdlist":
            self._use_linked_list = True
            block_size = 0
            heap_size = 0
        elif mode == "pack":
            self._use_packed_list = True
            if heap_size in [0, None]:
                raise ValueError("A heap size is expected for this mode")
        else:
            assert(False)

        self._block_size = block_size
        self._nbin = nbin
        if heap_size not in [None, 0]:
            get_internal_data(&self._data)._heap = new Heap(heap_size)
        else:
            get_internal_data(&self._data)._heap = NULL

        if self._use_blocks or self._use_linked_list:
            get_internal_data(&self._data)._bins = <PixelBin **>libc.stdlib.malloc(self._nbin * sizeof(PixelBin *))
            libc.string.memset(get_internal_data(&self._data)._bins, 0, self._nbin * sizeof(PixelBin *))
        elif self._use_heap_linked_list:
            get_internal_data(&self._data)._compact_bins = <compact_bin_t *>libc.stdlib.malloc(self._nbin * sizeof(compact_bin_t))
            libc.string.memset(get_internal_data(&self._data)._compact_bins, 0, self._nbin * sizeof(compact_bin_t))
        elif self._use_packed_list:
            self._sizes = <int *>libc.stdlib.malloc(self._nbin * sizeof(int))
            libc.string.memset(self._sizes, 0, self._nbin * sizeof(int))

        self._mode = mode

    def __dealloc__(self):
        """Release memory."""
        cdef:
            PixelBin *pixel_bin
            clist[PixelElementaryBlock*].iterator it_points
            PixelElementaryBlock* heap
            int i

        if self._use_blocks:
            for i in range(self._nbin):
                pixel_bin = get_internal_data(&self._data)._bins[i]
                if pixel_bin != NULL:
                    del pixel_bin
            libc.stdlib.free(get_internal_data(&self._data)._bins)
        elif self._use_heap_linked_list:
            libc.stdlib.free(get_internal_data(&self._data)._compact_bins)
        elif self._use_packed_list:
            libc.stdlib.free(self._sizes)

        if get_internal_data(&self._data)._heap != NULL:
            del get_internal_data(&self._data)._heap

    def mode(self):
        """Returns the storage mode used by the builder.

        :rtype: str
        """
        return self._mode

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void *_create_bin(self) nogil:
        """Create a bin object used to statore data for some formats.

        :rtype: PixelBin
        """
        return new PixelBin(self._block_size, get_internal_data(&self._data)._heap)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void cinsert(self, int bin_id, int index, cnumpy.float32_t coef) nogil:
        """Insert an indice and a value in a specific bin.

        This function to not have any overhead like `insert`. There is no check
        on arguments nor managing of Python exceptions.

        :param int bin_id: Index of the bin
        :param int index: Indice of the data to store
        :param int coef: Value of the data to store
        """
        cdef:
            pixel_t pixel
            PixelBin *pixel_bin
            chained_pixel_t* chained_pixel
            packed_data_t* packed_data
        if bin_id < 0 or bin_id >= self._nbin:
            return
        pixel.index = index
        pixel.coef = coef

        if self._use_heap_linked_list:
            chained_pixel = get_internal_data(&self._data)._heap.alloc_pixel()
            chained_pixel.data = pixel
            if get_internal_data(&self._data)._compact_bins[bin_id].front_ptr == NULL:
                get_internal_data(&self._data)._compact_bins[bin_id].front_ptr = chained_pixel
            else:
                get_internal_data(&self._data)._compact_bins[bin_id].back_ptr.next = chained_pixel
            get_internal_data(&self._data)._compact_bins[bin_id].back_ptr = chained_pixel
            get_internal_data(&self._data)._compact_bins[bin_id].size += 1
        elif self._use_packed_list:
            packed_data = get_internal_data(&self._data)._heap.alloc_packed_data()
            packed_data.bin_id = bin_id
            packed_data.data = pixel
            self._sizes[bin_id] += 1
        else:
            pixel_bin = get_internal_data(&self._data)._bins[bin_id]
            if pixel_bin == NULL:
                pixel_bin = <PixelBin*> self._create_bin()
                get_internal_data(&self._data)._bins[bin_id] = pixel_bin
            get_internal_data(&self._data)._bins[bin_id].push(pixel)

    def insert(self, bin_id, index, coef):
        """Insert an indice and a value in a specific bin.

        :param int bin_id: Index of the bin
        :param int index: Indice of the data to store
        :param int coef: Value of the data to store
        """
        if bin_id < 0 or bin_id >= self._nbin:
            raise ValueError("bin_id out of range")
        self.cinsert(bin_id, index, coef)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def get_bin_coefs(self, bin_id):
        """Returns the values stored in a specific bin.

        :param int bin_id: Index of the bin
        :rtype: numpy.array
        """
        cdef:
            int size
            cnumpy.float32_t[:] coefs
            cnumpy.float32_t *coefs_ptr

        if bin_id < 0 or bin_id >= self._nbin:
            raise ValueError("bin_id out of range")

        if self._use_packed_list:
            raise NotImplementedError("Not implemented for this mode (not efficient)")

        size = self.cget_bin_size(bin_id)
        coefs = numpy.empty(size, dtype=numpy.float32)
        coefs_ptr = &coefs[0]
        self._copy_bin_coefs_to(bin_id, coefs_ptr)
        return numpy.asarray(coefs)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def get_bin_indexes(self, bin_id):
        """Returns the indices stored in a specific bin.

        :param int bin_id: Index of the bin
        :rtype: numpy.array
        """
        cdef:
            int size
            cnumpy.int32_t[:] indexes
            cnumpy.int32_t *indexes_ptr

        if bin_id < 0 or bin_id >= self._nbin:
            raise ValueError("bin_id out of range")

        if self._use_packed_list:
            raise NotImplementedError("Not implemented for this mode (not efficient)")

        size = self.cget_bin_size(bin_id)
        indexes = numpy.empty(size, dtype=numpy.int32)
        indexes_ptr = &indexes[0]
        self._copy_bin_indexes_to(bin_id, indexes_ptr)
        return numpy.asarray(indexes)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int cget_bin_size(self, int bin_id) nogil:
        """Returns the size of a specific bin.

        :param int bin_id: Index of the bin
        :rtype: int
        """
        cdef:
            PixelBin *pixel_bin
        if self._use_heap_linked_list:
            return get_internal_data(&self._data)._compact_bins[bin_id].size
        elif self._use_packed_list:
            return self._sizes[bin_id]
        pixel_bin = get_internal_data(&self._data)._bins[bin_id]
        if pixel_bin == NULL:
            return 0
        return pixel_bin.size()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def get_bin_size(self, bin_id):
        """Returns the size of a specific bin.

        :param int bin_id: Number of the bin requested
        :rtype: int
        """
        if bin_id < 0 or bin_id >= self._nbin:
            raise ValueError("bin_id out of range")
        return self.cget_bin_size(bin_id)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def get_bin_sizes(self):
        """Returns the size of all the bins.

        :rtype: numpy.ndarray(dtype=int)
        """
        cdef:
            PixelBin *pixel_bin
            int bin_id
            cnumpy.int32_t[:] sizes

        sizes = numpy.empty(self._nbin, dtype=numpy.int32)

        if self._use_heap_linked_list:
            for bin_id in range(self._nbin):
                sizes[bin_id] = get_internal_data(&self._data)._compact_bins[bin_id].size
        elif self._use_packed_list:
            # FIXME: Can be done with a memcopy
            for bin_id in range(self._nbin):
                sizes[bin_id] = self._sizes[bin_id]
        else:
            for bin_id in range(self._nbin):
                pixel_bin = get_internal_data(&self._data)._bins[bin_id]
                if pixel_bin != NULL:
                    sizes[bin_id] = pixel_bin.size()
                else:
                    sizes[bin_id] = 0
        return numpy.asarray(sizes)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def size(self):
        """Returns the number of elements contained in the structure.

        :rtype: int
        """
        cdef:
            PixelBin *pixel_bin
            int size
            int bin_id

        size = 0

        if self._use_heap_linked_list:
            for bin_id in range(self._nbin):
                size += get_internal_data(&self._data)._compact_bins[bin_id].size
        elif self._use_packed_list:
            for bin_id in range(self._nbin):
                size += self._sizes[bin_id]
        else:
            for bin_id in range(self._nbin):
                pixel_bin = get_internal_data(&self._data)._bins[bin_id]
                if pixel_bin != NULL:
                    size += pixel_bin.size()
        return size

    cdef void _copy_bin_indexes_to(self, int bin_id, cnumpy.int32_t *dest) nogil:
        cdef:
            PixelBin *pixel_bin
            compact_bin_t *compact_bin
            chained_pixel_t *chained_pixel
        if self._use_heap_linked_list:
            compact_bin = &get_internal_data(&self._data)._compact_bins[bin_id]
            chained_pixel = compact_bin.front_ptr
            while chained_pixel != NULL:
                dest[0] = chained_pixel.data.index
                dest += 1
                if chained_pixel == compact_bin.back_ptr:
                    # The next ptr of the last element is not initialized
                    break
                chained_pixel = chained_pixel.next
        elif self._use_packed_list:
            # unsupported
            return
        else:
            pixel_bin = get_internal_data(&self._data)._bins[bin_id]
            if pixel_bin != NULL:
                pixel_bin.copy_indexes_to(dest)

    cdef void _copy_bin_coefs_to(self, int bin_id, cnumpy.float32_t *dest) nogil:
        cdef:
            PixelBin *pixel_bin
            compact_bin_t *compact_bin
            chained_pixel_t *chained_pixel
        if self._use_heap_linked_list:
            compact_bin = &get_internal_data(&self._data)._compact_bins[bin_id]
            chained_pixel = compact_bin.front_ptr
            while chained_pixel != NULL:
                dest[0] = chained_pixel.data.coef
                dest += 1
                if chained_pixel == compact_bin.back_ptr:
                    # The next ptr of the last element is not initialized
                    break
                chained_pixel = chained_pixel.next
        elif self._use_packed_list:
            # unsupported
            return
        else:
            pixel_bin = get_internal_data(&self._data)._bins[bin_id]
            if pixel_bin != NULL:
                pixel_bin.copy_coefs_to(dest)

    cdef void _copy_bin_data_to(self, int bin_id, pixel_t *dest) nogil:
        cdef:
            PixelBin *pixel_bin
            compact_bin_t *compact_bin
            chained_pixel_t *chained_pixel
        if self._use_heap_linked_list:
            compact_bin = &get_internal_data(&self._data)._compact_bins[bin_id]
            chained_pixel = compact_bin.front_ptr
            while chained_pixel != NULL:
                dest[0] = chained_pixel.data
                dest += 1
                if chained_pixel == compact_bin.back_ptr:
                    # The next ptr of the last element is not initialized
                    break
                chained_pixel = chained_pixel.next
        elif self._use_packed_list:
            # unsupported
            return
        else:
            pixel_bin = get_internal_data(&self._data)._bins[bin_id]
            if pixel_bin != NULL:
                pixel_bin.copy_data_to(dest)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def _to_csr_from_packed(self):
        cdef:
            cnumpy.int32_t[:] nbins
            cnumpy.int32_t[:] current_bin_pos
            cnumpy.int32_t[:] indexes
            cnumpy.int32_t *indexes_ptr
            cnumpy.float32_t[:] coefs
            cnumpy.float32_t *coefs_ptr
            int size
            int begin, end
            int bin_id
            int bin_size
            int pos
            packed_data_t *packed_block
            packed_data_t *packed_data

        # indexes of the first and the last+1 elements of each bins
        size = 0
        nbins = numpy.empty(self._nbin + 1, dtype=numpy.int32)
        current_bin_pos = numpy.empty(self._nbin + 1, dtype=numpy.int32)
        nbins[0] = size
        current_bin_pos[0] = size
        for bin_id in range(self._nbin):
            bin_size = self._sizes[bin_id]
            size += bin_size
            nbins[bin_id + 1] = size
            current_bin_pos[bin_id + 1] = size

        indexes = numpy.empty(size, dtype=numpy.int32)
        coefs = numpy.empty(size, dtype=numpy.float32)
        indexes_ptr = &indexes[0]
        coefs_ptr = &coefs[0]

        it_packed = get_internal_data(&self._data)._heap._packed_data.begin()
        while it_packed != get_internal_data(&self._data)._heap._packed_data.end():
            packed_block = dereference(it_packed)

            for i in range(get_internal_data(&self._data)._heap._block_size):
                if get_internal_data(&self._data)._heap._current_packed_block == packed_block:
                    if i >= get_internal_data(&self._data)._heap._packed_pos:
                        break
                packed_data = &packed_block[i]
                bin_id = packed_data.bin_id
                pos = current_bin_pos[bin_id]
                indexes_ptr[pos] = packed_data.data.index
                coefs_ptr[pos] = packed_data.data.coef
                current_bin_pos[bin_id] += 1

            preincrement(it_packed)

        return numpy.asarray(coefs), numpy.asarray(indexes), numpy.asarray(nbins)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def to_csr(self):
        """
        Returns a CSR representation from the stored data.

        - The first array contains all floating values. Sorted by bin number.
        - The second array contains all indices. Sorted by bin number.
        - Lookup table from the bin index to the first index in the 2 first
            arrays. `array[10 + 0]` contains the index of the first element of the
            bin 10. `array[10 + 1]` - 1 is the last elements. This array always
            starts with `0` and contains one more element than the number of
            bins.

        :rtype: Tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
        :returns: A tuple containing values, indices and bin indexes
        """
        cdef:
            cnumpy.int32_t[:] indexes
            cnumpy.float32_t[:] coefs
            cnumpy.float32_t *coefs_ptr
            cnumpy.int32_t[:] nbins
            cnumpy.int32_t *indexes_ptr
            int size
            int begin, end
            int bin_id
            int bin_size

        if self._use_packed_list:
            return self._to_csr_from_packed()

        # indexes of the first and the last+1 elements of each bins
        size = 0
        nbins = numpy.empty(self._nbin + 1, dtype=numpy.int32)
        nbins[0] = size
        for bin_id in range(self._nbin):
            bin_size = self.cget_bin_size(bin_id)
            size += bin_size
            nbins[bin_id + 1] = size

        indexes = numpy.empty(size, dtype=numpy.int32)
        coefs = numpy.empty(size, dtype=numpy.float32)
        indexes_ptr = &indexes[0]
        coefs_ptr = &coefs[0]

        for bin_id in range(self._nbin):
            begin = nbins[bin_id]
            end = nbins[bin_id + 1]
            if begin == end:
                continue
            self._copy_bin_indexes_to(bin_id, indexes_ptr)
            self._copy_bin_coefs_to(bin_id, coefs_ptr)
            indexes_ptr += end - begin
            coefs_ptr += end - begin

        return numpy.asarray(coefs), numpy.asarray(indexes), numpy.asarray(nbins)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def _to_lut_from_packed(self):
        cdef:
            cnumpy.int32_t[:] current_bin_pos
            cnumpy.int32_t[:] indexes
            cnumpy.int32_t *indexes_ptr
            cnumpy.float32_t[:] coefs
            cnumpy.float32_t *coefs_ptr
            pixel_t[:, :] lut
            int size
            int max_size
            int begin, end
            int bin_id
            int bin_size
            int pos
            packed_data_t *packed_block
            packed_data_t *packed_data

        # Reach the biggest bin size
        max_size = 0
        for bin_id in range(self._nbin):
            size = self.cget_bin_size(bin_id)
            if size > max_size:
                max_size = size

        # Alloc a very big array
        lut = numpy.zeros((self._nbin, max_size), dtype=lut_d)

        # Feed the array
        current_bin_pos = numpy.zeros(self._nbin, dtype=numpy.int32)
        it_packed = get_internal_data(&self._data)._heap._packed_data.begin()
        while it_packed != get_internal_data(&self._data)._heap._packed_data.end():
            packed_block = dereference(it_packed)

            for i in range(get_internal_data(&self._data)._heap._block_size):
                if get_internal_data(&self._data)._heap._current_packed_block == packed_block:
                    if i >= get_internal_data(&self._data)._heap._packed_pos:
                        break
                packed_data = &packed_block[i]
                bin_id = packed_data.bin_id
                pos = current_bin_pos[bin_id]
                lut[bin_id, pos] = packed_data.data
                current_bin_pos[bin_id] += 1

            preincrement(it_packed)

        return numpy.asarray(lut, dtype=lut_d)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def to_lut(self):
        """
        Returns a LUT representation from the stored data.

        - The first array contains all floating values. Sorted by bin number.
        - The second array contains all indices. Sorted by bin number.
        - Lookup table from the bin index to the first index in the 2 first
            arrays. `array[10 + 0]` contains the index of the first element of the
            bin 10. `array[10 + 1]` - 1 is the last elements. This array always
            starts with `0` and contains one more element than the number of
            bins.

        :rtype: numpy.ndarray
        :returns: A 2D array  tuple containing values, indices and bin indexes
        """
        cdef:
            pixel_t[:, :] lut
            pixel_t *data_ptr
            int bin_id
            int i
            int max_size
            int size
            cnumpy.int32_t[:] indexes
            cnumpy.float32_t[:] coefs
            cnumpy.float32_t *coefs_ptr
            cnumpy.int32_t *indexes_ptr

        if self._use_packed_list:
            return self._to_lut_from_packed()

        # Reach the biggest bin size
        max_size = 0
        for bin_id in range(self._nbin):
            size = self.cget_bin_size(bin_id)
            if size > max_size:
                max_size = size

        # Alloc a very big array
        lut = numpy.zeros((self._nbin, max_size), dtype=lut_d)

        # Feed the array
        for bin_id in range(self._nbin):
            data_ptr = &lut[bin_id, 0]
            self._copy_bin_data_to(bin_id, data_ptr)

        return numpy.asarray(lut, dtype=lut_d)
