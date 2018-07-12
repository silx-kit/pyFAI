import numpy
cimport numpy as cnumpy

from libcpp.vector cimport vector
from libcpp.list cimport list as clist
from libcpp cimport bool
from libc.math cimport fabs
cimport libc.stdlib
cimport libc.string

from cython.parallel import prange
from cython.operator cimport dereference
from cython.operator cimport preincrement
cimport cython
from cython cimport floating

cdef double EPS32 = (1.0 + numpy.finfo(numpy.float32).eps)


cdef struct pixel_t:
    cnumpy.int32_t index
    cnumpy.float32_t coef


cdef struct chained_pixel_t:
    pixel_t data
    chained_pixel_t *next


cdef struct compact_bin_t:
    int size
    chained_pixel_t *front_ptr
    chained_pixel_t *back_ptr


cdef cppclass Heap:
    clist[cnumpy.int32_t *] _indexes
    clist[cnumpy.float32_t *] _coefs
    clist[chained_pixel_t *] _pixels

    cnumpy.int32_t *_current_index_block
    cnumpy.float32_t *_current_coef_block
    chained_pixel_t *_current_pixel_block

    int _index_pos
    int _coef_pos
    int _pixel_pos
    int _block_size

    Heap(int block_size) nogil:
        this._block_size = block_size
        this._index_pos = 0
        this._coef_pos = 0
        this._current_index_block = NULL
        this._current_coef_block = NULL
        this._current_pixel_block = NULL

    __dealloc__() nogil:
        cdef:
            clist[cnumpy.int32_t *].iterator it_indexes
            clist[cnumpy.float32_t *].iterator it_coefs
            clist[chained_pixel_t *].iterator it_pixels
            cnumpy.int32_t *indexes
            cnumpy.float32_t *coefs
            chained_pixel_t *pixels

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

    cnumpy.int32_t * alloc_indexes(int size) nogil:
        cdef:
            cnumpy.int32_t *data
        if this._current_index_block == NULL or this._index_pos + size >= this._block_size:
            data = <cnumpy.int32_t *>libc.stdlib.malloc(this._block_size * sizeof(cnumpy.int32_t))
            this._current_index_block = data
            this._indexes.push_back(data)
            this._index_pos = 0
        else:
            this._index_pos += size
        return this._current_index_block + this._index_pos

    cnumpy.float32_t * alloc_coefs(int size) nogil:
        cdef:
            cnumpy.float32_t *data
        if this._current_coef_block == NULL or this._coef_pos + size >= this._block_size:
            data = <cnumpy.float32_t *>libc.stdlib.malloc(this._block_size * sizeof(cnumpy.float32_t))
            this._current_coef_block = data
            this._coefs.push_back(data)
            this._coef_pos = 0
        else:
            this._coef_pos += size
        return this._current_coef_block + this._coef_pos

    chained_pixel_t* alloc_pixel() nogil:
        cdef:
            chained_pixel_t *data
            int foo
        if this._current_pixel_block == NULL or this._pixel_pos + 1 >= this._block_size:
            data = <chained_pixel_t *>libc.stdlib.malloc(this._block_size * sizeof(chained_pixel_t))
            this._current_pixel_block = data
            this._pixels.push_back(data)
            this._pixel_pos = 0
        else:
            this._pixel_pos += 1
        return this._current_pixel_block + this._pixel_pos


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
        return this._size == this._max_size

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

    cnumpy.int32_t[:] index_array():
        cdef:
            clist[PixelElementaryBlock*].iterator it
            PixelElementaryBlock* block
            int size
            int begin
            cnumpy.int32_t[:] data
            cnumpy.int32_t *data_dest

        size = this.size()
        data = numpy.empty(size, dtype=numpy.int32)
        data_dest = &data[0]
        this.copy_indexes_to(data_dest)
        return data

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

    cnumpy.float32_t[:] coef_array():
        cdef:
            clist[PixelElementaryBlock*].iterator it
            PixelElementaryBlock* block
            int size
            int begin
            cnumpy.float32_t[:] data
            cnumpy.float32_t *data_dest

        size = this.size()
        data = numpy.empty(size, dtype=numpy.float32)
        data_dest = &data[0]
        this.copy_coefs_to(data_dest)
        return data


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

    cnumpy.int32_t[:] index_array():
        cdef:
            int i = 0
            clist[pixel_t].iterator it_points

        if this._pixels_in_block != NULL:
            return this._pixels_in_block.index_array()

        data = numpy.empty(this.size(), dtype=numpy.int32)
        it_points = this._pixels.begin()
        while it_points != this._pixels.end():
            data[i] = dereference(it_points).index
            preincrement(it_points)
            i += 1
        return data

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

    cnumpy.float32_t[:] coef_array():
        cdef:
            int i = 0
            clist[pixel_t].iterator it_points

        if this._pixels_in_block != NULL:
            return this._pixels_in_block.coef_array()

        data = numpy.empty(this.size(), dtype=numpy.float32)
        it_points = this._pixels.begin()
        while it_points != this._pixels.end():
            data[i] = dereference(it_points).coef
            preincrement(it_points)
            i += 1
        return data


cdef class SparseBuilder(object):

    cdef PixelBin **_bins
    cdef compact_bin_t *_compact_bins
    cdef int _nbin
    cdef int _block_size
    cdef Heap *_heap
    cdef bool _use_linked_list
    cdef bool _use_blocks
    cdef bool _use_heap_linked_list

    def __init__(self, nbin, block_size=512, heap_size=0):
        self._block_size = block_size
        self._nbin = nbin
        if heap_size != 0:
            if heap_size < block_size:
                raise ValueError("Heap size is supposed to be bigger than block size")
            self._heap = new Heap(heap_size)
        else:
            self._heap = NULL

        self._use_linked_list = False
        self._use_blocks = False
        self._use_heap_linked_list = False
        if block_size > 1:
            self._use_blocks = True
        else:
            if heap_size == 0:
                self._use_linked_list = True
            else:
                self._use_heap_linked_list = True

        if self._use_blocks:
            self._bins = <PixelBin **>libc.stdlib.malloc(self._nbin * sizeof(PixelBin *))
            libc.string.memset(self._bins, 0, self._nbin * sizeof(PixelBin *))
        elif self._use_heap_linked_list:
            self._compact_bins = <compact_bin_t *>libc.stdlib.malloc(self._nbin * sizeof(compact_bin_t))
            libc.string.memset(self._compact_bins, 0, self._nbin * sizeof(compact_bin_t))

    def __dealloc__(self):
        cdef:
            PixelBin *pixel_bin
            clist[PixelElementaryBlock*].iterator it_points
            PixelElementaryBlock* heap
            int i

        if self._use_blocks:
            for i in range(self._nbin):
                pixel_bin = self._bins[i]
                if pixel_bin != NULL:
                    del pixel_bin
            libc.stdlib.free(self._bins)
        elif self._use_heap_linked_list:
            libc.stdlib.free(self._compact_bins)

        if self._heap != NULL:
            del self._heap

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef PixelBin *_create_bin(self) nogil:
        return new PixelBin(self._block_size, self._heap)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void cinsert(self, int bin_id, int index, cnumpy.float32_t coef) nogil:
        cdef:
            pixel_t pixel
            PixelBin *pixel_bin
            chained_pixel_t* chained_pixel
        if bin_id < 0 or bin_id >= self._nbin:
            return
        pixel.index = index
        pixel.coef = coef

        if self._use_heap_linked_list:
            chained_pixel = self._heap.alloc_pixel()
            chained_pixel.data = pixel
            if self._compact_bins[bin_id].front_ptr == NULL:
                self._compact_bins[bin_id].front_ptr = chained_pixel
            else:
                self._compact_bins[bin_id].back_ptr.next = chained_pixel
            self._compact_bins[bin_id].back_ptr = chained_pixel
            self._compact_bins[bin_id].size += 1
        else:
            pixel_bin = self._bins[bin_id]
            if pixel_bin == NULL:
                pixel_bin = self._create_bin()
                self._bins[bin_id] = pixel_bin
            self._bins[bin_id].push(pixel)

    def insert(self, bin_id, index, coef):
        if bin_id < 0 or bin_id >= self._nbin:
            raise ValueError("bin_id out of range")
        self.cinsert(bin_id, index, coef)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def get_bin_coefs(self, bin_id):
        cdef:
            PixelBin *pixel_bin
        if bin_id < 0 or bin_id >= self._nbin:
            raise ValueError("bin_id out of range")
        pixel_bin = self._bins[bin_id]
        if pixel_bin == NULL:
            return numpy.empty(shape=(0, 1), dtype=numpy.float32)
        return numpy.array(pixel_bin.coef_array())

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def get_bin_indexes(self, bin_id):
        cdef:
            PixelBin *pixel_bin
        if bin_id < 0 or bin_id >= self._nbin:
            raise ValueError("bin_id out of range")
        pixel_bin = self._bins[bin_id]
        if pixel_bin == NULL:
            return numpy.empty(shape=(0, 1), dtype=numpy.int32)
        return numpy.array(pixel_bin.index_array())

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int cget_bin_size(self, bin_id):
        cdef:
            PixelBin *pixel_bin
        if self._use_heap_linked_list:
            return self._compact_bins[bin_id].size
        pixel_bin = self._bins[bin_id]
        if pixel_bin == NULL:
            return 0
        return pixel_bin.size()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def get_bin_size(self, bin_id):
        if bin_id < 0 or bin_id >= self._nbin:
            raise ValueError("bin_id out of range")
        return self.cget_bin_size(bin_id)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def size(self):
        cdef:
            PixelBin *pixel_bin
            int size
            int bin_id

        size = 0
        for bin_id in range(self._nbin):
            pixel_bin = self._bins[bin_id]
            if pixel_bin != NULL:
                size += pixel_bin.size()
        return size

    cdef void copy_bin_indexes_to(self, int bin_id, cnumpy.int32_t *dest) nogil:
        cdef:
            PixelBin *pixel_bin
            compact_bin_t *compact_bin
            chained_pixel_t *chained_pixel
        if self._use_heap_linked_list:
            compact_bin = &self._compact_bins[bin_id]
            chained_pixel = compact_bin.front_ptr
            while chained_pixel != NULL:
                dest[0] = chained_pixel.data.index
                dest += 1
                if chained_pixel == compact_bin.back_ptr:
                    # The next ptr of the last element is not initialized
                    break
                chained_pixel = chained_pixel.next
        else:
            pixel_bin = self._bins[bin_id]
            if pixel_bin != NULL:
                pixel_bin.copy_indexes_to(dest)

    cdef void copy_bin_coefs_to(self, int bin_id, cnumpy.float32_t *dest) nogil:
        cdef:
            PixelBin *pixel_bin
            compact_bin_t *compact_bin
            chained_pixel_t *chained_pixel
        if self._use_heap_linked_list:
            compact_bin = &self._compact_bins[bin_id]
            chained_pixel = compact_bin.front_ptr
            while chained_pixel != NULL:
                dest[0] = chained_pixel.data.coef
                dest += 1
                if chained_pixel == compact_bin.back_ptr:
                    # The next ptr of the last element is not initialized
                    break
                chained_pixel = chained_pixel.next
        else:
            pixel_bin = self._bins[bin_id]
            if pixel_bin != NULL:
                pixel_bin.copy_coefs_to(dest)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def to_csr(self):
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
            self.copy_bin_indexes_to(bin_id, indexes_ptr)
            self.copy_bin_coefs_to(bin_id, coefs_ptr)
            indexes_ptr += end - begin
            coefs_ptr += end - begin

        return numpy.asarray(coefs), numpy.asarray(indexes), numpy.asarray(nbins)
