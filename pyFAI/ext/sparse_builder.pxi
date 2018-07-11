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


cdef cppclass PixelElementaryBlock:
    cnumpy.int32_t *_indexes
    cnumpy.float32_t *_coefs
    int _size
    int _max_size

    PixelElementaryBlock(int size) nogil:
        this._indexes = <cnumpy.int32_t *>libc.stdlib.malloc(size * sizeof(cnumpy.int32_t))
        this._coefs = <cnumpy.float32_t *>libc.stdlib.malloc(size * sizeof(cnumpy.float32_t))
        this._size = 0
        this._max_size = size

    __dealloc__() nogil:
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


cdef cppclass PixelBlock:
    clist[PixelElementaryBlock*] _blocks
    int _block_size

    PixelBlock(int block_size) nogil:
        this._block_size = block_size

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
        if _blocks.size() == 0 or this._blocks.back().is_full():
            block = new PixelElementaryBlock(size=this._block_size)
            this._blocks.push_back(block)
        block = this._blocks.back()
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

    PixelBin(int block_size) nogil:
        if block_size > 0:
            this._pixels_in_block = new PixelBlock(block_size)
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
    cdef int _nbin
    cdef int _block_size

    def __init__(self, nbin, block_size=512):
        self._block_size = block_size
        self._nbin = nbin
        self._bins = <PixelBin **>libc.stdlib.malloc(self._nbin * sizeof(PixelBin *))
        libc.string.memset(self._bins, 0, self._nbin * sizeof(PixelBin *))

    def __dealloc__(self):
        cdef:
            PixelBin *pixel_bin
            int i
        for i in range(self._nbin):
            pixel_bin = self._bins[i]
            if pixel_bin != NULL:
                del pixel_bin
        libc.stdlib.free(self._bins)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef PixelBin *_create_bin(self) nogil:
        return new PixelBin(self._block_size)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void cinsert(self, int bin_id, int index, cnumpy.float32_t coef) nogil:
        cdef:
            pixel_t pixel
            PixelBin *pixel_bin
        if bin_id < 0 or bin_id >= self._nbin:
            return
        pixel.index = index
        pixel.coef = coef

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
    def get_bin_size(self, bin_id):
        cdef:
            PixelBin *pixel_bin
        if bin_id < 0 or bin_id >= self._nbin:
            raise ValueError("bin_id out of range")
        pixel_bin = self._bins[bin_id]
        if pixel_bin == NULL:
            return 0
        return pixel_bin.size()

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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def to_csr(self):
        cdef:
            cnumpy.int32_t[:] indexes
            cnumpy.float32_t[:] coefs
            cnumpy.int32_t[:] nbins
            PixelBin *pixel_bin
            int size
            int begin, end
            int bin_id
            int bin_size
            cnumpy.int32_t *indexes_ptr
            cnumpy.float32_t *coefs_ptr

        # indexes of the first and the last+1 elements of each bins
        size = 0
        nbins = numpy.empty(self._nbin + 1, dtype=numpy.int32)
        nbins[0] = size
        for bin_id in range(self._nbin):
            pixel_bin = self._bins[bin_id]
            if pixel_bin != NULL:
                bin_size = pixel_bin.size()
            else:
                bin_size = 0
            size += bin_size
            nbins[bin_id + 1] = size

        indexes = numpy.empty(size, dtype=numpy.int32)
        coefs = numpy.empty(size, dtype=numpy.float32)
        indexes_ptr = &indexes[0]
        coefs_ptr = &coefs[0]

        for bin_id in range(self._nbin):
            pixel_bin = self._bins[bin_id]
            if pixel_bin == NULL or pixel_bin.size() == 0:
                continue
            begin = nbins[bin_id]
            end = nbins[bin_id + 1]
            pixel_bin.copy_indexes_to(indexes_ptr)
            pixel_bin.copy_coefs_to(coefs_ptr)
            indexes_ptr += end - begin
            coefs_ptr += end - begin

        return numpy.asarray(coefs), numpy.asarray(indexes), numpy.asarray(nbins)
