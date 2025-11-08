# -*- coding: utf-8 -*-
#cython: embedsignature=True, language_level=3, binding=True
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False,
################################################################################
# #This is for developping
##cython: profile=True, warn.undeclared=True, warn.unused=True, warn.unused_result=False, warn.unused_arg=True
################################################################################
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2025-2025 European Synchrotron Radiation Facility, Grenoble, France
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

"""
The class is doing raytracing to assess how much energy a photon entering a
given pixel is depositing energy in the different adjacent pixels.

It is used to produce a sparse matrix which represents the "blurring".
pixel close to the PONI will probably not contaminate much, but those far away
-the ones with the most inclined rays- will spread over a large area.

The built sparse matrix can be used for deconvolution with Richardson-Lucy or
other methods like MLEM.
"""

__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__date__ = "07/11/2025"
__copyright__ = "2011-2022, ESRF"
__contact__ = "jerome.kieffer@esrf.fr"

from .shared_types cimport int8_t, uint8_t, int16_t, uint16_t, \
                           int32_t, uint32_t, int64_t, uint64_t,\
                           float32_t, float64_t
import logging
import cython
import numpy
from libc.math cimport sqrt, exp
from cython.parallel import prange

logger = logging.getLogger(__name__)

cdef float64_t BIG = numpy.finfo(numpy.float32).max


cdef class Raytracing:
    "Calculate the point spread function as function of the geometry of the experiment"
    cdef:
        public float64_t vox
        public float64_t voy
        public float64_t voz
        public float64_t mu
        public float64_t dist
        public float64_t poni1
        public float64_t poni2
        public int oversampling
        public int buffer_size
        public int width
        public int height
        public int size
        public int8_t[:, ::1] mask

    def __init__(self,
                 geom,
                 int buffer_size=16):
        """Constructor of the class:

        :param geom: Geometry instance with parallax enabled
        :param buffer_size: how many pixel a photon could be spread on ... if too small, you will get a warning (not an error)
        """
        if not geom.parallax:
            raise RuntimeError("Expected a geometry with parallax enabled")
        detector = geom.detector
        shape = detector.shape
        self.vox = float(detector.pixel2)
        self.voy = float(detector.pixel2)
        self.voz = float(detector.sensor.thickness)
        self.mu = float(geom.parallax.sensor.mu)
        self.dist = float(geom.dist)
        self.poni1 = float(geom.poni1)
        self.poni2 = float(geom.poni2)
        self.width = int(detector.shape[1])
        self.height = int(detector.shape[0])
        self.size = self.width*self.height
        if detector.mask is None:
            self.mask = numpy.zeros(shape, numpy.int8)
        else:
            self.mask = numpy.ascontiguousarray(detector.mask, numpy.int8)
        self.oversampling = 1
        self.buffer_size = int(buffer_size)

    def calc_one_ray(self, entx, enty):
        """For a ray entering at position (entx, enty), with a propagation vector (kx, ky,kz),
        calculate the length spent in every voxel where energy is deposited from a bunch of photons entering the detector
        at a given position and and how much energy they deposit in each voxel.

        Direct implementation of http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.42.3443&rep=rep1&type=pdf

        :param entx, enty: coordinate of the entry point in meter (2 components, x,y)
        :return: coordinates voxels in x, y and length crossed when leaving the associated voxel
        """

        cdef:
            int last_id
            float64_t _entx = float(entx)
            float64_t _enty = float(enty)
            int32_t[::1] array_x = numpy.empty(self.buffer_size, dtype=numpy.int32)
            int32_t[::1] array_y = numpy.empty(self.buffer_size, dtype=numpy.int32)
            float32_t[::1] array_len = numpy.empty(self.buffer_size, dtype=numpy.float32)
        with nogil:
            last_id = self._calc_one_ray(_entx, _enty,
                                         array_x, array_y, array_len)
        if last_id>self.buffer_size:
            raise RuntimeError(f"Temporary buffer size ({last_id}) larger than expected ({self.buffer_size})")
        return (numpy.asarray(array_x[:last_id]),
                numpy.asarray(array_y[:last_id]),
                numpy.asarray(array_len[:last_id]))

    cdef int _calc_one_ray(self,
                           float64_t entx,
                           float64_t enty,
                           int32_t[::1] array_x,
                           int32_t[::1] array_y,
                           float32_t[::1] array_len
                          )noexcept nogil:
        """Return number of entries in the array_x[:last_id], array_y[:last_id], array_len[:last_id]"""
        cdef:
            float64_t kx, ky, kz, n, t_max_x, t_max_y, t_max_z, t_delta_x, t_delta_y#, t_delta_z
            int step_X, step_Y, X, Y, last_id
            bint finished

        # reset arrays
        array_x[:] = -1
        array_y[:] = -1
        array_len[:] = 0.0

        # normalize the input propagation vector
        kx = entx - self.poni2
        ky = enty - self.poni1
        kz = self.dist
        n = sqrt(kx*kx + ky*ky + kz*kz)
        kx /= n
        ky /= n
        kz /= n

        step_X = -1 if kx<0.0 else 1
        step_Y = -1 if ky<0.0 else 1

        X = int(entx/self.vox)
        Y = int(enty/self.voy)

        if kx>0.0:
            t_max_x = ((entx//self.vox+1)*(self.vox)-entx)/ kx
        elif kx<0.0:
            t_max_x = ((entx//self.vox)*(self.vox)-entx)/ kx
        else:
            t_max_x = BIG

        if ky>0.0:
            t_max_y = ((enty//self.voy+1)*(self.voy)-enty)/ ky
        elif ky<0.0:
            t_max_y = ((enty//self.voy)*(self.voy)-enty)/ ky
        else:
            t_max_y = BIG

        #Only one case for z as the ray is travelling in one direction only
        t_max_z = self.voz / kz

        t_delta_x = abs(self.vox/kx) if kx!=0 else BIG
        t_delta_y = abs(self.voy/ky) if ky!=0 else BIG
        # t_delta_z = self.voz/kz

        finished = False
        last_id = 0
        array_x[last_id] = X
        array_y[last_id] = Y

        while not finished:
            if t_max_x < t_max_y:
                if t_max_x < t_max_z:
                    array_len[last_id] = t_max_x
                    last_id = last_id + 1
                    X = X + step_X
                    array_x[last_id] = X
                    array_y[last_id] = Y
                    t_max_x = t_max_x + t_delta_x
                else:
                    array_len[last_id] = t_max_z
                    last_id = last_id + 1
                    finished = True
            else:
                if t_max_y < t_max_z:
                    array_len[last_id] = t_max_y
                    last_id = last_id +1
                    Y = Y + step_Y
                    array_x[last_id] = X
                    array_y[last_id] = Y
                    t_max_y = t_max_y + t_delta_y
                else:
                    array_len[last_id] = t_max_z
                    last_id = last_id +1
                    finished = True
            if last_id>=self.buffer_size:
                return self.buffer_size
        return last_id


    def one_pixel(self, row, col, sample=0):
        """calculate the contribution of one pixel to the sparse matrix and populate it.

        :param row: row index of the pixel of interest
        :param col: column index of the pixel of interest
        :param sample: Oversampling rate, 10 will cast 10x10 ray per pixel
        """
        cdef:
            int entries = 0
            int _row = int(row)
            int _col = int(col)
            int32_t[::1] tmp_idx = numpy.empty(self.buffer_size, dtype=numpy.int32)
            float32_t[::1] tmp_coef = numpy.empty(self.buffer_size, dtype=numpy.float32)
            int32_t[::1] array_x = numpy.empty(self.buffer_size, dtype=numpy.int32)
            int32_t[::1] array_y = numpy.empty(self.buffer_size, dtype=numpy.int32)
            float32_t[::1] array_len = numpy.empty(self.buffer_size, dtype=numpy.float32)

        if  sample:
            self.oversampling = sample
        with nogil:
            entries = self._one_pixel(_row, _col, tmp_idx, tmp_coef, array_x, array_y, array_len)
        if entries<self.buffer_size:
            return (numpy.asarray(tmp_idx)[:entries],numpy.asarray(tmp_coef)[:entries])
        else:
            raise RuntimeError(f"Pixel produced {entries} voxels, limited to {self.buffer_size}. Increase Buffer size ! in constructor")

    cdef int _one_pixel(self, int row, int col,
                       int32_t[::1] tmp_idx,
                       float32_t[::1] tmp_coef,
                       int32_t[::1] array_x,
                       int32_t[::1] array_y,
                       float32_t[::1] array_len
                    )noexcept nogil:
        "return the number of elements in tmp_idx/tmp_coef which are to be used"

        cdef:
            int i, j, tmp_size, last_buffer_size, n, x, y, idx
            float64_t rem, l, posx, posy, value, dos

        if self.mask[row, col]:
            return 0

        tmp_size = 0
        last_buffer_size = tmp_idx.shape[0]
        tmp_idx[:] = -1
        tmp_coef[:] = 0.0

        for i in range(self.oversampling):
            posx = (col+1.0*i/self.oversampling)*self.vox
            for j in range(self.oversampling):
                posy = (row+1.0*j/self.oversampling)*self.voy
                n = self._calc_one_ray(posx, posy, array_x, array_y, array_len)
                rem = 1.0
                for i in range(min(n, self.buffer_size)):
                    x = array_x[i]
                    y = array_y[i]
                    l = array_len[i]
                    if (x<0) or (y<0) or (y>=self.height) or (x>=self.width):
                        break
                    elif (self.mask[y, x]):
                        continue
                    idx = x + y*self.width
                    dos = exp(-self.mu*l)
                    value = rem - dos
                    rem = dos
                    for j in range(self.buffer_size):
                        if tmp_idx[j] == idx:
                            tmp_coef[j] = tmp_coef[j] + value
                            break
                        elif tmp_idx[j] < 0:
                            tmp_idx[j] = idx
                            tmp_coef[j] = value
                            tmp_size = tmp_size + 1
                            break
                    if tmp_size >= self.buffer_size:
                            break
        return tmp_size

    def calc_csr(self, sample=0, int threads=0):
        """Calculate the content of the sparse matrix for the whole image in CSR format.
        The blurring matrix is actually the transposed array.

        :param sample: Oversampling factor, actually if you request 2 it will trace 2x2 rays
        :param threads: number of threads to be used, 0 for max_threads
        :return: 3-tuple of arrays to build a CSR sparse matrix (in scipy)
        """
        cdef:
            int pos, i, current, next, size
            int32_t[::1] sizes, indptr = numpy.zeros(self.size+1, dtype=numpy.int32)
            int32_t[:, ::1] indices
            float32_t[:, ::1] data
            int32_t[::1] csr_indices
            float32_t[::1] csr_data
            int32_t[:, ::1] array_x
            int32_t[:, ::1] array_y
            float32_t[:, ::1] array_len

        if sample:
            self.oversampling = sample
        self.oversampling = sample
        data = numpy.zeros((self.size, self.buffer_size), dtype=numpy.float32)
        indices = numpy.zeros((self.size, self.buffer_size),dtype=numpy.int32)
        sizes = numpy.zeros(self.size, dtype=numpy.int32)

        #single threaded version:
        array_x = numpy.empty((self.size, self.buffer_size), dtype=numpy.int32)
        array_y = numpy.empty((self.size, self.buffer_size), dtype=numpy.int32)
        array_len = numpy.empty((self.size, self.buffer_size), dtype=numpy.float32)

        for pos in prange(self.size, num_threads=threads, nogil=True):
            size = sizes[pos] = self._one_pixel(pos//self.width, pos%self.width,
                                             indices[pos], data[pos],
                                             array_x[pos], array_y[pos], array_len[pos])
        if numpy.max(sizes) >= self.buffer_size:
            logger.warning("It is very possible that some rays were not recorded properly due to a buffer too small."
                           f" Please restart with a buffer size >{self.buffer_size} !")
        size = numpy.sum(sizes)
        csr_indices = numpy.empty(size, numpy.int32)
        csr_data = numpy.empty(size, numpy.float32)
        current = 0
        for i in range(self.size):
            size = sizes[i]
            next = current + size
            indptr[i+1] = next
            csr_indices[current:next] = indices[i,:size]
            csr_data[current:next] = data[i,:size]
            current = next
        return (numpy.asarray(csr_data)/(self.oversampling*self.oversampling),
                numpy.asarray(csr_indices),
                numpy.asarray(indptr))
