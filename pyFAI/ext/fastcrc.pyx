# -*- coding: utf-8 -*-
#cython: embedsignature=True, language_level=3, binding=True
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False,
## This is for developping:
##cython: profile=True, warn.undeclared=True, warn.unused=True, warn.unused_result=False, warn.unused_arg=True
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2020 European Synchrotron Radiation Facility, Grenoble, France
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

"""
Simple Cython module for doing CRC32 for checksums, possibly with SSE4 acceleration
"""
__author__ = "Jérôme Kieffer"
__date__ = "04/01/2023"
__contact__ = "Jerome.Kieffer@esrf.fr"
__license__ = "MIT"

import time
import zlib
import cython
import numpy
from libc.stdint cimport uint8_t, uint32_t, int8_t
from .crc32 cimport (pyFAI_crc32, _get_crc32_table, _get_crc32_table_key, _check_sse4, _is_crc32_sse4_available, _crc32_sse4, _crc32_table, _crc32_table_init)

def get_crc_table_key():
    "KEY with which the table was initialized"
    return _get_crc32_table_key()

def is_crc32_sse4_available():
    "Tells if the SSE4 implementation is available"
    return _is_crc32_sse4_available()

def check_sse4():
    "Checks if the SSE4 implementation is available"
    return bool(_check_sse4())

def get_crc_table():
    "retruns the internal table for calculating the CRC"
    cdef:
        uint32_t[::1] table = numpy.empty(1<<8, dtype=numpy.uint32)
    _get_crc32_table(<uint32_t*> &table[0])
    return numpy.asarray(table)

def init_crc32_table(uint32_t key=0x1EDC6F41):
    "Initialize the CRC table"
    _crc32_table_init(key)
    return key

def crc32_table(data):
    "Calculate the CRC32 on the data using the look-up table"
    cdef:
        uint32_t size
        uint8_t[::1] view
    view = data.ravel().view(numpy.uint8)
    size = view.shape[0]
    return _crc32_table(<char *> &view[0], size)

def crc32_sse4(data):
    "Calculate the CRC32 on the data using the SSE4 implementation"
    cdef:
        uint32_t size
        uint8_t[::1] view
    view = data.ravel().view(numpy.uint8)
    size = view.shape[0]
    return _crc32_sse4(<char *> &view[0], size)


cdef class SlowCRC:
    """This class implements a fail-safe version of CRC using a look-up table ... not very performant"""
    cdef:
        readonly uint32_t initialized
        readonly uint32_t[::1] table

    def __cinit__(self, uint32_t key=0x1EDC6F41):
        cdef:
            uint32_t i, j, a, s=1<<8
        self.initialized = key
        self.table = numpy.empty(s, dtype=numpy.uint32)

        for i in range(s):
            a = i << 24;
            for j in range(8):
                if (a & 0x80000000):
                    a = (a << 1) ^ key;
                else:
                    a = (a << 1);
            self.table[i] = a

    def __dealloc__(self):
        self.initialized = 0
        self.table = None

    def crc(self, buffer):
        """Calculate the CRC checksum of the numpy array"""
        cdef:
            uint32_t i, size, lcrc=0xffffffff
            uint8_t[::1] view
        view = buffer.ravel().view(numpy.uint8)
        size = view.shape[0]
        for i in range(size):
            lcrc = (lcrc >> 8) ^ self.table[(lcrc ^ (view[i])) & 0xff]
        return ~lcrc

config = {"SSE4": check_sse4(),
          "key": init_crc32_table()}

# benchmark the different implementations
dummy = numpy.arange(1<<20)
start_time = time.perf_counter()
zlib.crc32(dummy)
time_zlib = time.perf_counter() - start_time
start_time = time.perf_counter()
crc32_table(dummy)
time_table = time.perf_counter() - start_time
if time_table<time_zlib:
    crc32 =  config["default"] = crc32_table
    config["fastest"] = "table"
else:
    crc32 = config["default"] = zlib.crc32
    config["fastest"] = "zlib"
if config["SSE4"]:
    start_time = time.perf_counter()
    crc32_sse4(dummy)
    time_sse4 = time.perf_counter() - start_time
    if time_sse4< min(time_table, time_zlib):
        crc32 = config["default"] = crc32_sse4
        config["fastest"] = "sse4"

# clean up array
del dummy
