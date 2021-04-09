"""This replaces the `cimport numpy` for C-level datatypes"""

from libc.stdint cimport int8_t, uint8_t, int16_t, uint16_t, \
                         int32_t, uint32_t, int64_t, uint64_t

ctypedef double float64_t
ctypedef float float32_t
