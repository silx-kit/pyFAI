"""This replaces the `cimport numpy` for C-level datatypes"""

from libc.stdint cimport int8_t, uint8_t, int16_t, uint16_t, \
                         int32_t, uint32_t, int64_t, uint64_t

ctypedef double float64_t
ctypedef float float32_t

from cython cimport floating

ctypedef fused any_int_t:
    uint8_t
    uint16_t
    uint32_t
    uint64_t
    int8_t
    int16_t
    int32_t
    int64_t

ctypedef fused any_t:
    int
    long
    uint8_t
    uint16_t
    uint32_t
    uint64_t
    int8_t
    int16_t
    int32_t
    int64_t
    float32_t
    float64_t
