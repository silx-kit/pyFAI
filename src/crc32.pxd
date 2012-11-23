from libc.stdint cimport uint32_t
cdef extern from "crc32.h":
#    void slowcrc_init()
#    uint32_t slowcrc(char * str, uint32_t len)
#    uint32_t fastcrc(char * str, uint32_t len)
    uint32_t crc32(char * str, uint32_t len)
