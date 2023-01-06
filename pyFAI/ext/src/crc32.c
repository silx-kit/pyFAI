/*
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <stdint.h>
#include "hwinfo.h"
#include "crc32.h"

#define bit_SSE4_2 (1<<20)

#if defined(HWINFO_CPU_X86)

#if defined(_MSC_VER)
// Windows

#if defined(__clang__)
//clang on windows !
#include <Intrin.h>
#endif

static inline void cpuid(uint32_t op, uint32_t *eax, uint32_t *ebx, uint32_t *ecx, uint32_t *edx){
    uint32_t ary[4];
    __cpuidex(&ary, op, 0);
    eax[0] = ary[0];
    ebx[0] = ary[1];
    ecx[0] = ary[2];
    edx[0] = ary[3];
}

#elif defined(HWINFO_APPLE)
// MacOS
static inline void cpuid(uint32_t op, uint32_t *eax, uint32_t *ebx, uint32_t *ecx, uint32_t *edx){
    __asm__ volatile ("cpuid":
                  "=a" (*eax),
                  "=b" (*ebx),
                  "=c" (*ecx),
                  "=d" (*edx):
                  "a"(op),
                  "b" (0),
                  "c" (0),
                  "d" (0));
}

#elif defined(__x86_64__) || defined(_M_AMD64) || defined (_M_X64)
//amd64 linux
static inline void cpuid(uint32_t op, uint32_t *eax, uint32_t *ebx, uint32_t *ecx, uint32_t *edx){
  __asm__ __volatile__("cpuid":
          "=a" (*eax),
          "=b" (*ebx),
          "=c" (*ecx),
          "=d" (*edx) :
          "a" (op) :
          "cc");
}
#else
//i386 linux
static inline void cpuid(uint32_t op, uint32_t *eax, uint32_t *ebx, uint32_t *ecx, uint32_t *edx){
    __asm__ __volatile__("cpuid":
            "=a" (*eax),
            "=b" (*ebx),
            "=c" (*ecx),
            "=d" (*edx) :
            "0" (op),
            "1" (0),
            "2" (0));
    /*
    unsigned int tmp;
    __asm volatile
    ("push %%ebx\n\t"
     "cpuid\n\t"
     "mov %%ebx, (%1)\n\t"
     "pop %%ebx"
     : "=a" (*eax),
       "=S" (tmp),
       "=c" (*ecx),
       "=d" (*edx)
      : "0" (*eax));
    *ebx = tmp;
    */
}
#endif
#endif

static uint32_t CRC_TABLE_INITIALIZED = 0;
static uint32_t CRC_TABLE[1 << 8];
static int8_t CRC_SSE4_AVAILABLE = -1; // -1 means uninitialized

PYFAI_VISIBILITY_HIDDEN uint32_t _get_crc32_table_key(){
    return CRC_TABLE_INITIALIZED;
}

PYFAI_VISIBILITY_HIDDEN int8_t _is_crc32_sse4_available(){
    return CRC_SSE4_AVAILABLE;
}

PYFAI_VISIBILITY_HIDDEN void _get_crc32_table(uint32_t *table){
    for (int i=0; i<1<<8; i++)
        table[i] = CRC_TABLE[i];

}


PYFAI_VISIBILITY_HIDDEN void _crc32_table_init(uint32_t key){
    uint32_t i, j, a;

    for (i = 0; i < (1 << 8); i++){
        a = ((uint32_t) i) << 24;
        for (j = 0; j < 8; j++){
            if (a & 0x80000000)
                a = (a << 1) ^ key;
            else
                a = (a << 1);
        }
        CRC_TABLE[i] = a;
    }
    CRC_TABLE_INITIALIZED = key;
}


PYFAI_VISIBILITY_HIDDEN uint32_t _crc32_table(char *str, uint32_t len)
{
    uint32_t lcrc = ~0;
    char *p, *e;

    e = str + len;
    for (p = str; p < e; ++p)
        lcrc = (lcrc >> 8) ^ CRC_TABLE[(lcrc ^ (*p)) & 0xff];
    return ~lcrc;
}

#if defined(HWINFO_CPU_X86)

PYFAI_VISIBILITY_HIDDEN int8_t _check_sse4(){
    uint32_t eax, ebx, ecx, edx;
    cpuid(1, &eax, &ebx, &ecx, &edx);
    CRC_SSE4_AVAILABLE = (ecx & bit_SSE4_2)>0;
    return CRC_SSE4_AVAILABLE;
}

PYFAI_VISIBILITY_HIDDEN uint32_t _crc32_sse4(char *str, uint32_t len)
{
    uint32_t q = len / sizeof(uint32_t);
    uint32_t r = len % sizeof(uint32_t);
    uint32_t crc = 0;
    uint32_t *p = (uint32_t*) str;

    while (q--)
    {
#if defined(_MSC_VER)
        crc = _mm_crc32_u32(crc, *p);
#else
        __asm__ __volatile__(
                ".byte 0xf2, 0xf, 0x38, 0xf1, 0xf1;"
                :"=S"(crc)
                :"0"(crc), "c"(*p) );
#endif
        p++;
    }
    str = (char*) p;
    while (r--)
    {
#if defined(_MSC_VER)
        crc = _mm_crc32_u8(crc, *str);
#else
        __asm__ __volatile__(
                ".byte 0xf2, 0xf, 0x38, 0xf0, 0xf1"
                :"=S"(crc) :"0"(crc), "c"(*str));
#endif
        str++;
    }

    return crc;
}

PYFAI_VISIBILITY_HIDDEN uint32_t pyFAI_crc32(char *str, uint32_t len) {

    if (CRC_SSE4_AVAILABLE<0)
        _check_sse4();

  if (CRC_SSE4_AVAILABLE){
        return _crc32_sse4(str, len);
    }
    else
    {
        if (!CRC_TABLE_INITIALIZED)
        {
            _crc32_table_init((uint32_t) 0x1EDC6F41u);
        }
        return _crc32_table(str, len);
    }
}

#else
//Non x86 version
PYFAI_VISIBILITY_HIDDEN int8_t _check_sse4(){
  return 0;
}

PYFAI_VISIBILITY_HIDDEN uint32_t pyFAI_crc32(char *str, uint32_t len) {
    if (!CRC_TABLE_INITIALIZED){
        _crc32_table_init((uint32_t) 0x1EDC6F41u);
    }
    return _crc32_table(str, len);
}

PYFAI_VISIBILITY_HIDDEN uint32_t _crc32_sse4(char *str, uint32_t len) {
    return pyFAI_crc32(str, len);
}
#endif
