#include <stdint.h>
//#include <smmintrin.h>
#include <cpuid.h>

int is_initialized=0;
uint32_t slowcrc_table[1<<8];

void slowcrc_init() {
	uint32_t i, j, a;

	for (i=0;i<(1<<8);i++) {
		a=((uint32_t)i)<<24;
		for (j=0;j<8;j++) {
			if (a&0x80000000)
				a=(a<<1)^0x11EDC6F41;
			else
				a=(a<<1);
		}
		slowcrc_table[i]=a;
	}
	is_initialized=1;
}

uint32_t slowcrc(char *str, uint32_t len) {
	uint32_t lcrc=~0;
	char *p, *e;

	e=str+len;
	for (p=str;p < e;++p)
		lcrc=(lcrc>>8)^slowcrc_table[(lcrc^(*p))&0xff];
	return ~lcrc;
}

uint32_t fastcrc(const char *str, uint32_t len) {
	uint32_t q=len/sizeof(uint32_t),
		     r=len%sizeof(uint32_t),
		     *p=(uint32_t*)str,
		     crc=0;

	while (q--) {
//		crc = _mm_crc32_u32(crc,*p);
		__asm__ __volatile__(
				".byte 0xf2, 0xf, 0x38, 0xf1, 0xf1;"
				:"=S"(crc)
				:"0"(crc), "c"(*p)
				);
		p++;
	}

	str=(char*)p;
	while (r--) {
//		crc = _mm_crc32_u8(crc,*str);
		__asm__ __volatile__(
				".byte 0xf2, 0xf, 0x38, 0xf0, 0xf1"
				:"=S"(crc)
				:"0"(crc), "c"(*str)
		);
		str++;
	}

	return crc;
}

uint32_t crc32(char *str, uint32_t len) {
  uint32_t eax, ebx, ecx, edx;
  __get_cpuid(1, &eax, &ebx, &ecx, &edx);

  if (ecx & bit_SSE4_2){
	    return fastcrc(str,len);
  	 }else{
  		 if (!is_initialized){
  			slowcrc_init();
  		 }
  		 return slowcrc(str,len);
  	 }
}
