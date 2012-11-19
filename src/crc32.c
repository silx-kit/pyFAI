#include <stdint.h>
#include <smmintrin.h>
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
	uint64_t q=len/sizeof(uint64_t),
		     r=len%sizeof(uint64_t),
		     *p=(uint64_t*)str,
		     crc64=0;
	uint32_t crc=0;

	while (q--) {
		crc64 = _mm_crc32_u64(crc64,*p);
		p++;
	}

	str=(char*)p;
	crc=crc64;
	while (r--) {
		crc = _mm_crc32_u8(crc,*str);
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
