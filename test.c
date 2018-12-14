#include <stdio.h>
#include <inttypes.h>
#include <x86intrin.h>
void print128_num(__m128i var)
{
    uint16_t *val = (uint16_t*) &var;
    printf("Numerical: %i %i %i %i %i %i %i %i \n",
           val[0], val[1], val[2], val[3], val[4], val[5],
           val[6], val[7]);
}
int main()
{
    char a[] = {'a','b','c','d',
    'a','b','c','d',
    'a','b','c','d',
    'a','b','c','d',
    'a','b','c','d',
    'a','b','c','d',
    'a','b','c','d',
    'a','b','c','d'};
    int *p = a;
    __m128i xmm = _mm_cvtsi64_si128(*p);
    __m128i xmm2 = _mm_set_epi8(2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1);
    __m128i xmm3 = _mm_sad_epu8(xmm2, xmm);
    int *result;
    int p2 = _mm_cvtsi128_si64(xmm3);
    // int16_t *p3;
    result = &p2;
    printf("%ld",*p);
    // xmm = _mm_cvtepu8_epi16(xmm);
    // print128_num(xmm3);
    // printf("%c %c %c %c %c %c %c %c ",p3[0], p3[1], p3[2], p3[3], p3[4], p3[5], p3[6], p3[7]);
}
