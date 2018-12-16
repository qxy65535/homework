#include <stdio.h>
#include <inttypes.h>
#include <x86intrin.h>
void print128_num(__m128i var)
{
    uint8_t *val = (uint8_t*) &var;
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
    int64_t *p = a;
    __m128i xmm = _mm_cvtsi64_si128(*p);
    __m128i xmm2 = _mm_set_epi8(2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1);
    __m128i xmm3 = _mm_sad_epu8(xmm2, xmm);
    int *result;
    // *result = _mm_cvtsi128_si64(xmm);
    int p2 = _mm_cvtsi128_si64(xmm);
    // int16_t *p3;
    result = &p2;
    printf("%d\n",result[0]);
    // xmm = _mm_cvtepu8_epi16(xmm);
    print128_num(xmm);
    // printf("%c %c %c %c %c %c %c %c ",p3[0], p3[1], p3[2], p3[3], p3[4], p3[5], p3[6], p3[7]);
}

void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
{
    *result = 0;
    int v;
    for (v=0; v<8; v+=4) {
        uint8_t *block1_cur = block1+v*stride;
        uint8_t *block2_cur = block2+v*stride;

        __m128i xmm0 = _mm_cvtsi64_si128(*(int64_t*)block1_cur);
        __m128i xmm1 = _mm_cvtsi64_si128(*(int64_t*)block2_cur);
        xmm1 = _mm_sad_epu8(xmm1, xmm0);
        *result += _mm_cvtsi128_si32(xmm1);

        block1_cur += stride;
        block2_cur += stride;
        __m128i xmm2 = _mm_cvtsi64_si128(*(int64_t*)(block1_cur));
        __m128i xmm3 = _mm_cvtsi64_si128(*(int64_t*)(block2_cur));
        xmm3 = _mm_sad_epu8(xmm3, xmm2);
        *result += _mm_cvtsi128_si32(xmm3);

        block1_cur += stride;
        block2_cur += stride;
        __m128i xmm4 = _mm_cvtsi64_si128(*(int64_t*)(block1_cur));
        __m128i xmm5 = _mm_cvtsi64_si128(*(int64_t*)(block2_cur));
        xmm5 = _mm_sad_epu8(xmm5, xmm4);
        *result += _mm_cvtsi128_si32(xmm5);

        block1_cur += stride;
        block2_cur += stride;
        __m128i xmm6 = _mm_cvtsi64_si128(*(int64_t*)(block1_cur));
        __m128i xmm7 = _mm_cvtsi64_si128(*(int64_t*)(block2_cur));
        xmm7 = _mm_sad_epu8(xmm7, xmm6);
        *result += _mm_cvtsi128_si32(xmm7);
    }
}

static void scale_block(float *in_data, float *out_data)
{
    int u,v;
    __m128 in_data_03, in_data_47;
    __m128 xmm_a1_0, xmm_a1_1, xmm_a2_0, xmm_a2_1;
    __m128 xmm_out_data03, xmm_out_data47;

    xmm_a1_0 = _mm_set_ps(1.0f, 1.0f, 1.0f, ISQRT2);
    // xmm_a1_1 = xmm_a2_1 = _mm_set1_ps(1.0f);
    xmm_a2_0 = _mm_set1_ps(ISQRT2);

            in_data_03 = _mm_load_ps(in_data);
            in_data_47 = _mm_load_ps(in_data+4);

            xmm_out_data03 = _mm_mul_ps(xmm_a1_0, in_data_03);
            xmm_out_data03 = _mm_mul_ps(xmm_a2_0, xmm_out_data03);
            // xmm_out_data47 = _mm_mul_ps(xmm_a1_1, in_data_47);
            xmm_out_data47 = _mm_mul_ps(xmm_a2_0, in_data_47);

            /* Scale according to normalizing function */
            _mm_store_ps(out_data, xmm_out_data03);
            _mm_store_ps(out_data+4, xmm_out_data47);
    for (v=1; v<8; ++v)
    {

            // __m128 xmm_a2 = !v ? xmm_a2_0 : xmm_a1_1;

            in_data_03 = _mm_load_ps(in_data+v*8);
            in_data_47 = _mm_load_ps(in_data+v*8+4);

            xmm_out_data03 = _mm_mul_ps(xmm_a1_0, in_data_03);
            // xmm_out_data03 = _mm_mul_ps(xmm_a2, xmm_out_data03);
            // xmm_out_data47 = _mm_mul_ps(xmm_a1_1, in_data_47);
            // xmm_out_data47 = _mm_mul_ps(xmm_a2, in_data_47);

            /* Scale according to normalizing function */
            _mm_store_ps(out_data+v*8, xmm_out_data03);
            _mm_store_ps(out_data+v*8+4, xmm_out_data47);
            // out_data[v*8+u] = in_data[v*8+u] * a1 * a2;

    }
}