#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <x86intrin.h>

#include "tables.h"

#define ISQRT2 0.70710678118654f

static void transpose_block(float *in_data, float *out_data)
{
    int i,j;
    for (i=0; i<8; ++i)
        for (j=0; j<8; ++j)
        {
            out_data[i*8+j] = in_data[j*8+i];
        }
}

static void dct_1d(float *in_data, float *out_data)
{
    int i,j;

    for (j=0; j<8; ++j)
    {
        float dct = 0;

        for (i=0; i<8; ++i)
        {
            dct += in_data[i] * dctlookup[i][j];
        }

        out_data[j] = dct;
    }
}

static void idct_1d(float *in_data, float *out_data)
{
    int i,j;

    for (j=0; j<8; ++j)
    {
        float idct = 0;

        for (i=0; i<8; ++i)
        {
            idct += in_data[i] * dctlookup[j][i];
        }

        out_data[j] = idct;
    }
}


static void scale_block(float *in_data, float *out_data)
{
    int u,v;

    for (v=0; v<8; ++v)
    {
        for (u=0; u<8; ++u)
        {
            float a1 = !u ? ISQRT2 : 1.0f;
            float a2 = !v ? ISQRT2 : 1.0f;

            /* Scale according to normalizing function */
            out_data[v*8+u] = in_data[v*8+u] * a1 * a2;
        }
    }
}

static void quantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
{
    int zigzag;
    for (zigzag=0; zigzag < 64; ++zigzag)
    {
        uint8_t u = zigzag_U[zigzag];
        uint8_t v = zigzag_V[zigzag];

        float dct = in_data[v*8+u];

        /* Zig-zag and quantize */
        out_data[zigzag] = round((dct / 4.0) / quant_tbl[zigzag]);
    }
}

static void dequantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
{
    int zigzag;
    for (zigzag=0; zigzag < 64; ++zigzag)
    {
        uint8_t u = zigzag_U[zigzag];
        uint8_t v = zigzag_V[zigzag];

        float dct = in_data[zigzag];

        /* Zig-zag and de-quantize */
        out_data[v*8+u] = round((dct * quant_tbl[zigzag]) / 4.0);
    }
}

void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data, uint8_t *quant_tbl)
{
    float mb[8*8] __attribute((aligned(16)));
    float mb2[8*8] __attribute((aligned(16)));

    int i, v;

    for (i=0; i<64; ++i)
        mb2[i] = in_data[i];

    for (v=0; v<8; ++v)
    {
        dct_1d(mb2+v*8, mb+v*8);
    }

    transpose_block(mb, mb2);

    for (v=0; v<8; ++v)
    {
        dct_1d(mb2+v*8, mb+v*8);
    }

    transpose_block(mb, mb2);
    scale_block(mb2, mb);
    quantize_block(mb, mb2, quant_tbl);

    for (i=0; i<64; ++i)
        out_data[i] = mb2[i];
}


void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data, uint8_t *quant_tbl)
{
    float mb[8*8] __attribute((aligned(16)));
    float mb2[8*8] __attribute((aligned(16)));

    int i, v;

    for (i=0; i<64; ++i)
        mb[i] = in_data[i];

    dequantize_block(mb, mb2, quant_tbl);

    scale_block(mb2, mb);

    for (v=0; v<8; ++v)
    {
        idct_1d(mb+v*8, mb2+v*8);
    }

    transpose_block(mb2, mb);

    for (v=0; v<8; ++v)
    {
        idct_1d(mb+v*8, mb2+v*8);
    }

    transpose_block(mb2, mb);

    for (i=0; i<64; ++i)
        out_data[i] = mb[i];
}

// 原始代码
// void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
// {
//     *result = 0;

//     int u,v;
//     for (v=0; v<8; ++v)
//         for (u=0; u<8; ++u)
//             *result += abs(block2[v*stride+u] - block1[v*stride+u]);
// }

// 改进后 v1
// void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
// {
//     *result = 0;
//     int v;
//     for (v=0; v<8; ++v) {
//         __m128i xmm2 = _mm_cvtsi64_si128(*(int64_t*)(block2+v*stride));
//         __m128i xmm1 = _mm_cvtsi64_si128(*(int64_t*)(block1+v*stride));
//         __m128i xmm3 = _mm_sad_epu8(xmm2, xmm1);
//         *result += _mm_cvtsi128_si32(xmm3);
//     }
// }

// 改进后 v2.3
void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
{
    *result = 0;
    int v;
    for (v=0; v<8; v+=4) {
        uint8_t *block1_cur = block1+v*stride;
        uint8_t *block2_cur = block2+v*stride;

        __m128i xmm0 = _mm_cvtsi64_si128(*(int64_t*)(block1_cur));
        __m128i xmm1 = _mm_cvtsi64_si128(*(int64_t*)(block2_cur));
        xmm1 = _mm_sad_epu8(xmm1, xmm0);
        // *result += _mm_cvtsi128_si32(xmm1);

        block1_cur += stride;
        block2_cur += stride;
        __m128i xmm2 = _mm_cvtsi64_si128(*(int64_t*)(block1_cur));
        __m128i xmm3 = _mm_cvtsi64_si128(*(int64_t*)(block2_cur));
        xmm3 = _mm_sad_epu8(xmm3, xmm2);
        // *result += _mm_cvtsi128_si32(xmm3);

        block1_cur += stride;
        block2_cur += stride;
        __m128i xmm4 = _mm_cvtsi64_si128(*(int64_t*)(block1_cur));
        __m128i xmm5 = _mm_cvtsi64_si128(*(int64_t*)(block2_cur));
        xmm5 = _mm_sad_epu8(xmm5, xmm4);
        // *result += _mm_cvtsi128_si32(xmm5);

        block1_cur += stride;
        block2_cur += stride;
        __m128i xmm6 = _mm_cvtsi64_si128(*(int64_t*)(block1_cur));
        __m128i xmm7 = _mm_cvtsi64_si128(*(int64_t*)(block2_cur));
        xmm7 = _mm_sad_epu8(xmm7, xmm6);

        // 累加，一次性加回result，即只读回一次
        xmm1 = _mm_add_epi32(xmm1, xmm3);
        xmm5 = _mm_add_epi32(xmm5, xmm7);
        xmm7 = _mm_add_epi32(xmm1, xmm5);
        *result += _mm_cvtsi128_si32(xmm7);
    }
}

//  改进后 v2 ← 可能因为操作变多了，失败。弃。
// void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
// {
//     *result = 0;
//     int v;
//     for (v=0; v<8; v+=2) {
//         int64_t *block2_8 = block2+v*stride;
//         int64_t *block1_8 = block1+v*stride;

//         // 填充各寄存器的低64位
//         __m128i xmm1 = _mm_cvtsi64_si128(*block1_8);
//         __m128i xmm2 = _mm_cvtsi64_si128(*block2_8);
//         __m128i xmm3 = _mm_cvtsi64_si128(*(block1+(v+1)*stride));
//         __m128i xmm4 = _mm_cvtsi64_si128(*(block2+(v+1)*stride));

//         // 扩充每个数据从8位到16位，高位写0，8个数据占满整个寄存器
//         xmm1 = _mm_cvtepi8_epi16(xmm1);
//         xmm2 = _mm_cvtepi8_epi16(xmm2);
//         xmm3 = _mm_cvtepi8_epi16(xmm3);
//         xmm4 = _mm_cvtepi8_epi16(xmm4);

//         // 打包，将xmm1 xmm3，xmm2 xmm4中的共16个数据作为8位一个数据分别填充到xmm5 xmm6中
//         // 此时，xmm5和xmm6中都有16个数据，完全利用整个寄存器
//         __m128i xmm5 = _mm_packus_epi16(xmm1, xmm3);
//         __m128i xmm6 = _mm_packus_epi16(xmm2, xmm4);
        
//         // 求差取绝对值后求和，作为16位数据写入r0, r4
//         __m128i xmm7 = _mm_sad_epu8(xmm6, xmm5);

//         // 取出数据
//         uint16_t *val = (uint16_t*) &xmm7;
//         *result = *result + val[0] + val[4];
//     }
// }

// 改进后 v2.1 ← 也变慢了，弃
// void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
// {
//     *result = 0;
//     int v;
//     for (v=0; v<8; v+=4) {
//         int pos = v*stride;
//         __m128i xmm0 = _mm_set_epi32(*(int*)(block1+pos+stride+4), *(int*)(block1+pos+stride), *(int*)(block1+pos+4), *(int*)(block1+pos));
//         __m128i xmm1 = _mm_set_epi32(*(int*)(block2+pos+stride+4), *(int*)(block2+pos+stride), *(int*)(block2+pos+4), *(int*)(block2+pos));
//         // 求差取绝对值后求和，作为16位数据写入r0, r4
//         __m128i xmm2 = _mm_sad_epu8(xmm1, xmm0);

//         __m128i xmm3 = _mm_set_epi32(*(int*)(block1+pos+3*stride+4), *(int*)(block1+pos+3*stride), *(int*)(block1+pos+2*stride+4), *(int*)(block1+2*stride+pos));
//         __m128i xmm4 = _mm_set_epi32(*(int*)(block2+pos+3*stride+4), *(int*)(block2+pos+3*stride), *(int*)(block2+pos+2*stride+4), *(int*)(block2+2*stride+pos));
//         // 求差取绝对值后求和，作为16位数据写入r0, r4
//         __m128i xmm5 = _mm_sad_epu8(xmm4, xmm3);


//         __m128i xmm6 = _mm_add_epi32(xmm2, xmm5);

//         // 取出数据
//         int val[4];
//         // _mm_store_si128(val, xmm6);
//         // *result = *result + val[0] + val[2];
//     }
// }

// 改进后 v2.2 ← 循环展开 也慢
// void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
// {
//     *result = 0;
//     int v;
//     for (v=0; v<8; v+=2) {
//         uint8_t *block2_8 = block2+v*stride;
//         uint8_t *block1_8 = block1+v*stride;

//         __m128i xmm1 = _mm_cvtsi64_si128(*(int64_t*)block1_8);
//         __m128i xmm2 = _mm_cvtsi64_si128(*(int64_t*)block2_8);
//         __m128i xmm3 = _mm_sad_epu8(xmm2, xmm1);
//         *result += _mm_cvtsi128_si32(xmm3);

//         __m128i xmm4 = _mm_cvtsi64_si128(*(int64_t*)(block1+stride));
//         __m128i xmm5 = _mm_cvtsi64_si128(*(int64_t*)(block2+stride));
//         __m128i xmm6 = _mm_sad_epu8(xmm5, xmm4);
//         *result += _mm_cvtsi128_si32(xmm6);
//     }
// }

