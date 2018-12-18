#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <xmmintrin.h>

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

// static void transpose_4x4(float *in_data, float *out_data)
// {
//     __m128 row0 = _mm_load_ps(in_data);
//     __m128 row1 = _mm_load_ps(in_data+8);
//     __m128 row2 = _mm_load_ps(in_data+16);
//     __m128 row3 = _mm_load_ps(in_data+24);

//     _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

//     _mm_store_ps(out_data, row0);
// 	_mm_store_ps(out_data + 8, row1);
// 	_mm_store_ps(out_data + 16, row2);
// 	_mm_store_ps(out_data + 24, row3);
// }

//  改进后 v1：负优化
// static void transpose_block(float *in_data, float *out_data)
// {
//     transpose_4x4(in_data, out_data);
//     transpose_4x4(in_data+4, out_data+32);
//     transpose_4x4(in_data+32, out_data+4);
//     transpose_4x4(in_data+36, out_data+36);
// }

// 原始代码
// static void dct_1d(float *in_data, float *out_data)
// {
//     int i,j;

//     for (j=0; j<8; ++j)
//     {
//         float dct = 0;

//         for (i=0; i<8; ++i)
//         {
//             dct += in_data[i] * dctlookup[i][j];
//         }

//         out_data[j] = dct;
//     }
// }

//  改进后 v1：效果肉眼不可见，但好像没有负优化
static void dct_1d(float *in_data, float *out_data)
{
    int j;
    __m128 row0_03 = _mm_load_ps(dctlookup[0]);
    __m128 row1_03 = _mm_load_ps(dctlookup[1]);
    __m128 row2_03 = _mm_load_ps(dctlookup[2]);
    __m128 row3_03 = _mm_load_ps(dctlookup[3]);
    __m128 row4_03 = _mm_load_ps(dctlookup[4]);
    __m128 row5_03 = _mm_load_ps(dctlookup[5]);
    __m128 row6_03 = _mm_load_ps(dctlookup[6]);
    __m128 row7_03 = _mm_load_ps(dctlookup[7]);

    __m128 row0_47 = _mm_load_ps(dctlookup[0]+4);
    __m128 row1_47 = _mm_load_ps(dctlookup[1]+4);
    __m128 row2_47 = _mm_load_ps(dctlookup[2]+4);
    __m128 row3_47 = _mm_load_ps(dctlookup[3]+4);
    __m128 row4_47 = _mm_load_ps(dctlookup[4]+4);
    __m128 row5_47 = _mm_load_ps(dctlookup[5]+4);
    __m128 row6_47 = _mm_load_ps(dctlookup[6]+4);
    __m128 row7_47 = _mm_load_ps(dctlookup[7]+4);

    for (j=0; j<8; ++j)
    {
        __m128 in_data_0 = _mm_set1_ps(in_data[8*j]);
        __m128 in_data_1 = _mm_set1_ps(in_data[1+8*j]);
        __m128 in_data_2 = _mm_set1_ps(in_data[2+8*j]);
        __m128 in_data_3 = _mm_set1_ps(in_data[3+8*j]);
        __m128 in_data_4 = _mm_set1_ps(in_data[4+8*j]);
        __m128 in_data_5 = _mm_set1_ps(in_data[5+8*j]);
        __m128 in_data_6 = _mm_set1_ps(in_data[6+8*j]);
        __m128 in_data_7 = _mm_set1_ps(in_data[7+8*j]);
        __m128 out_data_03 = _mm_add_ps(
            _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(row0_03, in_data_0), _mm_mul_ps(row1_03, in_data_1)), 
                _mm_add_ps(_mm_mul_ps(row2_03, in_data_2), _mm_mul_ps(row3_03, in_data_3))
            ), 
            _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(row4_03, in_data_4), _mm_mul_ps(row5_03, in_data_5)), 
                _mm_add_ps(_mm_mul_ps(row6_03, in_data_6), _mm_mul_ps(row7_03, in_data_7))
            ));
        __m128 out_data_47 = _mm_add_ps(
            _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(row0_47, in_data_0), _mm_mul_ps(row1_47, in_data_1)), 
                _mm_add_ps(_mm_mul_ps(row2_47, in_data_2), _mm_mul_ps(row3_47, in_data_3))
            ), 
            _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(row4_47, in_data_4), _mm_mul_ps(row5_47, in_data_5)), 
                _mm_add_ps(_mm_mul_ps(row6_47, in_data_6), _mm_mul_ps(row7_47, in_data_7))
            ));
        _mm_store_ps(out_data+8*j, out_data_03);
        _mm_store_ps(out_data+8*j+4, out_data_47);
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

// 原始代码
// static void scale_block(float *in_data, float *out_data)
// {
//     int u,v;

//     for (v=0; v<8; ++v)
//     {
//         for (u=0; u<8; ++u)
//         {
//             float a1 = !u ? ISQRT2 : 1.0f;
//             float a2 = !v ? ISQRT2 : 1.0f;

//             /* Scale according to normalizing function */
//             out_data[v*8+u] = in_data[v*8+u] * a1 * a2;
//         }
//     }
// }

// 改进后 v1：有优化，好像比 zigzag那两个作用大一点
static void scale_block(float *in_data, float *out_data)
{
    int v;
    __m128 in_data_03, in_data_47;
    __m128 xmm_a1_0, xmm_a2_0;
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
        in_data_03 = _mm_load_ps(in_data+v*8);
        xmm_out_data47 = _mm_load_ps(in_data+v*8+4);
        xmm_out_data03 = _mm_mul_ps(xmm_a1_0, in_data_03);
        _mm_store_ps(out_data+v*8, xmm_out_data03);
        _mm_store_ps(out_data+v*8+4, xmm_out_data47);
    }
}

// 原始代码
// static void quantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
// {
//     int zigzag;
//     for (zigzag=0; zigzag < 64; ++zigzag)
//     {
//         uint8_t u = zigzag_U[zigzag];
//         uint8_t v = zigzag_V[zigzag];

//         float dct = in_data[v*8+u];

//         /* Zig-zag and quantize */
//         out_data[zigzag] = round((dct / 4.0) / quant_tbl[zigzag]);
//     }
// }

// 改进后 v1：有一点点优化
static void quantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
{
    int zigzag;
    for (zigzag=0; zigzag < 64; zigzag+=4)
    {
        uint8_t u0 = zigzag_U[zigzag];
        uint8_t v0 = zigzag_V[zigzag];
        uint8_t u1 = zigzag_U[zigzag+1];
        uint8_t v1 = zigzag_V[zigzag+1];
        uint8_t u2 = zigzag_U[zigzag+2];
        uint8_t v2 = zigzag_V[zigzag+2];
        uint8_t u3 = zigzag_U[zigzag+3];
        uint8_t v3 = zigzag_V[zigzag+3];

        // float dct = in_data[v*8+u];
        __m128 xmm_dct = _mm_set_ps(in_data[v3*8+u3], in_data[v2*8+u2], in_data[v1*8+u1], in_data[v0*8+u0]);
        __m128 xmm_4 = _mm_set1_ps(4.0);
        __m128 xmm_quant = _mm_set_ps(quant_tbl[zigzag+3], quant_tbl[zigzag+2], quant_tbl[zigzag+1], quant_tbl[zigzag]);
        /* Zig-zag and quantize */
        __m128 xmm = _mm_div_ps(_mm_div_ps(xmm_dct, xmm_4), xmm_quant);
        _mm_store_ps(out_data+zigzag, xmm);
        out_data[zigzag] = round(out_data[zigzag]);
        out_data[zigzag+1] = round(out_data[zigzag+1]);
        out_data[zigzag+2] = round(out_data[zigzag+2]);
        out_data[zigzag+3] = round(out_data[zigzag+3]);
    }
}

// 原始代码
// static void dequantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
// {
//     int zigzag;
//     for (zigzag=0; zigzag < 64; ++zigzag)
//     {
//         uint8_t u = zigzag_U[zigzag];
//         uint8_t v = zigzag_V[zigzag];

//         float dct = in_data[zigzag];

//         /* Zig-zag and de-quantize */
//         out_data[v*8+u] = round((dct * quant_tbl[zigzag]) / 4.0);
//     }
// }

// 改进后 v1：有一点点优化
static void dequantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
{
    int zigzag;
    for (zigzag=0; zigzag < 64; zigzag+=4)
    {
        uint8_t u0 = zigzag_U[zigzag];
        uint8_t v0 = zigzag_V[zigzag];
        uint8_t u1 = zigzag_U[zigzag+1];
        uint8_t v1 = zigzag_V[zigzag+1];
        uint8_t u2 = zigzag_U[zigzag+2];
        uint8_t v2 = zigzag_V[zigzag+2];
        uint8_t u3 = zigzag_U[zigzag+3];
        uint8_t v3 = zigzag_V[zigzag+3];

        // float dct = in_data[v*8+u];
        __m128 xmm_dct = _mm_set_ps(in_data[zigzag+3], in_data[zigzag+2], in_data[zigzag+1], in_data[zigzag]);
        __m128 xmm_4 = _mm_set1_ps(4.0);
        __m128 xmm_quant = _mm_set_ps(quant_tbl[zigzag+3], quant_tbl[zigzag+2], quant_tbl[zigzag+1], quant_tbl[zigzag]);
        /* Zig-zag and quantize */
        __m128 xmm = _mm_div_ps(_mm_mul_ps(xmm_dct, xmm_quant), xmm_4);
        float p[4] __attribute((aligned(16)));
        _mm_store_ps(p, xmm);
        out_data[v0*8+u0] = round(p[0]);
        out_data[v1*8+u1] = round(p[1]);
        out_data[v2*8+u2] = round(p[2]);
        out_data[v3*8+u3] = round(p[3]);
    }
}

void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data, uint8_t *quant_tbl)
{
    _mm_prefetch((char*)in_data, _MM_HINT_T0);
    float mb[8*8] __attribute((aligned(16)));
    float mb2[8*8] __attribute((aligned(16)));

    int i, v;

    for (i=0; i<64; ++i)
        mb2[i] = in_data[i];

    // for (v=0; v<8; ++v)
    // {
    //     dct_1d(mb2+v*8, mb+v*8);
    // }
dct_1d(mb2, mb);
    transpose_block(mb, mb2);

    // for (v=0; v<8; ++v)
    // {
    //     dct_1d(mb2+v*8, mb+v*8);
    // }
dct_1d(mb2, mb);
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

// // 原始代码
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
//     __m128i res = _mm_setzero_si128();
//     for (v=0; v<8; ++v) {
//         __m128i xmm2 = _mm_cvtsi64_si128(*(int64_t*)(block2+v*stride));
//         __m128i xmm1 = _mm_cvtsi64_si128(*(int64_t*)(block1+v*stride));
//         __m128i xmm3 = _mm_sad_epu8(xmm2, xmm1);
//         res = _mm_add_epi32(res, xmm3);
        
//     }
//     *result = _mm_cvtsi128_si32(res);
// }

// 改进后 v2.4
// void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
// {
//     *result = 0;
//     int v;
//     __m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;

//     for (v=0; v<8; v+=4) {
//         uint8_t *block1_cur = block1+v*stride;
//         uint8_t *block2_cur = block2+v*stride;

//         xmm0 = _mm_cvtsi64_si128(*(int64_t*)(block1_cur));
//         xmm1 = _mm_cvtsi64_si128(*(int64_t*)(block2_cur));
//         xmm1 = _mm_sad_epu8(xmm1, xmm0);
//         // *result += _mm_cvtsi128_si32(xmm1);

//         block1_cur += stride;
//         block2_cur += stride;
//         xmm2 = _mm_cvtsi64_si128(*(int64_t*)(block1_cur));
//         xmm3 = _mm_cvtsi64_si128(*(int64_t*)(block2_cur));
//         xmm3 = _mm_sad_epu8(xmm3, xmm2);

//         xmm1 = _mm_add_epi32(xmm1, xmm3);
//         // *result += _mm_cvtsi128_si32(xmm3);

//         block1_cur += stride;
//         block2_cur += stride;
//         xmm3 = _mm_cvtsi64_si128(*(int64_t*)(block1_cur));
//         xmm4 = _mm_cvtsi64_si128(*(int64_t*)(block2_cur));
//         xmm4 = _mm_sad_epu8(xmm4, xmm3);
//         // *result += _mm_cvtsi128_si32(xmm5);

//         block1_cur += stride;
//         block2_cur += stride;
        
//         xmm5 = _mm_cvtsi64_si128(*(int64_t*)(block1_cur));
//         xmm6 = _mm_cvtsi64_si128(*(int64_t*)(block2_cur));
//         xmm6 = _mm_sad_epu8(xmm6, xmm5);

//         // 累加，一次性加回result，即只读回一次
        
//         xmm6 = _mm_add_epi32(xmm5, xmm6);
//         xmm5 = xmm7;
//         xmm7 = _mm_add_epi32(xmm1, xmm6);
//         // if (!v) {
//         //     xmm7 = _mm_add_epi32(xmm1, xmm5);
//         // } else {
//         //     xmm5 = _mm_add_epi32(xmm1, xmm5);
//         // }
        
//     }
//     xmm7 = _mm_add_epi32(xmm7, xmm5);
//     *result += _mm_cvtsi128_si32(xmm7);
// }

// 改进后 v2.5
void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
{
    // *result = 0;
    int v;
    __m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;

    // block1：8x8块的第0，1行
    xmm0 = _mm_cvtsi64_si128(*(int64_t*)(block1));
    xmm1 = _mm_cvtsi64_si128(*(int64_t*)(block1+stride));
    // 打包到一个寄存器
    xmm0 = _mm_unpacklo_epi64(xmm0, xmm1);
    // block2：8x8块的第0，1行
    xmm1 = _mm_cvtsi64_si128(*(int64_t*)(block2));
    xmm2 = _mm_cvtsi64_si128(*(int64_t*)(block2+stride));
    // 打包到一个寄存器
    xmm1 = _mm_unpacklo_epi64(xmm1, xmm2);
    xmm0 = _mm_sad_epu8(xmm1, xmm0);

    // block1：8x8块的第2，3行
    xmm1 = _mm_cvtsi64_si128(*(int64_t*)(block1+2*stride));
    xmm2 = _mm_cvtsi64_si128(*(int64_t*)(block1+3*stride));
    xmm1 = _mm_unpacklo_epi64(xmm1, xmm2);
    // block2：8x8块的第2，3行
    xmm2 = _mm_cvtsi64_si128(*(int64_t*)(block2+2*stride));
    xmm3 = _mm_cvtsi64_si128(*(int64_t*)(block2+3*stride));
    xmm2 = _mm_unpacklo_epi64(xmm2, xmm3);
    xmm1 = _mm_sad_epu8(xmm2, xmm1);

    xmm0 = _mm_add_epi32(xmm0, xmm1);

    // block1：8x8块的第4，5行
    xmm4 = _mm_cvtsi64_si128(*(int64_t*)(block1+4*stride));
    xmm5 = _mm_cvtsi64_si128(*(int64_t*)(block1+5*stride));
    xmm4 = _mm_unpacklo_epi64(xmm4, xmm5);
    // block2：8x8块的第4，5行
    xmm5 = _mm_cvtsi64_si128(*(int64_t*)(block2+4*stride));
    xmm6 = _mm_cvtsi64_si128(*(int64_t*)(block2+5*stride));
    xmm5 = _mm_unpacklo_epi64(xmm5, xmm6);
    xmm4 = _mm_sad_epu8(xmm5, xmm4);

    // block1：8x8块的第6，7行
    xmm5 = _mm_cvtsi64_si128(*(int64_t*)(block1+6*stride));
    xmm6 = _mm_cvtsi64_si128(*(int64_t*)(block1+7*stride));
    xmm5 = _mm_unpacklo_epi64(xmm5, xmm6);
    // block2：8x8块的第6，7行
    xmm6 = _mm_cvtsi64_si128(*(int64_t*)(block2+6*stride));
    xmm7 = _mm_cvtsi64_si128(*(int64_t*)(block2+7*stride));
    xmm6 = _mm_unpacklo_epi64(xmm6, xmm7);
    xmm5 = _mm_sad_epu8(xmm6, xmm5);
    
    xmm4 = _mm_add_epi32(xmm4, xmm5);

    xmm0 = _mm_add_epi32(xmm0, xmm4);
    xmm1 = _mm_shuffle_epi32(xmm0, _MM_SHUFFLE(1,1,1,2));
    xmm0 = _mm_add_epi32(xmm0, xmm1);

    *result = _mm_cvtsi128_si32(xmm0);
}

// 改进后 v2.3
// void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
// {
//     *result = 0;
//     int v;
//     __m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;

//     for (v=0; v<8; v+=4) {
//         uint8_t *block1_cur = block1+v*stride;
//         uint8_t *block2_cur = block2+v*stride;

//         xmm0 = _mm_cvtsi64_si128(*(int64_t*)(block1_cur));
//         xmm1 = _mm_cvtsi64_si128(*(int64_t*)(block2_cur));
//         xmm1 = _mm_sad_epu8(xmm1, xmm0);
//         // *result += _mm_cvtsi128_si32(xmm1);

//         block1_cur += stride;
//         block2_cur += stride;
//         xmm2 = _mm_cvtsi64_si128(*(int64_t*)(block1_cur));
//         xmm3 = _mm_cvtsi64_si128(*(int64_t*)(block2_cur));
//         xmm3 = _mm_sad_epu8(xmm3, xmm2);
//         // *result += _mm_cvtsi128_si32(xmm3);

//         block1_cur += stride;
//         block2_cur += stride;
//         xmm4 = _mm_cvtsi64_si128(*(int64_t*)(block1_cur));
//         xmm5 = _mm_cvtsi64_si128(*(int64_t*)(block2_cur));
//         xmm5 = _mm_sad_epu8(xmm5, xmm4);
//         // *result += _mm_cvtsi128_si32(xmm5);

//         block1_cur += stride;
//         block2_cur += stride;
//         xmm6 = _mm_cvtsi64_si128(*(int64_t*)(block1_cur));
//         xmm7 = _mm_cvtsi64_si128(*(int64_t*)(block2_cur));
//         xmm7 = _mm_sad_epu8(xmm7, xmm6);

//         // 累加，一次性加回result，即只读回一次
//         xmm1 = _mm_add_epi32(xmm1, xmm3);
//         xmm5 = _mm_add_epi32(xmm5, xmm7);
//         xmm7 = _mm_add_epi32(xmm1, xmm5);
//         *result += _mm_cvtsi128_si32(xmm7);
//     }
// }

// 改进后 v2.3-展开版
// void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
// {
//     *result = 0;
//     int v;
//     __m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
//     __m128i xmm01, xmm11, xmm21, xmm31, xmm41, xmm51, xmm61, xmm71;

//     // for (v=0; v<8; v+=4) {
//         uint8_t *block1_cur = block1;
//         uint8_t *block2_cur = block2;

//         xmm0 = _mm_cvtsi64_si128(*(int64_t*)(block1_cur));
//         xmm1 = _mm_cvtsi64_si128(*(int64_t*)(block2_cur));
//         xmm1 = _mm_sad_epu8(xmm1, xmm0);
//         // *result += _mm_cvtsi128_si32(xmm1);

//         block1_cur += stride;
//         block2_cur += stride;
//         xmm2 = _mm_cvtsi64_si128(*(int64_t*)(block1_cur));
//         xmm3 = _mm_cvtsi64_si128(*(int64_t*)(block2_cur));
//         xmm3 = _mm_sad_epu8(xmm3, xmm2);
//         // *result += _mm_cvtsi128_si32(xmm3);

//         block1_cur += stride;
//         block2_cur += stride;
//         xmm4 = _mm_cvtsi64_si128(*(int64_t*)(block1_cur));
//         xmm5 = _mm_cvtsi64_si128(*(int64_t*)(block2_cur));
//         xmm5 = _mm_sad_epu8(xmm5, xmm4);
//         // *result += _mm_cvtsi128_si32(xmm5);

//         block1_cur += stride;
//         block2_cur += stride;
//         xmm6 = _mm_cvtsi64_si128(*(int64_t*)(block1_cur));
//         xmm7 = _mm_cvtsi64_si128(*(int64_t*)(block2_cur));
//         xmm7 = _mm_sad_epu8(xmm7, xmm6);

//         // 累加，一次性加回result，即只读回一次
//         xmm1 = _mm_add_epi32(xmm1, xmm3);
//         xmm5 = _mm_add_epi32(xmm5, xmm7);
//         xmm7 = _mm_add_epi32(xmm1, xmm5);



//         block1_cur += stride;
//         block2_cur += stride;

//         xmm01 = _mm_cvtsi64_si128(*(int64_t*)(block1_cur));
//         xmm11 = _mm_cvtsi64_si128(*(int64_t*)(block2_cur));
//         xmm11 = _mm_sad_epu8(xmm11, xmm01);
//         // *result += _mm_cvtsi128_si32(xmm1);

//         block1_cur += stride;
//         block2_cur += stride;
//         xmm21 = _mm_cvtsi64_si128(*(int64_t*)(block1_cur));
//         xmm31 = _mm_cvtsi64_si128(*(int64_t*)(block2_cur));
//         xmm31 = _mm_sad_epu8(xmm31, xmm21);
//         // *result += _mm_cvtsi128_si32(xmm3);

//         block1_cur += stride;
//         block2_cur += stride;
//         xmm41 = _mm_cvtsi64_si128(*(int64_t*)(block1_cur));
//         xmm51 = _mm_cvtsi64_si128(*(int64_t*)(block2_cur));
//         xmm51 = _mm_sad_epu8(xmm51, xmm41);
//         // *result += _mm_cvtsi128_si32(xmm5);

//         block1_cur += stride;
//         block2_cur += stride;
//         xmm61 = _mm_cvtsi64_si128(*(int64_t*)(block1_cur));
//         xmm71 = _mm_cvtsi64_si128(*(int64_t*)(block2_cur));
//         xmm71 = _mm_sad_epu8(xmm71, xmm61);

//         // 累加，一次性加回result，即只读回一次
//         xmm11 = _mm_add_epi32(xmm11, xmm31);
//         xmm51 = _mm_add_epi32(xmm51, xmm71);
//         xmm71 = _mm_add_epi32(xmm11, xmm51);
//         xmm7 = _mm_add_epi32(xmm7, xmm71);
//         *result = _mm_cvtsi128_si32(xmm7);
//     // }
// }

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

