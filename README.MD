
## 目录
#### - [更新内容](#update)
#### - [优化说明](#optimization)
- [SSE 指令简单说明](#sse)
- [sad\_block\_8x8 函数的优化v1](#optimization1)
- [sad\_block\_8x8 函数的优化v2.3](#optimization2)
- [quantize\_block & dequantize\_block 函数的优化](#optimization3)
- [scale\_block 函数的优化](#optimization4)
- [原版代码 BUG 解决](#bug)
#### - [优化结果](#result)
#### - [执行指令](#shell)
#### - [日志记录（碎碎念）](#log)



<h2 id="update">更新内容</h2>

<h4>2018/12/17</h4>

推了三处优化。1080p 的处理速度在我的电脑上提高了10秒左右。努力写说明中。

<h4>2018/12/15   13:26</h4>

更新有关 c63enc.c 存在的 bug 说明，优化了 sad\_block\_8x8 (v2.3) 并写了优化说明。正在写优化结果对比。

<h4>2018/12/14</h4>

找到并修正了 c63enc.c 存在的 bug（会导致 common.c 中数组访问越界），写了相关说明

<h4>2018/12/13</h4>

初次提交，优化了 sad\_block\_8x8 (v1)，写了相关 md 说明




<h2 id="optimization">优化说明</h2>

<h3 id="sse">SSE 指令简单说明</h3>

根据上课讲的，SSE 指令集有共有八个128位的寄存器 xmm0 ~ xmm7，因此一个寄存器可以存放 4 个 32 位的数据（如 int, float），并利用指令同时进行一个寄存器（四个 32 位数据）的运算。例如：

```c
#include <stdio.h>
#include <x86intrin.h>

int main() 
{
    float a[4] = {0,1,2,3};
    float b[4] = {4,5,6,7};
    float c[4];
    
    __m128 xmm0 = _mm_load_ps(a); // 载入指针 a 所指向的四个数据到寄存器
    /* xmm0:
    * R0 = a[0]
    * R1 = a[1]
    * R2 = a[2]
    * R3 = a[3]
    */
    __m128 xmm1 = _mm_load_ps(b); // 载入指针 b 所指向的四个数据到寄存器
    /* xmm1:
    * R0 = b[0]
    * R1 = b[1]
    * R2 = b[2]
    * R3 = b[3]
    */
    __m128 xmm2 = _mm_add_ps(xmm0,xmm1); // 将 xmm0 和 xmm1 中的四个数据分别相加
    /* xmm2 xmm0 xmm1
    * R0  =  R0 + R0 = a[0]+b[0]
    * R1  =  R1 + R1 = a[1]+b[1]
    * R2  =  R2 + R2 = a[2]+b[2]
    * R3  =  R3 + R4 = a[3]+b[3]
    */
    _mm_store_ps(c, xmm2); // 将结果写回指针 c 指向的内存
    /* xmm2:
    * c[0] = R0
    * c[1] = R1
    * c[2] = R2
    * c[3] = R3
    */
    
    printf("%.2f, %.2f, %.2f, %.2f", c[0], c[1], c[2], c[3]);
    //output: 4.00, 6.00, 8.00, 10.00	
    return 0;
}
```

<h3 id="optimization1">sad_block_8x8 函数的优化v1</h3>

c63\_motion\_estimate -> me\_block\_8x8 -> me\_block\_8x8 -> sad\_block\_8x8

首先根据查看各个主要步骤的执行时间，发现最耗时的操作为 c63\_motion\_estimate()。跟进运行步骤，查找到其中最主要的步骤是 sad\_block\_8x8()，它为最内层的循环，执行了非常多的次数，所以我选择它作为最初的优化目标。它的原版代码如下：

```c
// 原始代码
void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
{
    *result = 0;
    int u,v;
    for (v=0; v<8; ++v) {
        for (u=0; u<8; ++u)
            *result += abs(block2[v*stride+u] - block1[v*stride+u]);
    }
}
```

内层的 u 循环将两个 block 中的八个连续数据相减取绝对值并累加。因此最先想到的是，可以利用 SSE 的思想将最内层的数据打包一起运算，u = 0, 1, 2, 3 为一组运算，u = 4, 5, 6, 7 为一组运算。但在实际操作过程中发现，[SSE 指令简单说明](#sse) 处的代码是将四个 32 位的数据打包，但是这段代码里由于图像处理的特殊情况，每个数据都是 uint8\_t 类型，即 8 位，而且 u 层循环一共处理八个数据，即 8\*8=64 位。SSE 指令集在批量存取和计算的数据为 16 字节（128位）对齐的时候效率才是最高的，也没有针对每个单元为 1 字节的数组特定的存取办法，因此数据的存取方法、运算方法等是这里要解决的关键问题。

**涉及指令介绍**

```c
// 将 64 位的数据 a 填充到寄存器的低 64 位，高 64 位置零
__m128i _mm_cvtsi64_si128(int64_t a)
/**
* R0 = a
* R1 = 0
*/

// 计算两个 128 位寄存器中 16 个 8 位无符号整数的绝对值差，计算结果为两个 16 位无符号整数，分别写入 128 位新寄存器的低 16 位的最低位和高 64 位的最低位寄存器。
__m128i _mm_sad_epu8(__m128i a, __m128i b)
/**
* r0 = abs(a0 - b0) + abs(a1 - b1) +...+ abs(a7 - b7)
* r1 = 0， r2 = 0， r3 = 0
* r4 = abs(a8 - b8) + abs(a9 - b9) +...+ abs(a15 - b15)
* r5 = 0， r6 = 0， r7 = 0
*/

// 将 128 位寄存器 a 的低 32 位取出
int _mm_cvtsi128_si32(__m128i xmm)
/**
* a = R0
*/
```

**修改后 sad\_block\_8x8 内部代码**

```c
// 改进后 v1
void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
{
    *result = 0;
    int v;
    for (v=0; v<8; ++v) {
        __m128i xmm2 = _mm_cvtsi64_si128(*(int64_t*)(block2+v*stride));
        __m128i xmm1 = _mm_cvtsi64_si128(*(int64_t*)(block1+v*stride));
        __m128i xmm3 = _mm_sad_epu8(xmm2, xmm1);
        *result += _mm_cvtsi128_si32(xmm3);
    }
}
```

首先用 int64\_t 类型的指针指向 block 数据块的当前位置。指针取出一块数据的长度与指针类型有关，因此 int64\_t 类型的指针能一次性取出 block 数据块中 64 位（8 字节）的数据，即一次 u 循环中所有的 8 个数据，避免用 int8\_t 类型多次取数据造成的性能浪费。然后利用 \_mm\_cvtsi64\_si128 将 block1 和 block2 中的 8bit\*8=64bit 数据分别载入 xmm1 和 xmm2 的低 64 位 ，并利用 \_mm\_sad\_epu8 将两个寄存器的每 8 位相减求绝对值之和写入 xmm3，最后用 \_mm\_cvtsi128\_si32 取出 xmm3 的低 32 位与 int 型的 result 累加。这样就去掉了内层的八次循环，并实现了八个数据的同时运算。

<h3 id="optimization2">sad_block_8x8 函数的优化v2.3</h3>

由于对 sad\_block\_8x8() 这个函数的再优化经历了两个失败版本所以取名为 v2.3。

首先是觉得128位的寄存器没有完全利用（只利用了低64位存放8个数据），想把它占满同时运算16个数据，失败；然后是想到循环展开，利用八个寄存器进行四倍的循环展开，也失败。在研究代码后发现取数据占用了相当大的时间，因此考虑尽量减少从寄存器取出数据的次数，从而减少时间开销。

具体代码如下：

```c
// 改进后 v2.3
void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
{
    *result = 0;
    int v;
    for (v=0; v<8; v+=4) {
        uint8_t *block1_cur = block1+v*stride;
        uint8_t *block2_cur = block2+v*stride;
        int pos = v*stride;
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
```

局限于7个 xmm 寄存器，最多进行四倍的循环展开，与此同时利用 SSE 的 add 指令，可以减少四分之三的数据取回。add 这个地方其实只需要用到寄存器的前32位，感觉还是存在一些浪费，但目前比较没辙。

<h3 id="optimization3">quantize_block & dequantize_block 函数的优化</h3>

quantize\_block 和 dequantize\_block 这两个函数的内部非常相似，因此这里仅对 quantize\_block 的优化进行说明。函数原始代码如下：

```c
// 原始代码
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
```

优化主要目标是优化最后的浮点除法。由于取用的 dct 的数据不连续，因此不能像之前一样批量取数据载入，这里使用 \_mm\_set\_ps 依次载入四个数据。同时，由于 SSE 指令似乎没有 c 中的 round 函数，因此数据读回后再使用 round 函数进行处理。

优化后代码如下：

```c
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
```

<h3 id="optimization4">scale_block 函数的优化</h3>

相比于其它函数，对 scale\_block 这个函数的优化手段要更加特殊一点点。首先看它的原始代码：

```c
// 原始代码
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
```

除了浮点乘是可考虑优化的内容，其中还有 a1 和 a2 的判断赋值。观察后可以发现，仅当 u = 0 的时候 a1 = ISQRT2，同时也仅当 v = 0 的时候 a2 = ISQRT2，其余时候它们都取 1 ，并没有什么进行乘法运算的必要。因此对于一组 in\_data 数据，其第 0 行第 0 个数据需要乘 a1 和 a2， 第 1~7 个数据只需要乘 a2；其余各行第 0 个数据需要乘 a1，别的数据都不用乘。区分这些数据，对部分数据不进行操作，因此改进后代码如下：

```c
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
```

心里觉得这个代码完全没前面的优美，可能还存在改进的余地。改进后暂时先放在这儿提供一些思路。

<h3 id="bug">原版代码 BUG 解决</h3>

**问题发现**

在初测试过程中发现了编码 1080p 的视频时，在 ubuntu on windows 上会报段错误，而在虚拟机的 ubuntu 18.04 上不会的问题。

跟进到代码里，发现是 common.c: dct\_quantize\_row() 中的

```c
block[i*8+j] = ((int16_t)in_data[i*w+j+x] - prediction[i*w+j+x]);
```

这一行出现问题。这样就很可能是发生了数组访问越界。查看代码：

```c
void dequantize_idct(int16_t *in_data, uint8_t *prediction, uint32_t width, uint32_t height,
			 uint8_t *out_data, uint8_t *quantization)
{
    int y;
    for (y=0; y<height; y+=8)
    {
        dequantize_idct_row(in_data+y*width, prediction+y*width, width, height, y, out_data+y*width, quantization);
    }
}

void dct_quantize_row(uint8_t *in_data, uint8_t *prediction, int w, int h,
        int16_t *out_data, uint8_t *quantization)
{
    int x;

    int16_t block[8*8];

    /* Perform the DCT and quantization */
    for(x = 0; x < w; x += 8)
    {
        int i,j;
        for (i=0; i<8; ++i)
            for (j=0; j<8; ++j)
                block[i*8+j] = ((int16_t)in_data[i*w+j+x] - prediction[i*w+j+x]);

        /* Store MBs linear in memory, i.e. the 64 coefficients are stored continous.
         * This allows us to ignore stride in DCT/iDCT and other functions. */
        dct_quant_block_8x8(block, out_data+(x*8), quantization);
    }
}
```

根据传参，1920\*1080的视频 width 为 1920，height 为1080，padw 为1920，padh 为1088。review 调用这里的函数，此处传参的 width，height，w，h 都是 padw 和 padh。

dct\_quantize\_row() 函数中 in\_data 的起始地址为 in\_data+y\*width， y 最大取值1080，因此 in\_data 的起始地址最大为 in\_data+1080\*1920。在接下来的调用里还要访问 in\_data[i\*w+j+x] 的地址，而 in\_data 作为 image->Y，在 c63enc.c 中可以看到 image->Y = malloc(width\*height)，它只有 width\*height=1920\*1080 个字节的长度。因此此处的访问必然导致越界。ubuntu on windows 的报错是没有问题的，反而虚拟机中 ubuntu 不报错比较奇怪。

然后打印了一下 log：（dct\_quantize\_row() 函数中 i\*w+j+x 可取的最大值为15359）

```c
void dct_quantize(uint8_t *in_data, uint8_t *prediction,
        uint32_t width, uint32_t height,
        int16_t *out_data, uint8_t *quantization)
{
    int y;
    for (y=0; y<height; y+=8)
    {
        printf("in_data: %ld, in_data+y*width+15359: %d\n", malloc_usable_size(in_data), y*width+15359);
        dct_quantize_row(in_data+y*width, prediction+y*width, width, height, out_data+y*width, quantization);
    }
}
```

在 ubuntu on windows 的输出为

```shell
...
in_data: 2076656, in_data+y*width+15359:2088959
Segmentation fault (core dumped)
qxy@qxy:/mnt/c/Users/saltyfish/Desktop/codec63-0.3/codec63$
```

在虚拟机的 ubuntu 的输出为

```shell
...
in_data: 2076656, in_data+y*width+15359:2073599
in_data: 2076656, in_data+y*width+15359:2088959
in_data: 2076656, in_data+y*width+15359:15359
...
```

同样地越界了，但是虚拟机 ubuntu 并不报错。在小组成员的共同测试下，只要是真正的 Linux 系统，无关 gcc 版本，都不会报错。推测可能是 ubuntu on windows 共享了 windows 的内存，导致两者内存中的内核空间和用户空间不太一样，因此只有 ubuntu on windows 访问到了会导致段错误的区域而真正的 Linux 系统虽越界但并没有段错误。

**问题解决**

参考 mjpeg\_encoder.c 里的算法修正了一下 c63enc.c，DCT 和量化中不再使用 padw 和 padh 而是使用原始的 width 和 height。具体修正如下：

```c
// c63enc.c

// 修改前
dct_quantize(image->Y, cm->curframe->predicted->Y, cm->padw[0], cm->padh[0], cm->curframe->residuals->Ydct, cm->quanttbl[0]);
dct_quantize(image->U, cm->curframe->predicted->U, cm->padw[1], cm->padh[1], cm->curframe->residuals->Udct, cm->quanttbl[1]);
dct_quantize(image->V, cm->curframe->predicted->V, cm->padw[2], cm->padh[2], cm->curframe->residuals->Vdct, cm->quanttbl[2]);

// 修改后
dct_quantize(image->Y, cm->curframe->predicted->Y, cm->width, cm->height, cm->curframe->residuals->Ydct, cm->quanttbl[0]);
dct_quantize(image->U, cm->curframe->predicted->U, cm->width*UX/YX, cm->height*UY/YY, cm->curframe->residuals->Udct, cm->quanttbl[1]);
dct_quantize(image->V, cm->curframe->predicted->V, cm->width*VX/YX, cm->height*VY/YY, cm->curframe->residuals->Vdct, cm->quanttbl[2]);
```

经测试后发现两个平台下都能跑通了。而且编码后的文件能被正常解码，解码后的视频能正常播放。



<h2 id="result">优化结果</h2>

|                              | foreman.yuv | tractor.yuv | PSNR |
| :--------------------------: | ----------- | ----------- | ---- |
|           原始代码           | 25.85s      | 1283.31s    |      |
|  sad_block_8x8 函数的优化v1  | 4.32s       | 209.87s     |      |
| sad_block_8x8 函数的优化v2.3 | 3.73s       | 180.13s     |      |



<h2 id="shell">执行指令</h2>

**compile**

```shell
$ make
```

**encode**

```shell
$ ./c63enc -w 352 -h 288 -o tmp/FOREMAN_352x288_30_orig_01.c63 video/FOREMAN_352x288_30_orig_01.yuv
$ ./c63enc -w 1920 -h 1080 -o tmp/1080p_tractor.c63 video/1080p_tractor.yuv
```

**decode**
```shell
$ ./c63dec tmp/FOREMAN_352x288_30_orig_01.c63  tmp/foreman.yuv
$ ./c63dec tmp/1080p_tractor.c63  tmp/tractor.yuv
```

**play the raw yuv file**

```shell
$ vlc --rawvid-width 352 --rawvid-height 288 --rawvid-fps 30 --rawvid-chroma I420 tmp/foreman.yuv
$ vlc --rawvid-width 1920 --rawvid-height 1080 --rawvid-fps 30 --rawvid-chroma I420 tmp/tractor.yuv
```



<h2 id="log">日志记录（碎碎念）</h2>

<h4>2018/12/12</h4>

首先根据给的word文档试运行了代码，一开始发现编译不过去，然后修改Makefile文件，把ldflag放到了最后。
发现用Makefile-new编译的话，用c63dec解码后视频会变模糊和绿屏，去掉-DC63\_PRED就不会了。

<h4>2018/12/13</h4>

最初尝试：从c63enc.c文件的main函数开始查看代码，main -> c63_encode_image -> dct\_quantize -> dct\_quantize\_row, 看到循环存在，尝试用 SSE 指令集优化最内层的8个j循环。想法就是一次性往128位的寄存器载入4个数据进行计算，但在做的途中发现图像的数组每一个元素都是一个字节(8位)而不是32位，正常做法无法直接载入数据。我把数据量类型强制转化为 float 尝试，但结果处理速度比原来慢不说，视频质量也非常非常的差。代码如下：
```c

mm_in_data = _mm_set_ps((float)in_data[i*w+0+x], (float)in_data[i*w+1+x], 
                        (float)in_data[i*w+2+x], (float)in_data[i*w+3+x]);
mm_prediction = _mm_set_ps((float)prediction[i*w+0+x], (float)prediction[i*w+1+x], 
                            (float)prediction[i*w+2+x], (float)prediction[i*w+3+x]);
mm_block = _mm_sub_ps(mm_in_data, mm_prediction);
_mm_store_ps(result, mm_block);
block[i*8+0] = (int16_t) result[0];
block[i*8+1] = (int16_t) result[1];
block[i*8+2] = (int16_t) result[2];
block[i*8+3] = (int16_t) result[3];

mm_in_data = _mm_set_ps((float)in_data[i*w+4+x], (float)in_data[i*w+5+x], 
                        (float)in_data[i*w+6+x], (float)in_data[i*w+7+x]);
mm_prediction = _mm_set_ps((float)prediction[i*w+4+x], (float)prediction[i*w+5+x], 
                            (float)prediction[i*w+6+x], (float)prediction[i*w+7+x]);
mm_block = _mm_sub_ps(mm_in_data, mm_prediction);
_mm_store_ps(result, mm_block);
block[i*8+4] = (int16_t) result[0];
block[i*8+5] = (int16_t) result[1];
block[i*8+6] = (int16_t) result[2];
block[i*8+7] = (int16_t) result[3];
```

然后在代码的各个部分加clock()函数查看各个计算耗费的时间，发现编码一个帧大约需要88000左右个时钟，而c63\_motion\_estimate这一步需要约84000个时钟。显然编码中最耗时的操作在这里，之前修改的dct\_quantize仅需要1300左右个时钟，对整体而言只占非常小一部分的时间。因此，就算在那边负优化了，也决定从运动估计最先下手。虽然还不知道运动估计是什么。

↑以上是在ubuntu虚拟机的工作。后来实在卡的受不了了。。转到windows下工作。。编译使用的是windows的ubuntu子系统。速度（时钟）有偏差，但依然是运动估计占大头。

从 c63\_motion\_estimate -> me\_block\_8x8 -> me\_block\_8x8 -> sad\_block\_8x8 下手。
写了个 test.c 测试一些想法的可行性，以及不同数据类型的取数据范围之类的小细节。

卧槽。。。好像速度有点不得了。。。但是 .c63 文件大了好多是怎么回事。。
视频可以播放，不知道有没有不清晰，现在眼有点花。休息了。

测试了一下tractor，发现段错误了，因此测试只能通过foreman。我自以为是修改的部分能达到和原来完全一样的效果，有点受打击，review代码找一下原因。

发现在win10的ubuntu子系统会段错误但是虚拟机的ubuntu不会。。心里苦啊。。睡觉睡觉

于12.14补充12.13的工作：

还是很在意为什么在 ubuntu for windows 下会段错误，review 了一下我改动的部分，发现有小 BUG：8 个 u 的循环层内部改了但是循环忘了去掉，空转了八次；result 赋值累加也可以再改的漂亮一点。但改完这个并没有对段错误产生任何效果。

于是换个想法，用原版文件换回我改动过的部分，测试是哪段代码改出了问题。但是找不到。这个时候我是坚信是我写出问题来了的，因为印象中这段代码跑通过 1080p（其实是在虚拟机的 ubuntu 跑通过），百思不得其解。最后使用最终手段，用完完全全原版的代码再跑 1080p，终于意识到是原版代码的问题了。然后尝试了虚拟机跑原版代码和我改后的代码。。。。。。。血和泪啊

<h4>2018/12/14</h4>

跟进到代码里，找到出错行后，发现按照这个代码的写法必然出现数组访问越界，反而是不报错的那方比较奇怪。参考 mjpeg\_encoder.c 里的算法修正了一下 c63enc.c，在测试后发现两个平台下都能跑通了。而且编码后的文件能被正常解码，解码后的视频能正常播放。具体问题和做法参考 [原版代码 BUG 解决](#bug)

此外还有个小小的不算问题的事儿，就是发现虚拟机的 ubuntu 比 windows 下的 ubuntu 运算要更快点，为什么呢。

在改进的 sad\_block\_8x8 基础上尝试了用两组数据填满一个128位的寄存器，使外层循环只需要循环一半的次数。尝试了 v2：先将八个占低64位数据扩充到16位填充到整个寄存器再通过 pack 将两个寄存器打包使一个128位寄存器中有16个8位数据；以及 v3：用 set 指令将四个 int 型数据填充到128位寄存器中，但效率都没有第一版优化高。另外似乎在 cflag 中使用 -msse4 时速度会变慢，不知何解。

然后想到了利用循环展开的思想。循环展开四倍，但是发现效果仍不理想，没有明显改进甚至有点变慢。

<h4>2018/12/15</h4>

在继续观察代码后，发现从128位寄存器取数据是个非常耗时间的操作（读数据不可避，而且想不到办法优化了），因此在循环展开四倍的基础上，将取回数据的操作只进行一次。经测试这样做之后编码时间又减少了不少。

<h4>2018/12/16</h4>

想不到 sad\_block\_8x8 还能怎么再优化了，所以回到 dct\_quantize 进行优化，因为除了 c63\_motion\_estimate，就数它和 dequantize\_idct 占用时间最久。本想着可能是因为第一次尝试时姿势不对，导致效率奇低完成负优化，于是又按照现在的想法对 dct\_quantize\_row 函数中的循环进行修改，修改如下：

```c
void dct_quantize_row(uint8_t *in_data, uint8_t *prediction, int w, int h,
        int16_t *out_data, uint8_t *quantization)
{
    int x;
    int16_t block[8*8];

    __m128i xmm_in_data, xmm_prediction, xmm_result;
    // __m128i zero = _mm_setzero_si128();

    /* Perform the DCT and quantization */
    for(x = 0; x < w; x += 8)
    {
        int i,j;
        for (i=0; i<8; ++i) {

            mm_in_data = _mm_cvtsi64_si128(*(int64_t*)(in_data+i*w+x));
            xmm_prediction = _mm_cvtsi64_si128(*(int64_t*)(prediction+i*w+x));
            xmm_result = _mm_sub_epi8(xmm_in_data, xmm_prediction);
            xmm_result = _mm_unpacklo_epi8(xmm_result, zero);
            int64_t p2 = _mm_cvtsi128_si64(xmm_result);
            *(int64_t*)(block+i*8) = p2;
            block[i*8+j]=1;

        }

        /* Store MBs linear in memory, i.e. the 64 coefficients are stored continous.
         * This allows us to ignore stride in DCT/iDCT and other functions. */
        dct_quant_block_8x8(block, out_data+(x*8), quantization);
    }
}
```

但结果依然是负优化的。想了想，会不会也是因为存取数据时间太长以至于完全掩盖了打包数据进行运算带来的时间优化，但这样就表明这段代码本身的时间占用就非常少。于是干脆用非常极端的方式，干脆不要做减法运算，而是把 block[i\*8+j] 每一个元素赋值为1，发现这样做后也并没有减少什么时间，证明了我的想法。然后尝试注解掉 dct\_quant\_block\_8x8(block, out\_data+(x\*8), quantization)，效果拔群。于是下一步决定放着上面的循环不动，去优化 dct\_quant\_block\_8x8。

进入到函数，首先看到 dct\_1d，尝试优化它。该函数原始代码如下：

```c
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
```

其中 dctlookup 为一个 float 类型的 8x8 矩阵硬编码在 tables.c 中。矩阵形如：

x<sub>11</sub>　x<sub>12</sub>　...　x<sub>17</sub>　x<sub>18</sub>

x<sub>21</sub>　x<sub>22</sub>　...　x<sub>27</sub>　x<sub>28</sub>

...

x<sub>81</sub>　x<sub>82</sub>　...　x<sub>87</sub>　x<sub>88</sub>

这个代码需要把 dctlookup 中第 j 列的八个元素乘以某个数并累加，赋值到 out\_data 的第 j 个元素。

利用 SSE 指令集，将 x<sub>11</sub>, x<sub>12</sub>, x<sub>13</sub>, x<sub>14</sub> 放入一个寄存器，x<sub>15</sub>, x<sub>16</sub>, x<sub>17</sub>, x<sub>18</sub> 放入另一个寄存器，与之相乘的元素均为 in\_data[1]，每轮寄存器数据分别累加，最后将两个寄存器内容分别取出并写回 out\_data 的对应位置。

按这样的思路直接改完发现变慢了，注解某些代码尝试感觉是累加操作写的不太合适。那边占用里很多时间。

<h4>2018/12/17</h4>

继续昨日的工作。发现昨日的 dct\_1d 中某个数组访问下标写错了，修正后顺便把整体代码修改了下，性能。。。肉眼不可见的优化，但也没有负优化，就放在那边了。理论上我只读取了 dctlookup 一次，应该大大减少了访存时间，但事实上并没有明显优化效果。可能是寄存器不足，虽然 load 了但数据还是被放到了什么堆区栈区内存区，导致它变慢。

继而优化了 quantize\_block 和 dequantize\_block。很神奇，我都没有抱着它们能变快的想法。。。结果竟然优化了一点点点。编码速度从平均180秒改进到了175秒左右。已经完全搞不懂什么时候能加速什么时候不能了。。。

重新打算推代码和文档，review 代码的时候又看了看前天看过的 scale\_block。揪心它很久了，之前做的时候不知为何效率上去了但视频质量下降了很多。肉眼可见的模糊。今天终于找到问题所在了！！！！！鼓掌！！！！！是因为有一组数据没写回。。它没改变我就忘了写回了。。于是速度优化到了170秒左右。