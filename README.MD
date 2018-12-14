
### 目录
#### - [优化说明](#optimization)
- [SSE 指令简单说明](#sse)
- [sad_block_8x8 函数的优化](#optimization1)

#### - [执行指令](#shell)
#### - [日志记录（碎碎念）](#log)


<h3 id="optimization">优化说明</h3>

<h4 id="sse">SSE 指令简单说明</h4>

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

<h4 id="optimization1">sad_block_8x8 函数的优化</h4>

c63_motion_estimate -> me_block_8x8 -> me_block_8x8 -> sad_block_8x8



首先根据查看各个主要步骤的执行时间，发现最耗时的操作为 c63_motion_estimate()。跟进运行步骤，查找到其中最主要的步骤是 sad_block_8x8()，它为最内层的循环，执行了非常多的次数，所以我选择它作为最初的优化目标。它的原版代码如下：

```c
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

内层的 u 循环将两个 block 中的八个连续数据相减取绝对值并累加。因此最先想到的是，可以利用 SSE 的思想将最内层的数据打包一起运算，u = 0, 1, 2, 3 为一组运算，u = 4, 5, 6, 7 为一组运算。但在实际操作过程中发现，[SSE 指令简单说明](#sse) 处的代码是将四个 32 位的数据打包，但是这段代码里由于图像处理的特殊情况，每个数据都是 uint8_t 类型，即 8 位，而且 u 层循环一共处理八个数据，即 8*8=64 位。SSE 指令集在批量存取和计算的数据为 16 字节（128位）对齐的时候效率才是最高的，也没有针对每个单元为 1 字节的数组特定的存取办法，因此数据的存取方法、运算方法等是这里要解决的关键问题。

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

**修改后 sad_block_8x8 内部代码**

```c
void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
{
    *result = 0;
    int u,v;
    for (v=0; v<8; ++v) {
        int64_t *block2_8 = block2+v*stride;
        int64_t *block1_8 = block1+v*stride;
        __m128i xmm2 = _mm_cvtsi64_si128(*block2_8);
        __m128i xmm1 = _mm_cvtsi64_si128(*block1_8);
        __m128i xmm3 = _mm_sad_epu8(xmm2, xmm1);
        *result += _mm_cvtsi128_si32(xmm3);
    }
}
```

首先用 int64_t 类型的指针指向 block 数据块的当前位置。指针取出一块数据的长度与指针类型有关，因此 int64_t 类型的指针能一次性取出 block 数据块中 64 位（8 字节）的数据，即一次 u 循环中所有的 8 个数据，避免用 int8_t 类型多次取数据造成的性能浪费。然后利用 _mm_cvtsi64_si128 将 block1 和 block2 中的 8bit*8=64bit 数据分别载入 xmm1 和 xmm2 的低 64 位 ，并利用 _mm_sad_epu8 将两个寄存器的每 8 位相减求绝对值之和写入 xmm3，最后用 _mm_cvtsi128_si32 取出 xmm3 的低 32 位与 int 型的 result 累加。这样就去掉了内层的八次循环，并实现了八个数据的同时运算。



<h3 id="shell">执行指令</h3>

**compile**

```shell
$ make
```

**encode**

```shell
$ ./c63enc -w 352 -h 288 -o tmp/test.c63 video/FOREMAN_352x288_30_orig_01.yuv
$ ./c63enc -w 1920 -h 1080 -o tmp/test.c63 video/1080p_tractor.yuv
```

**decode**
```shell
$ ./c63dec tmp/test.c63  tmp/test.yuv
```

**play the raw yuv file**

```shell
$ vlc --rawvid-width 352 --rawvid-height 288 --rawvid-fps 30 --rawvid-chroma I420 tmp/test.yuv
```



<h3 id="log">日志记录（碎碎念）</h3>

<h4>2018/12/12</h4>

首先根据给的word文档试运行了代码，一开始发现编译不过去，然后修改Makefile文件，把ldflag放到了最后。
发现用Makefile-new编译的话，用c63dec解码后视频会变模糊和绿屏，去掉-DC63_PRED就不会了。

<h4>2018/12/13</h4>

最初尝试：从c63enc.c文件的main函数开始查看代码，main -> c63_encode_image -> dct_quantize -> dct_quantize_row, 看到循环存在，尝试用 SSE 指令集优化最内层的8个j循环。想法就是一次性往128位的寄存器载入4个数据进行计算，但在做的途中发现图像的数组每一个元素都是一个字节(8位)而不是32位，正常做法无法直接载入数据。我把数据量类型强制转化为 float 尝试，但结果处理速度比原来慢不说，视频质量也非常非常的差。代码如下：
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

然后在代码的各个部分加clock()函数查看各个计算耗费的时间，发现编码一个帧大约需要88000左右个时钟，而c63_motion_estimate这一步需要约84000个时钟。显然编码中最耗时的操作在这里，之前修改的dct_quantize仅需要1300左右个时钟，对整体而言只占非常小一部分的时间。因此，就算在那边负优化了，也决定从运动估计最先下手。虽然还不知道运动估计是什么。

↑以上是在ubuntu虚拟机的工作。后来实在卡的受不了了。。转到windows下工作。。编译使用的是windows的ubuntu子系统。速度（时钟）有偏差，但依然是运动估计占大头。

从 c63_motion_estimate -> me_block_8x8 -> me_block_8x8 -> sad_block_8x8 下手。
写了个 test.c 测试一些想法的可行性，以及不同数据类型的取数据范围之类的小细节。

卧槽。。。好像速度有点不得了。。。但是 .c63 文件打了好多是怎么回事。。
视频可以播放，不知道有没有不清晰，现在眼有点花。休息了。

测试了一下tractor，发现段错误了，因此测试只能通过foreman。我自以为是修改的部分能达到和原来完全一样的效果，有点受打击，review代码找一下原因。

发现在win10的ubuntu子系统会段错误但是虚拟机的ubuntu不会。。心里苦啊。。睡觉睡觉