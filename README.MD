# compile
make

# encode
./c63enc -w 352 -h 288 -o tmp/test.c63 video/FOREMAN_352x288_30_orig_01.yuv
./c63enc -w 1920 -h 1080 -o tmp/test.c63 video/

# decode
./c63dec tmp/test.c63  tmp/test.yuv

# play the raw yuv file
vlc --rawvid-width 352 --rawvid-height 288 --rawvid-fps 30 --rawvid-chroma I420 tmp/test.yuv

首先根据给的word文档试运行了代码，一开始发现编译不过去，然后修改Makefile文件，把ldflag放到了最后。
发现用Makefile-new编译的话，用c63dec解码后视频会变模糊和绿屏，去掉-DC63_PRED就不会了。

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
