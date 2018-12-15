#! /bin/bash

rm total_time.log
for i in {1..10};do
    ./c63enc -w 352 -h 288 -o tmp/FOREMAN_352x288_30_orig_01.c63 video/FOREMAN_352x288_30_orig_01.yuv
done

echo -e "\n" >> total_time.log

for i in {1..10};do
    ./c63enc -w 1920 -h 1080 -o tmp/1080p_tractor.c63 video/1080p_tractor.yuv
done
