#!/usr/bin/env bash

# /home/cjmcv/android-ndk-r21e/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang++ -O3 -g gemm.cpp
/home/cjmcv/android-ndk-r21e/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang++ -O3 -g -I../../ gemm_kernel.s gemm.cpp -o a.out 

# 汇编
# /home/cjmcv/android-ndk-r21e/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang++ -O3 -S -c gemm.cpp
# /home/cjmcv/android-ndk-r21e/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android-objdump -j .text -ld -C -S a.out > b.txt

# perf
# /home/cjmcv/android-ndk-r21e/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang++ -O3 peak_perf_detector.cpp

adb push a.out /data/local/tmp/gemm/a.out
#adb push libc++_shared.so /data/local/tmp/gemm/libc++_shared.so
adb shell "chmod 777 -R ./data/local/tmp/gemm/ && export LD_LIBRARY_PATH=/data/local/tmp/gemm/:$LD_LIBRARY_PATH && ./data/local/tmp/gemm/a.out"