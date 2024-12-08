#!/usr/bin/env bash

# 注意工具链版本太低，会显示不支持armv8.2的指令，如 error: use of undeclared identifier 'vdotq_s32' 
# always_inline function 'vdotq_s32' requires target feature 'dotprod', but would be inlined into function 'main' that is compiled without support for 'dotprod'
# 需要加-march=armv8.4-a

/home/cjmcv/android-ndk-r27c/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android24-clang++ -O3 -g -march=armv8.4-a -I../../ gemm_int8.cpp
# /home/cjmcv/android-ndk-r27c/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android24-clang++ -O3 -g -I../../ gemm_fp32_ext.s gemm_fp32.cpp -o a.out 
# /home/cjmcv/android-ndk-r27c/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android24-clang++ -O3 -I../../ matrix_transpose.cpp 

# 汇编
# /home/cjmcv/android-ndk-r27c/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android24-clang++ -O3 -S -c -I../../ gemm_fp32.cpp
# /home/cjmcv/android-ndk-r27c/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android-objdump -j .text -ld -C -S a.out > b.txt

# perf
# /home/cjmcv/android-ndk-r27c/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android24-clang++ -O3 peak_perf_detector.cpp

adb push a.out /data/local/tmp/gemm/a.out
#adb push libc++_shared.so /data/local/tmp/gemm/libc++_shared.so
adb shell "chmod 777 -R ./data/local/tmp/gemm/ && export LD_LIBRARY_PATH=/data/local/tmp/gemm/:$LD_LIBRARY_PATH && ./data/local/tmp/gemm/a.out"