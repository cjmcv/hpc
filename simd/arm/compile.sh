#!/usr/bin/env bash

# /home/shared_dir/android-ndk-r21e/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang++ -O3 -g gemm.cpp
/home/shared_dir/android-ndk-r21e/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang++ -O3 -g gemm_kernel.s gemm.cpp -o a.out 

# 汇编
# /home/shared_dir/android-ndk-r21e/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang++ -O3 -S -c gemm.cpp
# /home/shared_dir/android-ndk-r21e/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android-objdump -j .text -ld -C -S a.out > b.txt

# perf
# /home/shared_dir/android-ndk-r21e/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang++ -O3 perf.cpp