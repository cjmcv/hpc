#!/usr/bin/env bash

# Usage: bash build_and_run_android.sh

############################################
## android vulkan
ANDROID_NDK=/home/cjmcv/android-ndk-r27c/

# ##### android armv7 neon
# mkdir -p build-android-armv7
# pushd build-android-armv7
# cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a" -DANDROID_PLATFORM=android-24 ..
# make -j8
# # make install
# popd

# ##### android aarch64
mkdir -p build-android-aarch64
pushd build-android-aarch64
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-24 ..
make -j4
popd

# Run on android
adb shell "rm -r /data/local/tmp/pai/"

adb push ./bin/pai_vk /data/local/tmp/pai/bin/pai_vk
adb push ./spv /data/local/tmp/pai/spv

adb shell "chmod 777 -R /data/local/tmp/pai/ && cd /data/local/tmp/pai/ && ./bin/pai_vk"
