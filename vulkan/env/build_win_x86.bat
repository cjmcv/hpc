
:: x86 vulkan
:: https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/8.1.0/threads-posix/sjlj/x86_64-8.1.0-release-posix-sjlj-rt_v6-rev0.7z/download
:: https://vulkan.lunarg.com/sdk/home#windows

set MAIN_SRC=main_gemm.cpp
set VULKAN_SDK=D:/software/VulkanSDK/1.3.231.1/

mkdir build-win-x86
pushd build-win-x86
cmake -G "MinGW Makefiles" -DMAIN_SRC:STRING=%MAIN_SRC% -DVULKAN_WIN_SDK:STRING=%VULKAN_SDK% ..
mingw32-make -j8
popd