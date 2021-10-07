# vky
A small computing framework based on vulkan. 

This framework is designed to help you quickly call vulkan's computing API to do the calculations you need.

## 主要功能

该框架旨在帮助你快速调用vulkan的计算API来完成所需的计算，简化vulkan计算的使用过程。

## 编译前

* 自行安装好显卡驱动；
* 自行安装vulkan sdk；
* 修改vs2015-build-sample.bat中VULKAN_SDK安装路径，如：set VULKAN_SDK=D:/software/VulkanSDK/1.2.189.2/;

## 编译

* 双击vs2015-build-sample.bat，生成VS2015工程；
* 进入windows文件夹，进入main_vky.sln编译；

## 相关资料

### Compute shaders

Vulkan is a new generation graphics and compute API that provides high-efficiency, cross-platform access to modern GPUs used in a wide variety of devices from PCs and consoles to mobile phones and embedded platforms.

Compute shaders in Vulkan have first class support in the API. Compute shaders give applications the ability to perform non-graphics related tasks on the GPU.

* [vulkan - Main Page](https://www.khronos.org/vulkan/)
* [vulkan - Reference guide](https://www.khronos.org/files/vulkan11-reference-guide.pdf)
* [vulkan - Tutorial](https://vulkan-tutorial.com/Drawing_a_triangle/Graphics_pipeline_basics/Shader_modules)
* [vulkan - Examples](https://github.com/SaschaWillems/Vulkan/tree/master/examples)

---
