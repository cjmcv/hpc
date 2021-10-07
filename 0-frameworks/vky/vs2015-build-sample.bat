set VULKAN_SDK=D:/software/VulkanSDK/1.2.189.2/

%VULKAN_SDK%/Bin/glslangValidator.exe -V shaders/add.comp -o shaders/add.spv
%VULKAN_SDK%/Bin/glslangValidator.exe -V shaders/saxpy.comp -o shaders/saxpy.spv

mkdir windows
cd windows
cmake -G "Visual Studio 14 2015" -DCMAKE_GENERATOR_PLATFORM=x64  -DVULKAN_HEADERS_INSTALL_DIR:STRING=%VULKAN_SDK%/Include -DVULKAN_LIBS:STRING=%VULKAN_SDK%/Lib/vulkan-1.lib .. 

pause