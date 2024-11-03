pushd shaders
md ../spv
popd

set GLSLANG_VALIDATIOR_PATH=D:\software\VulkanSDK\1.3.231.1\Bin

%GLSLANG_VALIDATIOR_PATH%\glslangValidator.exe -V ..\gemm_fp32_v1.comp -o ../spv/gemm_fp32_v1.spv
%GLSLANG_VALIDATIOR_PATH%\glslangValidator.exe -V ..\gemm_fp32_v2.comp -o ../spv/gemm_fp32_v2.spv
%GLSLANG_VALIDATIOR_PATH%\glslangValidator.exe -V ..\gemm_fp32_v3.comp -o ../spv/gemm_fp32_v3.spv
pause