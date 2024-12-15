#!/usr/bin/env bash

mkdir -p ./spv/

VULKANSDK_BIN=/home/cjmcv/VulkanSDK/1.3.296.0/x86_64/bin

$VULKANSDK_BIN/glslangValidator -V ../gemm_fp32_v1.comp -o ./spv/gemm_fp32_v1.spv
$VULKANSDK_BIN/glslangValidator -V ../gemm_fp32_v2.comp -o ./spv/gemm_fp32_v2.spv
$VULKANSDK_BIN/glslangValidator -V ../gemm_fp32_v3.comp -o ./spv/gemm_fp32_v3.spv