#pragma once

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

namespace marlin {

// Marlin params

// 8个warp是一个不错的选择，因为每个SM都有4个调度器，每个调度器里有超过1个warp可以隐藏更多的延迟。
// 同时，我们希望相对较少的warp，使每个warp和小分块上有更多寄存器可用。
// 所以选择8个warp，则每个调度器有2个warp，即有32*8=256个线程一个block。
// 8 warps are a good choice since every SM has 4 schedulers and having more
// than 1 warp per schedule allows some more latency hiding. At the same time,
// we want relatively few warps to have many registers per warp and small tiles.
static constexpr int default_threads = 256;

static constexpr int pipe_stages =
    4;  // 4 pipeline stages fit into shared memory

static constexpr int min_thread_n = 64;
static constexpr int min_thread_k = 64;

static constexpr int tile_size = 16;
static constexpr int max_par = 16; // 并行计算64行A矩阵的最大并行数？

// Repack params
static constexpr int repack_stages = 8;

static constexpr int repack_threads = 256;

// tile_shape[16, 64]: 针对权重矩阵B进行分块
static constexpr int tile_k_size = tile_size;
static constexpr int tile_n_size = tile_k_size * 4;

// Helpers
template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) { return elems[i]; }
};

using I4 = Vec<int, 4>;

constexpr int div_ceil(int a, int b) { return (a + b - 1) / b; }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
// No support for async
#else

// cp.async.cg.shared.global 从全局内存异步加载到共享内存，cg表示只缓存数据到global level cache L2。
// .pred表示谓语，.reg .pred p 定义谓语寄存器p
// setp表示设置谓语，ne是不等于，b32表示32位整数，整句的意思是%0与0进行判断，不相等则设置谓语寄存器p为true
// 如果p为true，则执行cp操作
__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr,
                                      bool pred = true) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   .reg .pred p;\n"
      "   setp.ne.b32 p, %0, 0;\n"
      "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
      "}\n" ::"r"((int)pred),
      "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async4(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   cp.async.cg.shared.global [%0], [%1], %2;\n"
      "}\n" ::"r"(smem),
      "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int n>
__device__ inline void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

#endif

}  // namespace marlin
