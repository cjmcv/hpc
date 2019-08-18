#include "operator/kernel_interface.h"
#include "executor.h"

__global__ void DotKernel(const int len, const float *vec_a, const float *vec_b, float *res) {
  // Prevents memory access across the border.
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < len;
    i += blockDim.x * gridDim.x) {
    //__shared__ float smem[BLOCK_SIZE];
    extern __shared__ float smem[]; // Dynamic allocation.
    smem[threadIdx.x] = vec_a[i] * vec_b[i];
    __syncthreads();

    // Summarize in blocks.
    // Limiting conditions: len should be a multiple of block_size.
    int count = blockDim.x / 2;
    while (count >= 1) {
      if (threadIdx.x < count) {
        smem[threadIdx.x] += smem[count + threadIdx.x];
      }
      // Synchronize the threads within the block,
      // then go to next round together.
      __syncthreads();
      count /= 2;
    }

    if (threadIdx.x == 0)
      atomicAdd(res, smem[0]);
  }
}

cux::KernelInterface *DotProductGPUPlugin() {
  using namespace cux;

  auto get_config = [&](int len) -> Config1D {
    Config1D config;
    config.threads_per_block = 1024;
    config.blocks_per_grid = (len + config.threads_per_block - 1) / config.threads_per_block;
    config.shared_memory_size = config.threads_per_block * sizeof(float);
    return config;
  };

  auto func = [&](Config1D config, int len, const void *vec_a, const void *vec_b, void *res) -> void {
    DotKernel << <config.blocks_per_grid,
      config.threads_per_block,
      config.shared_memory_size >> >
      (len, (float *)vec_a, (float *)vec_b, (float *)res);
  };

  DotGpuKernelIF *kernel = new DotGpuKernelIF();
  kernel->type_flag = TypeFlag::FLOAT32;
  kernel->describe_info = "Plugin example: It's the same as kernel V0";
  kernel->get_config = get_config;
  kernel->func = func;
  kernel->config_kernel = DotKernel;

  return kernel;
}