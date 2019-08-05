#include "operator/kernel_interface.h"
#include "executor.h"

__global__ void GemmKernel(const int M, const int N,
                           const int K, const float alpha,
                           const float *A, const int lda,
                           const float *B, const int ldb,
                           const float beta,
                           float *C, const int ldc) {

  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int xdim = blockDim.x * gridDim.x;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  C[y * xdim + x] *= beta;
  //__syncthreads();

  const int block_size = blockDim.x; // The Block is square.
  const int block_num_K = K / block_size;

  float c_sub_acc = 0;
  for (int bk = 0; bk < block_num_K; bk++) {
    for (int k = 0; k < block_size; k++) {
      c_sub_acc += alpha * A[(blockIdx.y * block_size + threadIdx.y) * lda + (bk * block_size + k)] *
        B[(bk * block_size + k) * ldb + (blockIdx.x * block_size + threadIdx.x)];
    }
  }

  C[(blockIdx.y * block_size + threadIdx.y) * ldc + (blockIdx.x * block_size + threadIdx.x)] += c_sub_acc;
}

cux::KernelInterface *GemmGPUPlugin() {
  using namespace cux;

  auto get_config = [&](int M, int N) -> Config2D {
    Config2D config;
    const int block_size = 32;
    config.threads_per_block = dim3(block_size, block_size);
    config.blocks_per_grid = dim3(N / config.threads_per_block.x, M / config.threads_per_block.y);
    return config;
  };
  auto func = [&](Config2D config,
    const int M, const int N,
    const int K, const float alpha,
    const void *A, const int lda,
    const void *B, const int ldb,
    const float beta,
    void *C, const int ldc) -> void {
    GemmKernel << <config.blocks_per_grid, config.threads_per_block >> >
      (M, N, K, alpha, (float *)A, lda, (float *)B, ldb, beta, (float *)C, ldc);
  };

  GemmGpuKernelIF *kernel = new GemmGpuKernelIF();
  kernel->type_flag = TypeFlag::FLOAT32;
  kernel->describe_info = "Plugin example: It's the same as kernel V0";
  kernel->get_config = get_config;
  kernel->func = func;
  kernel->kernel_address = GemmKernel;

  return kernel;
}