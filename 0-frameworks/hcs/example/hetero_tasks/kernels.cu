#include <cuda_runtime.h>
#include "device_launch_parameters.h"

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

int CallGemmGPU(const int M, const int N,
                const int K, const float alpha,
                const float *A, const int lda,
                const float *B, const int ldb,
                const float beta,
                float *C, const int ldc, 
                cudaStream_t stream = nullptr) {

  const int block_size = 32;
  dim3 threads_per_block = dim3(block_size, block_size);
  dim3 blocks_per_grid = dim3(N / threads_per_block.x, M / threads_per_block.y);

  if (stream != nullptr) {
    GemmKernel << <blocks_per_grid, threads_per_block, 0, stream >> >
      (M, N, K, alpha, (float *)A, lda, (float *)B, ldb, beta, (float *)C, ldc);
  }
  else {
    GemmKernel << <blocks_per_grid, threads_per_block >> >
      (M, N, K, alpha, (float *)A, lda, (float *)B, ldb, beta, (float *)C, ldc);
  }
  
  return 0;
}