/*!
* \brief gemm: C = A * B.
*/
#include "operator/gemm.h"

namespace cux {
// CUDA version 1: 72 ms
// It is rewrited from GEMMHostV2. 
// bi,bj can be replaced by blockIdx.x,blockIdx.y
// i,j can be replaced by threadIdx.x,threadIdx.y
// so just bk and k left. Grid and block is related to the dst matrix.
//
// \ C[ty, tx] = A[ty, k] * B[k, tx]
// for bk -> bk_num_per_grid
//     for k -> k_num_per_block
//         C[bi*bs + ty, bj*bs + tx] = A[bi*bs + ty, bk*bs + k] * B[k*bs + k, bj*bs + tx]
__global__ void GEMMDeviceV0(const int M, const int N, 
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
    for (int k = 0;k < block_size; k++) {
      c_sub_acc += alpha * A[(blockIdx.y * block_size + threadIdx.y) * lda + (bk * block_size + k)] *
        B[(bk * block_size + k) * ldb + (blockIdx.x * block_size + threadIdx.x)];
    }
  }

  C[(blockIdx.y * block_size + threadIdx.y) * ldc + (blockIdx.x * block_size + threadIdx.x)] += c_sub_acc;
}

// CUDA version 1.
// Use shared memory.
__global__ void GEMMDeviceV1(const int M, const int N, 
                             const int K, const float alpha,
                             const float *A, const int lda,
                             const float *B, const int ldb,
                             const float beta,
                             float *C, const int ldc) {

  const int block_size = blockDim.x;  // Side length.
  // PS: If change them to extern __shared__ float a_shared[];
  //                       extern __shared__ float b_shared[];
  //     There will be errors in the calculation.
  //     Dynamic allocation of Shared memory supports only one array with one dimension?
  extern __shared__ float a_shared[];
  float *b_shared = &a_shared[blockDim.x * blockDim.x];
  //extern __shared__ float b_shared[];

  // y * xdim + x
  C[(blockIdx.y * blockDim.y + threadIdx.y)
    * (blockDim.x * gridDim.x)
    + (blockIdx.x * blockDim.x + threadIdx.x)] *= beta;
  //__syncthreads();

  float c_sub_acc = 0;
  // For blocks in grid.
  for (int bk = 0; bk < K / block_size; bk++) {
    a_shared[threadIdx.y * block_size + threadIdx.x] = A[(blockIdx.y * block_size + threadIdx.y) * lda + (bk * block_size + threadIdx.x)];
    b_shared[threadIdx.y * block_size + threadIdx.x] = B[(bk * block_size + threadIdx.y) * ldb + (blockIdx.x * block_size + threadIdx.x)];
    // Wait for data to complete loading to Shared memory.
    __syncthreads();

    // For elements in a block.
    for (int k = 0; k < block_size; k++) {
      c_sub_acc += alpha * a_shared[threadIdx.y * block_size + k] * b_shared[k * block_size + threadIdx.x];
    }
    // To prevent the case from happening:
    // The next round of data is loaded when the data in share memory is not used up.
    __syncthreads();
  }

  C[(blockIdx.y * block_size + threadIdx.y) * ldc + (blockIdx.x * block_size + threadIdx.x)] += c_sub_acc;
}

void GEMM::PrepareLaunchConfig(int N, int M) {
  // V0 & V1
  int kernel_id;
  { 
    Config2D config;
    const int block_size = 32;
    config.threads_per_block = dim3(block_size, block_size);
    config.blocks_per_grid = dim3(N / config.threads_per_block.x, M / config.threads_per_block.y);

    // V0
    {
      kernel_id = 0;
      config_2d_[kernel_id] = config;

      op_params_.launch_config->QueryPotentialOccupancy(
        GEMMDeviceV0, config.threads_per_block.x * config.threads_per_block.y, 0,
        gpu_kernel_active_blocks_[kernel_id], gpu_kernel_occupancys_[kernel_id]);
    }
    // V1
    {
      kernel_id = 1;
      config.shared_memory_size = 2 * config.threads_per_block.x * config.threads_per_block.y * sizeof(float);
      config_2d_[kernel_id] = config;

      op_params_.launch_config->QueryPotentialOccupancy(
        GEMMDeviceV1, config.threads_per_block.x * config.threads_per_block.y, config.shared_memory_size,
        gpu_kernel_active_blocks_[kernel_id], gpu_kernel_occupancys_[kernel_id]);
    }
  }
}

void GEMM::GEMMDevice(const int kernel_id, 
                      const int M, const int N, 
                      const int K, const float alpha,
                      const float *A, const int lda,
                      const float *B, const int ldb,
                      const float beta,
                      float *C, const int ldc) {
  switch (kernel_id) {
  case 0:
    GEMMDeviceV0 << <config_2d_[kernel_id].blocks_per_grid, 
                     config_2d_[kernel_id].threads_per_block >> >
      (M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    break;
  case 1:
    // For:  __shared__ float a_shared[threads_per_block.y][threads_per_block.x];
    //       __shared__ float b_shared[threads_per_block.y][threads_per_block.x];
    GEMMDeviceV1 << <config_2d_[kernel_id].blocks_per_grid, 
                     config_2d_[kernel_id].threads_per_block, 
                     config_2d_[kernel_id].shared_memory_size >> >
      (M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    break;
  case 2:
    cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N);
    break;
  default:
    CUXLOG_ERR("Device Kernel id (%d) not found.", kernel_id);
  }
}

} // namespace cux