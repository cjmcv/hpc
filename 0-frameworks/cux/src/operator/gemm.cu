/*!
* \brief gemm: C = A * B.
*/
#include "operator/gemm.h"

namespace cux {
// CUDA version 0: 2.15 ms
// It is rewrited from GemmHostV2. 
// bi,bj can be replaced by blockIdx.x,blockIdx.y
// i,j can be replaced by threadIdx.x,threadIdx.y
// so just bk and k left. Grid and block is related to the dst matrix.
//
// \ C[ty, tx] = A[ty, k] * B[k, tx]
// for bk -> bk_num_per_grid
//     for k -> k_num_per_block
//         C[bi*bs + ty, bj*bs + tx] = A[bi*bs + ty, bk*bs + k] * B[k*bs + k, bj*bs + tx]
__global__ void GemmDeviceV0(const int M, const int N, 
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

  float c_sub_acc = 0; // Only works on the current thread.
  for (int bk = 0; bk < K / block_size; bk++) {
    for (int k = 0;k < block_size; k++) {
      c_sub_acc += alpha * A[(blockIdx.y * block_size + threadIdx.y) * lda + (bk * block_size + k)] *
        B[(bk * block_size + k) * ldb + (blockIdx.x * block_size + threadIdx.x)];
    }
  }
  //// The same.
  //for (int k = 0; k < block_size; k++) {
  //  c_sub_acc += alpha * A[(blockIdx.y * block_size + threadIdx.y) * lda + (bk * block_size + k)] *
  //    B[(bk * block_size + k) * ldb + (blockIdx.x * block_size + threadIdx.x)];
  //}

  C[(blockIdx.y * block_size + threadIdx.y) * ldc + (blockIdx.x * block_size + threadIdx.x)] += c_sub_acc;
}

// CUDA version 1: 1.33 ms
// Use 1D shared memory.
__global__ void GemmDeviceV1(const int M, const int N, 
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
#pragma unroll
    for (int k = 0; k < block_size; k++) {
      c_sub_acc += alpha * a_shared[threadIdx.y * block_size + k] * b_shared[k * block_size + threadIdx.x];
    }
    // To prevent the case from happening:
    // The next round of data is loaded when the data in share memory is not used up.
    __syncthreads();
  }

  C[(blockIdx.y * block_size + threadIdx.y) * ldc + (blockIdx.x * block_size + threadIdx.x)] += c_sub_acc;
}

// CUDA version 2: 0.95 ms
// Use 2D shared memory.
template <int BLOCK_SIZE> __global__ void GemmDeviceV2(const int M, const int N,
                                                       const int K, const float alpha,
                                                       const float *A, const int lda,
                                                       const float *B, const int ldb,
                                                       const float beta,
                                                       float *C, const int ldc) {
  // y * xdim + x
  C[(blockIdx.y * blockDim.y + threadIdx.y)
    * (blockDim.x * gridDim.x)
    + (blockIdx.x * blockDim.x + threadIdx.x)] *= beta;
  //__syncthreads();

  __shared__ float a_shared[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float b_shared[BLOCK_SIZE][BLOCK_SIZE];

  float c_sub_acc = 0;
  // For blocks in grid.
#pragma unroll
  for (int bk = 0; bk < K / BLOCK_SIZE; bk++) {
    a_shared[threadIdx.y][threadIdx.x] = A[(blockIdx.y * BLOCK_SIZE + threadIdx.y) * lda + (bk * BLOCK_SIZE + threadIdx.x)];
    b_shared[threadIdx.y][threadIdx.x] = B[(bk * BLOCK_SIZE + threadIdx.y) * ldb + (blockIdx.x * BLOCK_SIZE + threadIdx.x)];
    // Waiting for data to finish loading into Shared memory.
    __syncthreads();

    // For elements in a block.
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k++) {
      c_sub_acc += alpha * a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
    }
    // To prevent the case from happening:
    // The next round of data is loaded when the data in share memory is not used up.
    __syncthreads();
  }

  C[(blockIdx.y * BLOCK_SIZE + threadIdx.y) * ldc + (blockIdx.x * BLOCK_SIZE + threadIdx.x)] += c_sub_acc;
}

////////////////////////////////////
void Gemm::GpuKernelsSetup() {
  gpu_kernels_.clear();
  // Kernel v0.
  {
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
                    void *C, const int ldc) -> void{
      GemmDeviceV0 << <config.blocks_per_grid, config.threads_per_block >> >
        (M, N, K, alpha, (float *)A, lda, (float *)B, ldb, beta, (float *)C, ldc);
    };

    GemmGpuKernelIF *kernel = new GemmGpuKernelIF();
    kernel->type_flag = TypeFlag::FLOAT32;   
    kernel->describe_info = "Normal(Block-based)";
    kernel->get_config = get_config;
    kernel->func = func;
    kernel->config_kernel = GemmDeviceV0;

    gpu_kernels_.push_back(kernel);
  }
  // Kernel v1.
  {
    auto get_config = [&](int M, int N) -> Config2D {
      Config2D config;
      const int block_size = 32;
      config.threads_per_block = dim3(block_size, block_size);
      config.blocks_per_grid = dim3(N / config.threads_per_block.x, M / config.threads_per_block.y);
      config.shared_memory_size = 2 * config.threads_per_block.x * config.threads_per_block.y * sizeof(float);
      return config;
    };
    auto func = [&](Config2D config,
                    const int M, const int N,
                    const int K, const float alpha,
                    const void *A, const int lda,
                    const void *B, const int ldb,
                    const float beta,
                    void *C, const int ldc) -> void {
      GemmDeviceV1 << <config.blocks_per_grid, config.threads_per_block, config.shared_memory_size >> >
        (M, N, K, alpha, (float *)A, lda, (float *)B, ldb, beta, (float *)C, ldc);
    };

    GemmGpuKernelIF *kernel = new GemmGpuKernelIF();
    kernel->type_flag = TypeFlag::FLOAT32;  
    kernel->describe_info = "Shared memory";
    kernel->get_config = get_config;
    kernel->func = func;
    kernel->config_kernel = GemmDeviceV1;

    gpu_kernels_.push_back(kernel);
  }
  // Kernel v2 - 2D.
  {
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
      GemmDeviceV2<32> << <config.blocks_per_grid, config.threads_per_block >> >
        (M, N, K, alpha, (float *)A, lda, (float *)B, ldb, beta, (float *)C, ldc);
    };

    GemmGpuKernelIF *kernel = new GemmGpuKernelIF();
    kernel->type_flag = TypeFlag::FLOAT32;
    kernel->describe_info = "Shared memory - 2D";
    kernel->get_config = get_config;
    kernel->func = func;
    kernel->config_kernel = nullptr;

    gpu_kernels_.push_back(kernel);
  }
  // Kernel v3.
  {
    auto get_config = [&](int M, int N) -> Config2D {
      Config2D config;
      return config;
    };
    auto func = [&](Config2D config,
                    const int M, const int N,
                    const int K, const float alpha,
                    const void *A, const int lda,
                    const void *B, const int ldb,
                    const float beta,
                    void *C, const int ldc) -> void {
      // Note: Column first in cublas.
      CUBLAS_CHECK(cublasSgemm(assistor_->cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K, &alpha, (float *)B, ldb, (float *)A, lda, &beta, (float *)C, ldc));
    };

    GemmGpuKernelIF *kernel = new GemmGpuKernelIF();
    kernel->type_flag = TypeFlag::FLOAT32;   
    kernel->describe_info = "Cublas";
    kernel->get_config = get_config;
    kernel->func = func;
    kernel->config_kernel = nullptr;

    gpu_kernels_.push_back(kernel);
  }
  // Kernel v4.
  {
    auto get_config = [&](int M, int N) -> Config2D {
      Config2D config;
      return config;
    };
    auto func = [&](Config2D config,
                    const int M, const int N,
                    const int K, const float alpha,
                    const void *A, const int lda,
                    const void *B, const int ldb,
                    const float beta,
                    void *C, const int ldc) -> void {
      // Note: Column first in cublas.
      //CUBLAS_CHECK(cublasSgemm(assistor_->cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
      //  N, M, K, &alpha, (float *)B, ldb, (float *)A, lda, &beta, (float *)C, ldc));
      CUBLAS_CHECK(cublasSgemmEx(assistor_->cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha, (half *)B, CUDA_R_16F, ldb,
                                 (half *)A, CUDA_R_16F, lda, &beta,
                                 (half *)C, CUDA_R_16F, ldc));
    }; 

    GemmGpuKernelIF *kernel = new GemmGpuKernelIF();
    kernel->type_flag = TypeFlag::FLOAT16;
    kernel->describe_info = "Cublas / Half";
    kernel->get_config = get_config;
    kernel->func = func;
    kernel->config_kernel = nullptr;

    gpu_kernels_.push_back(kernel);
  }
}

} // namespace cux