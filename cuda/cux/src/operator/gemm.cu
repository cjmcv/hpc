/*!
* \brief gemm: C = A * B.
*/
#include "operator/gemm.h"

namespace cux {
// CUDA version 1: 72 ms
// It is rewrited from GemmCPUv2. 
// bi,bj can be replaced by blockIdx.x,blockIdx.y
// i,j can be replaced by threadIdx.x,threadIdx.y
// so just bk and k left. Grid and block is related to the dst matrix.
//
// \ C[ty, tx] = A[ty, k] * B[k, tx]
// for bk -> bk_num_per_grid
//     for k -> k_num_per_block
//         C[bi*bs + ty, bj*bs + tx] = A[bi*bs + ty, bk*bs + k] * B[k*bs + k, bj*bs + tx]
__global__ void GemmKernelv0(const int M, const int N, const int K, const float ALPHA,
  const float *A, const int lda,
  const float *B, const int ldb,
  float *C, const int ldc) {

  const int block_size = blockDim.x; // The Block is square.
  const int block_num_K = K / block_size;

  float c_sub_acc = 0;
  for (int bk = 0; bk < block_num_K; bk++) {
    for (int k = 0;k < block_size; k++) {
      c_sub_acc += ALPHA * A[(blockIdx.y * block_size + threadIdx.y) * lda + (bk * block_size + k)] *
        B[(bk * block_size + k) * ldb + (blockIdx.x * block_size + threadIdx.x)];
    }
  }

  C[(blockIdx.y * block_size + threadIdx.y) * ldc + (blockIdx.x * block_size + threadIdx.x)] += c_sub_acc;
}

// CUDA version 1.
// Use shared memory.
__global__ void GemmKernelv1(const int M, const int N, const int K, const float ALPHA,
  const float *A, const int lda,
  const float *B, const int ldb,
  float *C, const int ldc) {

  const int block_size = blockDim.x;  // Side length.
  // PS: If change them to extern __shared__ float a_shared[];
  //                       extern __shared__ float b_shared[];
  //     There will be errors in the calculation.
  //     Dynamic allocation of Shared memory supports only one array with one dimension?
  extern __shared__ float a_shared[];
  float *b_shared = &a_shared[blockDim.x * blockDim.x];
  //extern __shared__ float b_shared[];

  float c_sub_acc = 0;
  // For blocks in grid.
  for (int bk = 0; bk < K / block_size; bk++) {
    a_shared[threadIdx.y * block_size + threadIdx.x] = A[(blockIdx.y * block_size + threadIdx.y) * lda + (bk * block_size + threadIdx.x)];
    b_shared[threadIdx.y * block_size + threadIdx.x] = B[(bk * block_size + threadIdx.y) * ldb + (blockIdx.x * block_size + threadIdx.x)];
    // Wait for data to complete loading to Shared memory.
    __syncthreads();

    // For elements in a block.
    for (int k = 0; k < block_size; k++) {
      c_sub_acc += ALPHA * a_shared[threadIdx.y * block_size + k] * b_shared[k * block_size + threadIdx.x];
    }
    // To prevent the case from happening:
    // The next round of data is loaded when the data in share memory is not used up.
    __syncthreads();
  }

  C[(blockIdx.y * block_size + threadIdx.y) * ldc + (blockIdx.x * block_size + threadIdx.x)] += c_sub_acc;
}


void GemmKernel(const int kernel_id, const dim3 &blocks_per_grid, const dim3 &threads_per_block,
                const int M, const int N, const int K, const float ALPHA,
                const float *A, const int lda,
                const float *B, const int ldb,
                float *C, const int ldc) {
  int shared_memory_size = 0;
  switch (kernel_id) {
  case 0:
    GemmKernelv0 << <blocks_per_grid, threads_per_block >> >
      (M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    break;
  case 1:
    // For:  __shared__ float a_shared[threads_per_block.y][threads_per_block.x];
    //       __shared__ float b_shared[threads_per_block.y][threads_per_block.x];
    shared_memory_size = 2 * threads_per_block.x * threads_per_block.y * sizeof(float);
    GemmKernelv1 << <blocks_per_grid, threads_per_block, shared_memory_size >> >
      (M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    break;
  default:
    CUXLOG_ERR("Kernel id (%d) not found.", kernel_id);
  }
}

//////////////////
// cuda version.
void GEMM::RunOnDevice() {
  // Time recorder.
  GpuTimer gpu_timer;

  // Input.
  gpu_timer.Start();
  const float *A = A_->GetGpuData();
  const float *B = B_->GetGpuData();
  float *C = C_->GetGpuData();
  gpu_timer.Stop();
  gpu_time_in_record_ = gpu_timer.MilliSeconds();

  const float ALPHA = params_.alpha_;
  std::cout << "alpha: " << ALPHA << std::endl;
  const int M = A_->shape()[CuxShape::HEIGHT];
  const int N = B_->shape()[CuxShape::WIDTH];
  const int K = B_->shape()[CuxShape::HEIGHT]; // A_->shape()[CuxShape::WIDTH];
  const int lda = K;
  const int ldb = N;
  const int ldc = N;

  // Layout.
  const int block_size = 32;
  dim3 threads_per_block(block_size, block_size);
  dim3 blocks_per_grid(N / threads_per_block.x, M / threads_per_block.y);

  // Warm up.
  gpu_timer.Start();
  GemmKernel(0, blocks_per_grid, threads_per_block,
    M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
  gpu_timer.Stop();
  gpu_time_warnup_record_ = gpu_timer.MilliSeconds();

  // Run.
  gpu_time_kernel_record_.clear();
  for (int ki = 0; ki < 2; ki++) {
    gpu_timer.Start();
    for (int i = 0; i < loops_; i++) {
      cudaMemset(C, 0, sizeof(float) * M * N);
      GemmKernel(ki, blocks_per_grid, threads_per_block,
        M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    }
    gpu_timer.Stop();
    gpu_time_kernel_record_.push_back(gpu_timer.MilliSeconds() / loops_);
  }

  // Output.
  gpu_timer.Start();
  CUXLOG_COUT("result: %f.", *C_->GetCpuData());
  gpu_timer.Stop();
  gpu_time_out_record_ = gpu_timer.MilliSeconds();
}
}