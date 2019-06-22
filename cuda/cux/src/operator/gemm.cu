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
__global__ void GemmKernelv1(const int M, const int N, const int K, const float ALPHA,
  const float *A, const int lda,
  const float *B, const int ldb,
  float *C, const int ldc) {

  const int block_size = blockDim.x; // The Block is square.
  const int block_num_K = K / block_size;

  float c_sub_acc = 0;
  for (int bk = 0; bk < K / block_size; bk++) {
    for (int k = 0;k < block_size; k++) {
      c_sub_acc += A[(blockIdx.y * block_size + threadIdx.y) * lda + (bk * block_size + k)] *
        B[(bk * block_size + k) * ldb + (blockIdx.x * block_size + threadIdx.x)];
    }
  }

  C[(blockIdx.y * block_size + threadIdx.y) * ldc + (blockIdx.x * block_size + threadIdx.x)] += c_sub_acc;
}

// CUDA version 2.
// Use shared memory.
template <int BLOCK_SIZE>
__global__ void GemmKernelv2(const int M, const int N, const int K, const float ALPHA,
  const float *A, const int lda,
  const float *B, const int ldb,
  float *C, const int ldc) {

  __shared__ float a_shared[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float b_shared[BLOCK_SIZE][BLOCK_SIZE];

  float c_sub_acc = 0;
  // For blocks in grid.
  for (int bk = 0; bk < K / BLOCK_SIZE; bk++) {
    a_shared[threadIdx.y][threadIdx.x] = A[(blockIdx.y * BLOCK_SIZE + threadIdx.y) * lda + (bk * BLOCK_SIZE + threadIdx.x)];
    b_shared[threadIdx.y][threadIdx.x] = B[(bk * BLOCK_SIZE + threadIdx.y) * ldb + (blockIdx.x * BLOCK_SIZE + threadIdx.x)];
    // Wait for data to complete loading to Shared memory.
    __syncthreads();

    // For elements in a block.
    for (int k = 0;k < BLOCK_SIZE; k++) {
      c_sub_acc += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
    }
	  // To prevent the case from happening:
	  // The next round of data is loaded when the data in share memory is not used up.
    __syncthreads();
  }

  C[(blockIdx.y * BLOCK_SIZE + threadIdx.y) * ldc + (blockIdx.x * BLOCK_SIZE + threadIdx.x)] += c_sub_acc;
}

void GemmKernel(const int kernel_id, 
                const int M, const int N, const int K, const float ALPHA,
                const float *A, const int lda,
                const float *B, const int ldb,
                float *C, const int ldc) {
  switch (kernel_id) {
  case 0:
    break;
  case 1:
    break;
  default:
    CUXLOG_ERR("Kernel id not found.");
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
  const int M = A_->shape()[CuxShape::HEIGHT];
  const int N = A_->shape()[CuxShape::WIDTH];
  const int K = B_->shape()[CuxShape::WIDTH];
  const int lda = N;
  const int ldb = K;
  const int ldc = K;

  // Layout.
  const int block_size = 32;
  dim3 threads_per_block(block_size, block_size);
  dim3 blocks_per_grid(N / threads_per_block.x, M / threads_per_block.y);

  // Warm up.
  gpu_timer.Start();
  GemmKernelv1<< <blocks_per_grid, threads_per_block >> >
    (M, N, K, 1.0, A, lda, B, ldb, C, ldc);
  gpu_timer.Stop();
  gpu_time_warnup_record_ = gpu_timer.MilliSeconds();

  // Run.
  loops_ = 1;
  cudaMemset(C, 0, sizeof(float) * M * N);

  gpu_time_kernel_record_.clear();
  gpu_timer.Start();

  GemmKernelv1 << <blocks_per_grid, threads_per_block >> >
    (M, N, K, 1.0, A, lda, B, ldb, C, ldc);

  gpu_timer.Stop();
  gpu_time_kernel_record_.push_back(gpu_timer.MilliSeconds() / loops_);


  // Output.
  gpu_timer.Start();
  CUXLOG_COUT("result: %f.", *C_->GetCpuData());
  gpu_timer.Stop();
  gpu_time_out_record_ = gpu_timer.MilliSeconds();
}
}