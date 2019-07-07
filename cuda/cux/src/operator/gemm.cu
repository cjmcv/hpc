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

  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int xdim = blockDim.x * gridDim.x;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  C[y * xdim + x] *= beta;
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

// CUDA version 2.
// Base on version 1. Use fewer unnecessary registers.
//   Notice that v1 has only one active block, because 
// each thread USES three more registers than v2, 
// making it impossible to use more blocks
__global__ void GEMMDeviceV2(const int M, const int N,
                             const int K, const float alpha,
                             const float *A, const int lda,
                             const float *B, const int ldb,
                             const float beta,
                             float *C, const int ldc) {

  const int block_size = blockDim.x;  

  extern __shared__ float a_shared[];
  float *b_shared = &a_shared[blockDim.x * blockDim.x];

  // y * xdim + x
  C[(blockIdx.y * blockDim.y + threadIdx.y) 
    * (blockDim.x * gridDim.x) 
    + (blockIdx.x * blockDim.x + threadIdx.x)] *= beta;

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

void GEMM::GEMMDevice(const int kernel_id, 
                      const int M, const int N, 
                      const int K, const float alpha,
                      const float *A, const int lda,
                      const float *B, const int ldb,
                      const float beta,
                      float *C, const int ldc) {
  // Layout.
  const int block_size = 32;
  dim3 threads_per_block(block_size, block_size);
  dim3 blocks_per_grid(N / threads_per_block.x, M / threads_per_block.y);

  int shared_memory_size = 0;
  gpu_kernel_occupancys_.resize(gpu_kernel_cnt_);
  gpu_kernel_active_blocks_.resize(gpu_kernel_cnt_);
  switch (kernel_id) {
  case 0:
    GEMMDeviceV0 << <blocks_per_grid, threads_per_block >> >
      (M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    op_params_.launch_config->QueryPotentialOccupancy(
      GEMMDeviceV0, threads_per_block.x * threads_per_block.y, 0,
      gpu_kernel_active_blocks_[kernel_id], gpu_kernel_occupancys_[kernel_id]);
    break;
  case 1:
    // For:  __shared__ float a_shared[threads_per_block.y][threads_per_block.x];
    //       __shared__ float b_shared[threads_per_block.y][threads_per_block.x];
    shared_memory_size = 2 * threads_per_block.x * threads_per_block.y * sizeof(float);
    GEMMDeviceV1 << <blocks_per_grid, threads_per_block, shared_memory_size >> >
      (M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    op_params_.launch_config->QueryPotentialOccupancy(
      GEMMDeviceV1, threads_per_block.x * threads_per_block.y, shared_memory_size,
      gpu_kernel_active_blocks_[kernel_id], gpu_kernel_occupancys_[kernel_id]);
    break;
  case 2:
    shared_memory_size = 2 * threads_per_block.x * threads_per_block.y * sizeof(float);
    GEMMDeviceV2 << <blocks_per_grid, threads_per_block, shared_memory_size >> >
      (M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    op_params_.launch_config->QueryPotentialOccupancy(
      GEMMDeviceV2, threads_per_block.x * threads_per_block.y, shared_memory_size,
      gpu_kernel_active_blocks_[kernel_id], gpu_kernel_occupancys_[kernel_id]);
    break;
  default:
    CUXLOG_ERR("Device Kernel id (%d) not found.", kernel_id);
  }
}

//////////////////
// cuda version.
void GEMM::RunOnDevice() {
  // Time recorder.
  GpuTimer gpu_timer;

  // Input.
  gpu_timer.Start();
  const float *A = A_->GetGpuData(PUSH_IF_EMPTY);
  const float *B = B_->GetGpuData(PUSH_IF_EMPTY);
  float *C = C_->GetGpuData(PUSH_IF_EMPTY);
  gpu_timer.Stop();
  gpu_time_in_record_ = gpu_timer.MilliSeconds();

  const float alpha = kernel_params_.alpha;
  const float beta = kernel_params_.beta;
  const int M = A_->shape()[Shape::HEIGHT];
  const int N = B_->shape()[Shape::WIDTH];
  const int K = B_->shape()[Shape::HEIGHT]; // A_->shape()[Shape::WIDTH];
  const int lda = K;
  const int ldb = N;
  const int ldc = N;

  // Save original data.
  C_->Save(ON_DEVICE);

  // Warm up.
  gpu_timer.Start();
  GEMMDevice(0, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  gpu_timer.Stop();
  gpu_time_warnup_record_ = gpu_timer.MilliSeconds();

  // Run.
  gpu_time_kernel_record_.clear();
  for (int ki = 0; ki < gpu_kernel_cnt_; ki++) {
    gpu_timer.Start();
    for (int i = 0; i < op_params_.loop_cn; i++) {
      //cudaMemset(C, 0, sizeof(float) * M * N);
      C_->Restore(ON_DEVICE);
      GEMMDevice(ki, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    gpu_timer.Stop();
    gpu_time_kernel_record_.push_back(gpu_timer.MilliSeconds() / op_params_.loop_cn);

    // Output, Only record the first time.
    if (ki == 0) {
      gpu_timer.Start();
      C_->GetCpuData(PUSH);
      gpu_timer.Stop();
      gpu_time_out_record_ = gpu_timer.MilliSeconds();
    }
    checker_.CheckArray(C_->GetCpuData(PUSH), C_->num_element(), ki);
  }
}

} // namespace cux