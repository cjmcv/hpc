#include "operator/dot_product.h"

namespace cux {
  
// CUDA kernel v1 : 283ms
// Multiply and save to shared memory.
// Accumulate data from all of the shared memory to fewer blocks.
template <int BLOCK_SIZE>
__global__ void VectorDotProductKernelv1(const float *vec_a, const float *vec_b, const int len, float &res) {
  // Prevents memory access across the border.
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < len;
    i += blockDim.x * gridDim.x) {
    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = vec_a[i] * vec_b[i];
    __syncthreads();

    int count = BLOCK_SIZE / 2;
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
      atomicAdd(&res, smem[0]);
  }
}

// CUDA kernel v2 : 201ms
// Compute two blocks' data to the shared memory of one block.
template <int BLOCK_SIZE>
__global__ void VectorDotProductKernelv2(const float *vec_a, const float *vec_b, const int len, float &res) {
  // Prevents memory access across the border.
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < len / 2;
    i += blockDim.x * gridDim.x) {
    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = vec_a[i] * vec_b[i] + vec_a[i + gridDim.x / 2] * vec_b[i + gridDim.x / 2];  // Mainly in here.
    __syncthreads();

    int count = BLOCK_SIZE >> 1;
    while (count >= 1) {
      if (threadIdx.x < count) {
        smem[threadIdx.x] += smem[count + threadIdx.x];
      }
      // Synchronize the threads within the block,
      // then go to next round together.
      __syncthreads();
      count >>= 1;
    }

    if (threadIdx.x == 0)
      atomicAdd(&res, smem[0]);
  }
}

// CUDA kernel v3 : 179ms
// Condition: The block size should be bigger than 32
// Unroll the last warp
template <int BLOCK_SIZE>
__global__ void VectorDotProductKernelv3(const float *vec_a, const float *vec_b, const int len, float &res) {
  // Prevents memory access across the border.
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < len / 2;
    i += blockDim.x * gridDim.x) {
    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = vec_a[i] * vec_b[i] + vec_a[i + gridDim.x / 2] * vec_b[i + gridDim.x / 2];
    __syncthreads();

    for (int count = BLOCK_SIZE >> 1; count > 32; count >>= 1) {
      if (threadIdx.x < count) {
        smem[threadIdx.x] += smem[count + threadIdx.x];
      }
      __syncthreads();
    }

    // Mainly in here. Unroll the last warp. (It still need __syncthreads() in a warp ?)
    if (threadIdx.x < 32) {
      smem[threadIdx.x] += smem[threadIdx.x + 32]; __syncthreads();
      smem[threadIdx.x] += smem[threadIdx.x + 16]; __syncthreads();
      smem[threadIdx.x] += smem[threadIdx.x + 8];  __syncthreads();
      smem[threadIdx.x] += smem[threadIdx.x + 4];  __syncthreads();
      smem[threadIdx.x] += smem[threadIdx.x + 2];  __syncthreads();
      smem[threadIdx.x] += smem[threadIdx.x + 1];  __syncthreads();
    }

    if (threadIdx.x == 0)
      atomicAdd(&res, smem[0]);
  }
}

void VectorDotProductHost(const float *vec_a, const float *vec_b, const int len, float &res) {
  res = 0;
  for (int i = 0; i < len; i++) {
    res += vec_a[i] * vec_b[i];
  }
}
////////////////////////////////////////////////
// CPU version: 965ms
// Normal version in cpu as a reference
void VectorDotProduct::RunOnHost(const float *vec_a, const float *vec_b, const int len, float &result) {
  CpuTimer cpu_timer;

  cpu_timer.Start();
  for (int i = 0; i < loops_; i++) {
    result = 0;
    VectorDotProductHost(vec_a, vec_b, len, result);
  }
  cpu_timer.Stop();

  cpu_time_record_ = cpu_timer.MilliSeconds();
}

void VectorDotProduct::RunOnDevice(const float *vec_a, const float *vec_b, const int len, float &result) {
  // Time recorder.
  GpuTimer gpu_timer;

  const int threads_per_block = 1024; // data_len % threads_per_block == 0
  const int blocks_per_grid = (len + threads_per_block - 1) / threads_per_block;

  // Warm up.
  VectorDotProductKernelv3<threads_per_block> << <blocks_per_grid, threads_per_block >> >
    (vec_a, vec_b, len, result);

  gpu_timer.Start();

  for (int i = 0; i < loops_; i++) {
    cudaMemset(&result, 0, sizeof(float));
    VectorDotProductKernelv3<threads_per_block> << <blocks_per_grid, threads_per_block >> >
      (vec_a, vec_b, len, result);
  }

  gpu_timer.Stop();
  gpu_time_record_ = gpu_timer.MilliSeconds();
}

}