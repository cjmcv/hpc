#include "operator/dot_product.h"

namespace cux {
  
// CUDA kernel v0 : 283ms
// Multiply and save to shared memory.
// Accumulate data from all of the shared memory to fewer blocks.
//template <int BLOCK_SIZE>
__global__ void VectorDotProductKernelv0(const float *vec_a, const float *vec_b, const int len, float &res) {
  // Prevents memory access across the border.
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < len;
    i += blockDim.x * gridDim.x) {
    //__shared__ float smem[BLOCK_SIZE];
    extern __shared__ int smem[]; // Dynamic allocation.
    smem[threadIdx.x] = vec_a[i] * vec_b[i];
    __syncthreads();

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
      atomicAdd(&res, smem[0]);
  }
}

// CUDA kernel v1 : 201ms
// Compute two blocks' data to the shared memory of one block.
__global__ void VectorDotProductKernelv1(const float *vec_a, const float *vec_b, const int len, float &res) {
  // Prevents memory access across the border.
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < len / 2;
    i += blockDim.x * gridDim.x) {
    //__shared__ float smem[BLOCK_SIZE];
    extern __shared__ int smem[]; // Dynamic allocation.
    smem[threadIdx.x] = vec_a[i] * vec_b[i] + vec_a[i + gridDim.x / 2] * vec_b[i + gridDim.x / 2];  // Mainly in here.
    __syncthreads();

    int count = blockDim.x >> 1;
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

// CUDA kernel v2 : 179ms
// Condition: The block size should be bigger than 32
// Unroll the last warp
__global__ void VectorDotProductKernelv2(const float *vec_a, const float *vec_b, const int len, float &res) {
  // Prevents memory access across the border.
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < len / 2;
    i += blockDim.x * gridDim.x) {
    extern __shared__ int smem[]; // Dynamic allocation.
    smem[threadIdx.x] = vec_a[i] * vec_b[i] + vec_a[i + gridDim.x / 2] * vec_b[i + gridDim.x / 2];
    __syncthreads();

    for (int count = blockDim.x >> 1; count > 32; count >>= 1) {
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

void VectorDotProductKernel(const int kernel_id, const int blocks_per_grid, const int threads_per_block, 
                            const float *vec_a, const float *vec_b, const int len, float &res) {
  int shared_memory_size = threads_per_block * sizeof(float);
  switch (kernel_id) {
  case 0:
    VectorDotProductKernelv0<< <blocks_per_grid, threads_per_block, shared_memory_size >> >
      (vec_a, vec_b, len, res);
    break;
  case 1:
    VectorDotProductKernelv1<< <blocks_per_grid, threads_per_block, shared_memory_size >> >
      (vec_a, vec_b, len, res);
    break;
  case 2:
    VectorDotProductKernelv2<< <blocks_per_grid, threads_per_block, shared_memory_size >> >
      (vec_a, vec_b, len, res);
    break;
  default:
    CUXLOG_ERR("");
  }
}

//////////////////
// cuda version.
void VectorDotProduct::RunOnDevice() {
  // Time recorder.
  GpuTimer gpu_timer;

  // Input.
  gpu_timer.Start();
  const float *vec_a = in_a_->GetGpuData();
  const float *vec_b = in_b_->GetGpuData();
  const int len = in_a_->num_element();
  float *result = out_->GetGpuData();
  gpu_timer.Stop();
  gpu_time_in_record_ = gpu_timer.MilliSeconds();

  // Layout.
  const int threads_per_block = 1024; // data_len % threads_per_block == 0
  const int blocks_per_grid = (len + threads_per_block - 1) / threads_per_block;

  // Warm up.
  gpu_timer.Start();
  VectorDotProductKernel(0, blocks_per_grid, threads_per_block, 
    vec_a, vec_b, len, *result);
  gpu_timer.Stop();
  gpu_time_warnup_record_ = gpu_timer.MilliSeconds();

  // Run.
  gpu_time_kernel_record_.clear();
  for (int ki = 0; ki < 3; ki++) {
    gpu_timer.Start();
    for (int i = 0; i < loops_; i++) {
      cudaMemset(result, 0, sizeof(float));
      VectorDotProductKernel(ki, blocks_per_grid, threads_per_block,
        vec_a, vec_b, len, *result);
    }
    gpu_timer.Stop();
    gpu_time_kernel_record_.push_back(gpu_timer.MilliSeconds() / loops_);
  }

  // Output.
  gpu_timer.Start();
  CUXLOG_COUT("result: %f.", *out_->GetCpuData());
  gpu_timer.Stop();
  gpu_time_out_record_ = gpu_timer.MilliSeconds();
}

}