#include "operator/dot_product.h"

namespace cux {
  
// CUDA kernel V0 : 283ms
// Multiply and save to shared memory.
// Accumulate data from all of the shared memory to fewer blocks.
//template <int BLOCK_SIZE>
__global__ void VectorDotProductDeviceV0(const float *vec_a, const float *vec_b, const int len, float &res) {
  // Prevents memory access across the border.
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < len;
    i += blockDim.x * gridDim.x) {
    //__shared__ float smem[BLOCK_SIZE];
    extern __shared__ float smem[]; // Dynamic allocation.
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

// CUDA kernel V1 : 201ms
// Compute two blocks' data to the shared memory of one block.
__global__ void VectorDotProductDeviceV1(const float *vec_a, const float *vec_b, const int len, float &res) {
  // Prevents memory access across the border.
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < len / 2;
    i += blockDim.x * gridDim.x) {
    //__shared__ float smem[BLOCK_SIZE];
    extern __shared__ float smem[]; // Dynamic allocation.
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

// CUDA kernel V2 : 179ms
// Condition: The block size should be bigger than 32
// Unroll the last warp
__global__ void VectorDotProductDeviceV2(const float *vec_a, const float *vec_b, const int len, float &res) {
  // Prevents memory access across the border.
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < len / 2;
    i += blockDim.x * gridDim.x) {
    extern __shared__ float smem[]; // Dynamic allocation.
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

void VectorDotProduct::VectorDotProductDevice(const int kernel_id, const float *vec_a, 
                                              const float *vec_b, const int len, float &res) {
  // Default Layout.
  const int threads_per_block = 1024; // data_len % threads_per_block == 0
  const int blocks_per_grid = (len + threads_per_block - 1) / threads_per_block;
  int shared_memory_size = threads_per_block * sizeof(float);

  gpu_kernel_occupancys_.resize(gpu_kernel_cnt_);
  gpu_kernel_active_blocks_.resize(gpu_kernel_cnt_);
  switch (kernel_id) {
  case 0:
    VectorDotProductDeviceV0<< <blocks_per_grid, threads_per_block, shared_memory_size >> >
      (vec_a, vec_b, len, res);
    op_params_.launch_config->GetPotentialOccupancy(
      VectorDotProductDeviceV0, threads_per_block, shared_memory_size,
      gpu_kernel_active_blocks_[kernel_id], gpu_kernel_occupancys_[kernel_id]);
    break;
  case 1:
    VectorDotProductDeviceV1<< <blocks_per_grid, threads_per_block, shared_memory_size >> >
      (vec_a, vec_b, len, res);
    op_params_.launch_config->GetPotentialOccupancy(
      VectorDotProductDeviceV1, threads_per_block, shared_memory_size,
      gpu_kernel_active_blocks_[kernel_id], gpu_kernel_occupancys_[kernel_id]);
    break;
  case 2:
    VectorDotProductDeviceV2<< <blocks_per_grid, threads_per_block, shared_memory_size >> >
      (vec_a, vec_b, len, res);
    op_params_.launch_config->GetPotentialOccupancy(
      VectorDotProductDeviceV2, threads_per_block, shared_memory_size,
      gpu_kernel_active_blocks_[kernel_id], gpu_kernel_occupancys_[kernel_id]);
    break;
  default:
    CUXLOG_ERR("Device Kernel id (%d) not found.", kernel_id);
  }
}

//////////////////
// cuda version.
void VectorDotProduct::RunOnDevice() {
  // Time recorder.
  GpuTimer gpu_timer;

  // Input.
  gpu_timer.Start();
  const float *vec_a = in_a_->GetGpuData(PUSH_IF_EMPTY);
  const float *vec_b = in_b_->GetGpuData(PUSH_IF_EMPTY);
  const int len = in_a_->num_element();
  float *result = out_->GetGpuData(NO_PUSH);
  gpu_timer.Stop();
  gpu_time_in_record_ = gpu_timer.MilliSeconds();

  // Warm up.
  gpu_timer.Start();
  VectorDotProductDevice(0, vec_a, vec_b, len, *result);
  gpu_timer.Stop();
  gpu_time_warnup_record_ = gpu_timer.MilliSeconds();

  // Run.
  gpu_time_kernel_record_.clear();
  for (int ki = 0; ki < gpu_kernel_cnt_; ki++) {
    gpu_timer.Start();
    for (int i = 0; i < op_params_.loop_cn; i++) {
      cudaMemset(result, 0, sizeof(float));
      VectorDotProductDevice(ki, vec_a, vec_b, len, *result);
    }
    gpu_timer.Stop();
    gpu_time_kernel_record_.push_back(gpu_timer.MilliSeconds() / op_params_.loop_cn);

    // Output, Only record the first time.
    if (ki == 0) {
      gpu_timer.Start();
      out_->GetCpuData(PUSH);
      gpu_timer.Stop();
      gpu_time_out_record_ = gpu_timer.MilliSeconds();
    }
    checker_.CheckArray(out_->GetCpuData(PUSH), out_->num_element(), ki);
  }
}

}