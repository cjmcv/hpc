#include "operator/dot_product.h"

namespace cux {
  
// CUDA kernel V0
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

    // Summarize in blocks.
    // Limiting conditions: len should be a multiple of block_size.
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

// CUDA kernel V1
// Compute two blocks' data to the shared memory of one block.
__global__ void VectorDotProductDeviceV1(const float *vec_a, const float *vec_b, const int len, float &res) {
  // Prevents memory access across the border.
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < len / 2;
    i += blockDim.x * gridDim.x) {
    extern __shared__ float smem[]; // Dynamic allocation.
    // Mainly in here. 
    // Limiting conditions: len is at least twice as large as block_size and len will be a multiple of 2.
    smem[threadIdx.x] = vec_a[i] * vec_b[i] +vec_a[i + len / 2] * vec_b[i + len / 2];
    __syncthreads();

    // Limiting conditions: len should be a multiple of block_size.
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

// CUDA kernel V2
__global__ void VectorDotProductDeviceV2(const float *vec_a, const float *vec_b, const int len, float &res) {
  // Prevents memory access across the border.
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < len / 8;
    i += blockDim.x * gridDim.x) {
    extern __shared__ float smem[]; // Dynamic allocation.
    int len1 = len / 8;
    int len2 = len1 * 2;
    int len3 = len1 * 3;
    int len4 = len1 * 4;
    int len5 = len1 * 5;
    int len6 = len1 * 6;
    int len7 = len1 * 7;
    // Limiting conditions: len is at least eight times larger than block_size and len will be a multiple of 8.
    smem[threadIdx.x] = vec_a[i] * vec_b[i] + vec_a[i + len1] * vec_b[i + len1] + 
      vec_a[i + len2] * vec_b[i + len2] + vec_a[i + len3] * vec_b[i + len3] + 
      vec_a[i + len4] * vec_b[i + len4] + vec_a[i + len5] * vec_b[i + len5] + 
      vec_a[i + len6] * vec_b[i + len6] + vec_a[i + len7] * vec_b[i + len7];
    __syncthreads();

    for (int count = blockDim.x >> 1; count >= 1; count >>= 1) {
      if (threadIdx.x < count) {
        smem[threadIdx.x] += smem[count + threadIdx.x];
      }
      __syncthreads();
    }

    if (threadIdx.x == 0)
      atomicAdd(&res, smem[0]);
  }
}

// CUDA kernel V3
__global__ void VectorDotProductDeviceV3(const float *vec_a, const float *vec_b, const int len, float &res) {
  // Prevents memory access across the border.
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < len / 8;
    i += blockDim.x * gridDim.x) {
    extern __shared__ float smem[]; // Dynamic allocation.
    int len1 = len / 8;
    int len2 = len1 * 2;
    int len3 = len1 * 3;
    int len4 = len1 * 4;
    int len5 = len1 * 5;
    int len6 = len1 * 6;
    int len7 = len1 * 7;
    // Limiting conditions: len is at least eight times larger than block_size and len will be a multiple of 8.
    smem[threadIdx.x] = vec_a[i] * vec_b[i] + vec_a[i + len1] * vec_b[i + len1] +
      vec_a[i + len2] * vec_b[i + len2] + vec_a[i + len3] * vec_b[i + len3] +
      vec_a[i + len4] * vec_b[i + len4] + vec_a[i + len5] * vec_b[i + len5] +
      vec_a[i + len6] * vec_b[i + len6] + vec_a[i + len7] * vec_b[i + len7];
    __syncthreads();

    int count = blockDim.x >> 1;
    for ( ; count > 32; count >>= 1) {
      if (threadIdx.x < count) {
        smem[threadIdx.x] += smem[count + threadIdx.x];
      }
      __syncthreads();
    }

    // Loop unrolling.
    // Limiting conditions: len should be a multiple of 32.
    // TODO: Why do we need block synchronization for the same warp?
    if (threadIdx.x < 32) smem[threadIdx.x] += smem[32 + threadIdx.x]; __syncthreads();
    if (threadIdx.x < 16) smem[threadIdx.x] += smem[16 + threadIdx.x]; __syncthreads();
    if (threadIdx.x < 8) smem[threadIdx.x] += smem[8 + threadIdx.x]; __syncthreads();
    if (threadIdx.x < 4) smem[threadIdx.x] += smem[4 + threadIdx.x]; __syncthreads();
    if (threadIdx.x < 2) smem[threadIdx.x] += smem[2 + threadIdx.x]; __syncthreads();
    if (threadIdx.x < 1) smem[threadIdx.x] += smem[1 + threadIdx.x]; __syncthreads();

    if (threadIdx.x == 0)
      atomicAdd(&res, smem[0]);
  }
}

void VectorDotProduct::VectorDotProductDevice(const int kernel_id, const float *vec_a, 
                                              const float *vec_b, const int len, float &res) {
  // Layout.
  Config1D config;
  int shared_memory_size;

  gpu_kernel_occupancys_.resize(gpu_kernel_cnt_);
  gpu_kernel_active_blocks_.resize(gpu_kernel_cnt_);
  switch (kernel_id) {
  case 0:
    config = op_params_.launch_config->CalGetOccupancyConfig<Config1D>(&len, VectorDotProductDeviceV0, 0, len);
    shared_memory_size = config.threads_per_block * sizeof(float);

    VectorDotProductDeviceV0<< <config.blocks_per_grid, config.threads_per_block, shared_memory_size >> >
      (vec_a, vec_b, len, res);

    op_params_.launch_config->QueryPotentialOccupancy(
      VectorDotProductDeviceV0, config.threads_per_block, shared_memory_size,
      gpu_kernel_active_blocks_[kernel_id], gpu_kernel_occupancys_[kernel_id]);
    break;

  case 1:
    config = op_params_.launch_config->CalGetOccupancyConfig<Config1D>(&len, VectorDotProductDeviceV1, 0, len);
    shared_memory_size = config.threads_per_block * sizeof(float);

    VectorDotProductDeviceV1<< <config.blocks_per_grid, config.threads_per_block, shared_memory_size >> >
      (vec_a, vec_b, len, res);

    op_params_.launch_config->QueryPotentialOccupancy(
      VectorDotProductDeviceV1, config.threads_per_block, shared_memory_size,
      gpu_kernel_active_blocks_[kernel_id], gpu_kernel_occupancys_[kernel_id]);
    break;

  case 2:
    config = op_params_.launch_config->CalGetOccupancyConfig<Config1D>(&len, VectorDotProductDeviceV2, 0, len);
    shared_memory_size = config.threads_per_block * sizeof(float);

    VectorDotProductDeviceV2<< <config.blocks_per_grid, config.threads_per_block, shared_memory_size >> >
      (vec_a, vec_b, len, res);

    op_params_.launch_config->QueryPotentialOccupancy(
      VectorDotProductDeviceV2, config.threads_per_block, shared_memory_size,
      gpu_kernel_active_blocks_[kernel_id], gpu_kernel_occupancys_[kernel_id]);
    break;

  case 3:
    //config.threads_per_block = 1024;
    //config.blocks_per_grid = (len + config.threads_per_block - 1) / config.threads_per_block;    
    config = op_params_.launch_config->CalGetOccupancyConfig<Config1D>(&len, VectorDotProductDeviceV3, 0, len);
    shared_memory_size = config.threads_per_block * sizeof(float);

    VectorDotProductDeviceV3 << <config.blocks_per_grid, config.threads_per_block, shared_memory_size >> >
      (vec_a, vec_b, len, res);

    op_params_.launch_config->QueryPotentialOccupancy(
      VectorDotProductDeviceV3, config.threads_per_block, shared_memory_size,
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