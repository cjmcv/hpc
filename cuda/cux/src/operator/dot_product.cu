#include "operator/dot_product.h"

namespace cux {
  
// CUDA version 0: 1.09 ms
// Multiply and save to shared memory.
// Accumulate data from all of the shared memory to fewer blocks.
//template <int BLOCK_SIZE>
__global__ void VectorDotProductDeviceV0(const int len, const float *vec_a, const float *vec_b, float *res) {
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
      atomicAdd(res, smem[0]);
  }
}

// CUDA version 1: 0.46 ms
__global__ void VectorDotProductDeviceV1(const int len, const float *vec_a, const float *vec_b, float *res) {
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
      atomicAdd(res, smem[0]);
  }
}

// CUDA version 2: 0.45 ms
__global__ void VectorDotProductDeviceV2(const int len, const float *vec_a, const float *vec_b, float *res) {
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
      atomicAdd(res, smem[0]);
  }
}

void VectorDotProduct::PrepareLaunchConfig(int len) {
  // V0 & V1 & V2
  std::vector<void *> kernel_funcs = { VectorDotProductDeviceV0, VectorDotProductDeviceV1, 
                                       VectorDotProductDeviceV2 };
  for(int id = 0; id < kernel_funcs.size(); id++) {
    Config1D config;
    config = op_params_.launch_config->CalGetOccupancyConfig<Config1D>(&len, kernel_funcs[id], 0, len);
    config.shared_memory_size = config.threads_per_block * sizeof(float);
    config_1d_[id] = config;    
    
    op_params_.launch_config->QueryPotentialOccupancy(
      kernel_funcs[id], config.threads_per_block, config.shared_memory_size,
      gpu_kernel_active_blocks_[id], gpu_kernel_occupancys_[id]);
  }
  // default.
  //config.threads_per_block = 1024;
  //config.blocks_per_grid = (len + config.threads_per_block - 1) / config.threads_per_block;    
}

std::string &VectorDotProduct::GetDeviceKernelsInfo(int kernel_id) {
  static std::string info[4] = { "Shared memory",
                                 "Shared memory / Loop unrolling",
                                 "Shared memory / Loop unrolling",
                                 "Cublas" };
  if (kernel_id < 0 || kernel_id >= 4) {
    CUXLOG_ERR("GetDeviceKernelsInfo -> Device Kernel id (%d) not found.", kernel_id);
  }
  return info[kernel_id];
}

void VectorDotProduct::VectorDotProductDevice(int kernel_id, int len, 
                                              const float *vec_a, const float *vec_b, 
                                              float *res) {
  switch (kernel_id) {
  case 0:
    VectorDotProductDeviceV0<< <config_1d_[kernel_id].blocks_per_grid, 
                                config_1d_[kernel_id].threads_per_block, 
                                config_1d_[kernel_id].shared_memory_size >> >
      (len, vec_a, vec_b, res);
    break;
  case 1:
    VectorDotProductDeviceV1<< <config_1d_[kernel_id].blocks_per_grid, 
                                config_1d_[kernel_id].threads_per_block, 
                                config_1d_[kernel_id].shared_memory_size >> >
      (len, vec_a, vec_b, res);
    break;
  case 2:
    VectorDotProductDeviceV2<< <config_1d_[kernel_id].blocks_per_grid, 
                                config_1d_[kernel_id].threads_per_block, 
                                config_1d_[kernel_id].shared_memory_size >> >
      (len, vec_a, vec_b, res);
    break;
  case 3:
    // CUDA version 4: 0.59 ms
    // CUBLAS_POINTER_MODE_DEVICE: Return data on device -> res is a pointer for device.
    // CUBLAS_POINTER_MODE_HOST: On host.
    CUBLAS_CHECK(cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_DEVICE));
    CUBLAS_CHECK(cublasSdot(cublas_handle_, len, vec_a, 1, vec_b, 1, res));
    break;
  default:
    CUXLOG_ERR("VectorDotProductDevice -> Device Kernel id (%d) not found.", kernel_id);
  }
}

}