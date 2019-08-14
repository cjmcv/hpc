#include "operator/dot_product.h"

namespace cux {
  
// Kernel V0W
// Multiply and save to shared memory.
// Accumulate data from all of the shared memory to fewer blocks.
//template <int BLOCK_SIZE>
__global__ void DotDeviceV0(const int len, const float *vec_a, const float *vec_b, float *res) {
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

// Kernel V1
__global__ void DotDeviceV1(const int len, const float *vec_a, const float *vec_b, float *res) {
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

// Kernel V2.
__global__ void DotDeviceV2(const int len, const float *vec_a, const float *vec_b, float *res) {
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

void Dot::GpuKernelsSetup() {
  gpu_kernels_.clear();
  // Kernel v0
  {
    auto get_config = [&](int len) -> Config1D {
      Config1D config;
      config.threads_per_block = 1024;
      config.blocks_per_grid = (len + config.threads_per_block - 1) / config.threads_per_block;
      //config = op_params_.launch_config->CalGetOccupancyConfig<Config1D>(&len, DotDeviceV0, 0, len);
      config.shared_memory_size = config.threads_per_block * sizeof(float);
      return config;
    };
    auto func = [&](Config1D config, int len, const void *vec_a, const void *vec_b, void *res) -> void {
      DotDeviceV0 << <config.blocks_per_grid,
        config.threads_per_block,
        config.shared_memory_size >> >
        (len, (float *)vec_a, (float *)vec_b, (float *)res);
    };

    DotGpuKernelIF *kernel = new DotGpuKernelIF();
    kernel->type_flag = TypeFlag::FLOAT32;
    kernel->describe_info = "Shared memory";
    kernel->get_config = get_config;
    kernel->func = func;
    kernel->kernel_address = DotDeviceV0;

    gpu_kernels_.push_back(kernel);
  }
  // Kernel v1
  {
    auto get_config = [&](int len) -> Config1D {
      Config1D config;
      config.threads_per_block = 1024;
      config.blocks_per_grid = (len + config.threads_per_block - 1) / config.threads_per_block;
      //config = op_params_.launch_config->CalGetOccupancyConfig<Config1D>(&len, DotDeviceV1, 0, len);
      config.shared_memory_size = config.threads_per_block * sizeof(float);
      return config;
    };
    auto func = [&](Config1D config, int len, const void *vec_a, const void *vec_b, void *res) -> void {
      DotDeviceV1 << <config.blocks_per_grid,
        config.threads_per_block,
        config.shared_memory_size >> >
        (len, (float *)vec_a, (float *)vec_b, (float *)res);
    };

    DotGpuKernelIF *kernel = new DotGpuKernelIF();
    kernel->type_flag = TypeFlag::FLOAT32;
    kernel->describe_info = "Shared memory / Loop unrolling";
    kernel->get_config = get_config;
    kernel->func = func;
    kernel->kernel_address = DotDeviceV1;

    gpu_kernels_.push_back(kernel);
  }
  // Kernel v2
  {
    auto get_config = [&](int len) -> Config1D {
      Config1D config;
      config.threads_per_block = 1024;
      config.blocks_per_grid = (len + config.threads_per_block - 1) / config.threads_per_block;
      //config = op_params_.launch_config->CalGetOccupancyConfig<Config1D>(&len, DotDeviceV2, 0, len);
      config.shared_memory_size = config.threads_per_block * sizeof(float);
      return config;
    };
    auto func = [&](Config1D config, int len, const void *vec_a, const void *vec_b, void *res) -> void {
      DotDeviceV2 << <config.blocks_per_grid,
        config.threads_per_block,
        config.shared_memory_size >> >
        (len, (float *)vec_a, (float *)vec_b, (float *)res);
    };

    DotGpuKernelIF *kernel = new DotGpuKernelIF();
    kernel->type_flag = TypeFlag::FLOAT32;  
    kernel->describe_info = "Shared memory / Loop unrolling";
    kernel->get_config = get_config;
    kernel->func = func;
    kernel->kernel_address = DotDeviceV2;

    gpu_kernels_.push_back(kernel);
  }
  // Kernel v3.
  {
    auto get_config = [&](int len) -> Config1D {
      Config1D config;
      return config;
    };
    auto func = [&](Config1D config, int len, const void *vec_a, const void *vec_b, void *res) -> void {
      // CUBLAS_POINTER_MODE_DEVICE: Return data on device -> res is a pointer for device.
      // CUBLAS_POINTER_MODE_HOST: On host.
      CUBLAS_CHECK(cublasSetPointerMode(assistor_->cublas_handle(), CUBLAS_POINTER_MODE_DEVICE));
      CUBLAS_CHECK(cublasSdot(assistor_->cublas_handle(), len, (float *)vec_a, 1, (float *)vec_b, 1, (float *)res));
    };

    DotGpuKernelIF *kernel = new DotGpuKernelIF();
    kernel->type_flag = TypeFlag::FLOAT32;
    kernel->describe_info = "Cublas";
    kernel->get_config = get_config;
    kernel->func = func;
    kernel->kernel_address = nullptr;

    gpu_kernels_.push_back(kernel);
  }
}

}