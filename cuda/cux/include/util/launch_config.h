/*!
* \brief LaunchConfig.
*/

#ifndef CUX_LAUNCH_CONFIG_H_
#define CUX_LAUNCH_CONFIG_H_

#include <iostream>
#include <algorithm>

#include "util.h"

namespace cux { 

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

struct Config1D {
  // Logical number of thread that works on the elements. If each logical
  // thread works on exactly a single element, this is the same as the working
  // element count.
  int virtual_thread_count = -1;
  // Number of threads per block.
  int thread_per_block = -1;
  // Number of blocks for Cuda kernel launch.
  int block_per_grid = -1;
};

struct Config3D {
  dim3 virtual_thread_count = dim3(0, 0, 0);
  dim3 thread_per_block = dim3(0, 0, 0);
  dim3 block_per_grid = dim3(0, 0, 0);
};

class LaunchConfig {
public:
  LaunchConfig(Device *device) :device_(device) {}

  // occupancy = (double)active_warps / max_warps;
  void QueryPotentialOccupancy(const void *kernel, int thread_per_block,
                               size_t dynamic_shared_mem,
                               int &active_blocks, double &occupancy) {

    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, kernel, thread_per_block, dynamic_shared_mem));

    int active_warps = active_blocks * thread_per_block / device_->prop.warpSize;
    int max_warps = device_->prop.maxThreadsPerMultiProcessor / device_->prop.warpSize;

    occupancy = (double)active_warps / max_warps;
  }

  inline const Config3D &Get3DConfig() { return config_3d_; }
  inline const Config1D &Get1DConfig() { return config_1d_; }

  inline void CalHeuristicsConfig(int dimension, const int *num) {
    switch (dimension) {
    case 1:
      config_1d_ = GetKernelConfig1D(num[0]);
    case 2:
      config_3d_ = GetKernelConfig2D(num[0], num[1]);
    default:
      CUXLOG_ERR("%dD is not supported in GetHeuristicsConfig for now.", dimension);
    }
  }
  inline void CalGetOccupancyConfig(int dimension, const int *num, const void *kernel,
                                    size_t dynamic_smem_usage, int block_size_limit) {
    switch (dimension) {
    case 1:
      config_1d_ = GetKernelConfig1D(num[0], kernel, dynamic_smem_usage, block_size_limit);
    case 2:
      config_3d_ = GetKernelConfig2D(num[0], num[1], kernel, dynamic_smem_usage, block_size_limit);
    case 3:
      config_3d_ = GetKernelConfig3D(num[0], num[1], num[2], kernel, dynamic_smem_usage, block_size_limit);
    default:
      CUXLOG_ERR("%dD is not supported in GetOccupancyConfig for now.", dimension);
    }
  }

private:
  // Heuristics method. Without any knowledge of the device kernel
  // This is assuming the kernel is quite simple and will largely be memory-limited.
  inline Config1D GetKernelConfig1D(int element_count) const {
    Config1D config;

    const int virtual_thread_count = element_count;
    const int physical_thread_count = std::min(
      device_->prop.multiProcessorCount * device_->prop.maxThreadsPerMultiProcessor,
      virtual_thread_count);

    config.virtual_thread_count = virtual_thread_count;
    config.thread_per_block = std::min(1024, device_->prop.maxThreadsPerBlock);
    config.block_per_grid = std::min(DivUp(physical_thread_count, config.thread_per_block),
                              device_->prop.multiProcessorCount);
    return config;
  }

  // Occupancy-based launch configurator.
  // It suggests a block size that achieves the best theoretical occupancy.
  // But the occupancy can not be translated directly to performance.
  inline Config1D GetKernelConfig1D(int element_count, const void *kernel,
                                    size_t dynamic_smem_usage,
                                    int block_size_limit) const;

  // Heuristics method.
  inline Config3D GetKernelConfig2D(int xdim, int ydim) const {
    Config3D config;

    const int kThreadsPerBlock = 256;
    int block_cols = std::min(xdim, kThreadsPerBlock);
    // ok to round down here and just do more loops in the kernel
    int block_rows = std::max(kThreadsPerBlock / block_cols, 1);

    const int physical_thread_count =
      device_->prop.multiProcessorCount * device_->prop.maxThreadsPerMultiProcessor;

    const int max_blocks = std::max(physical_thread_count / kThreadsPerBlock, 1);

    config.virtual_thread_count = dim3(xdim, ydim, 1);
    config.thread_per_block = dim3(block_cols, block_rows, 1);

    int grid_x = std::min(DivUp(xdim, block_cols), max_blocks);
    int grid_y = std::min(max_blocks / grid_x, std::max(ydim / block_rows, 1));
    config.block_per_grid = dim3(grid_x, grid_y, 1);
    return config;
  }

  // Occupancy-based.
  inline Config3D GetKernelConfig2D(int xdim, int ydim, const void *kernel,
                                    size_t dynamic_shared_memory_size, 
                                    int block_size_limit) const {
    return GetKernelConfig3D(xdim, ydim, 1, kernel,
                             dynamic_shared_memory_size,
                             block_size_limit);
  }

  // Occupancy-based.
  Config3D GetKernelConfig3D(int xdim, int ydim, int zdim, const void *kernel,
                             size_t dynamic_shared_memory_size,
                             int block_size_limit) const;

private:
  Config1D config_1d_;
  Config3D config_3d_;

  Device *device_;
};

}	//namespace cux
#endif //CUX_LAUNCH_CONFIG_H_
