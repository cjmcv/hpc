/*!
* \brief LaunchConfig.
*/

#include "util/launch_config.h"

namespace cux {

//////////////////////////
// LaunchConfig
Config1D LaunchConfig::GetKernelConfig1D(int element_count, const void *kernel,
                                         size_t dynamic_smem_usage,
                                         int block_size_limit) const {
  Config1D config;
  config.virtual_thread_count = element_count;

  int min_grid_size;
  // This function needs to be placed in a cu file and compiled by NVCC, 
  // otherwise an "undefined" error message will appear.
  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
    &min_grid_size,             // Suggested min grid size to achieve a full machine launch.
    &config.thread_per_block,   // Suggested block size to achieve maximum occupancy.
    kernel, 
    dynamic_smem_usage,         // Size of dynamically allocated shared memory. 
    block_size_limit));         // Maximum size for each block. 
                                // In the case of 1D kernels, it can coincide with the number of input elements.
  // Round up.
  config.block_per_grid = std::min(min_grid_size, DivUp(element_count, config.thread_per_block));
  return config;
}

Config3D LaunchConfig::GetKernelConfig3D(int xdim, int ydim, int zdim, const void *kernel,
                                         size_t dynamic_shared_memory_size, 
                                         int block_size_limit) const {
  Config3D config;
  if (xdim <= 0 || ydim <= 0 || zdim <= 0) {
    return config;
  }

  int dev;
  cudaGetDevice(&dev);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  int xthreadlimit = deviceProp.maxThreadsDim[0];
  int ythreadlimit = deviceProp.maxThreadsDim[1];
  int zthreadlimit = deviceProp.maxThreadsDim[2];
  int xgridlimit = deviceProp.maxGridSize[0];
  int ygridlimit = deviceProp.maxGridSize[1];
  int zgridlimit = deviceProp.maxGridSize[2];

  int block_count = 0;
  int thread_per_block = 0;
  cudaError_t err = cudaOccupancyMaxPotentialBlockSize(
    &block_count, &thread_per_block, kernel, dynamic_shared_memory_size,
    block_size_limit);
  //CHECK_EQ(err, cudaSuccess);

  int threadsx = std::min({ xdim, thread_per_block, xthreadlimit });
  int threadsy =
    std::min({ ydim, std::max(thread_per_block / threadsx, 1), ythreadlimit });
  int threadsz =
    std::min({ zdim, std::max(thread_per_block / (threadsx * threadsy), 1),
      zthreadlimit });

  int blocksx = std::min({ block_count, DivUp(xdim, threadsx), xgridlimit });
  int blocksy = std::min(
  { DivUp(block_count, blocksx), DivUp(ydim, threadsy), ygridlimit });
  int blocksz = std::min({ DivUp(block_count, (blocksx * blocksy)),
    DivUp(zdim, threadsz), zgridlimit });

  config.virtual_thread_count = dim3(xdim, ydim, zdim);
  config.thread_per_block = dim3(threadsx, threadsy, threadsz);
  config.block_per_grid = dim3(blocksx, blocksy, blocksz);
  return config;
}
} // cux.
