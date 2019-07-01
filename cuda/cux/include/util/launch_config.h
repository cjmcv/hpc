/*!
* \brief LaunchConfig.
*/

#ifndef CUX_LAUNCH_CONFIG_H_
#define CUX_LAUNCH_CONFIG_H_

#include <iostream>
#include <string>
#include <map>

#include "util.h"

namespace cux { 

class LaunchConfig {
public:
  // TODO: Delete the Initialize and assign in constructor.
  void Initialize() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop_, device));
  }

  // occupancy = (double)active_warps / max_warps;
  void GetPotentialOccupancy(const void *kernel, const int block_size,
    const size_t dynamic_shared_mem,
    int &active_blocks, double &occupancy) {

    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, kernel, block_size, dynamic_shared_mem));

    int active_warps = active_blocks * block_size / prop_.warpSize;
    int max_warps = prop_.maxThreadsPerMultiProcessor / prop_.warpSize;

    occupancy = (double)active_warps / max_warps;
  }

  // It suggests a block size that achieves the best theoretical occupancy.
  // But the occupancy can not be translated directly to performance.
  static void GetSuggestedLayout(const void *kernel, const int count,
    const int dynamic_smem_usage,
    int &grid_size, int &block_size);

private:
  cudaDeviceProp prop_;
};

}	//namespace cux
#endif //CUX_LAUNCH_CONFIG_H_
