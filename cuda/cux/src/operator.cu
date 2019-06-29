/*!
* \brief Operator.
*/

#include "operator.h"

namespace cux {

//////////////////////////
// PerformanceEvaluator
void PerformanceEvaluator::GetSuggestedLayout(
  const void *kernel, const int count,
  const int dynamic_smem_usage,
  int &grid_size, int &block_size) {

  int min_grid_size;
  // This function needs to be placed in a cu file and compiled by NVCC, 
  // otherwise an "undefined" error message will appear.
  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
    &min_grid_size,
    &block_size,
    kernel,
    dynamic_smem_usage,
    count));

  // Round up.
  grid_size = (count + block_size - 1) / block_size;
}

} // cux.
