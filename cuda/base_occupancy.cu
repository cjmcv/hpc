/*!
* \brief Record the basic usage of cudaOccupancyMaxPotentialBlockSize.
*/

#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      fprintf(stderr, "CUDA_CHECK error in line %d of file %s \
              : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
      exit(EXIT_FAILURE); \
    } \
  } while(0);

// A simple kernel that can be called with any execution configuration.
__global__ void square(int *arr, int len) {
  extern __shared__ int smem[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < len) {
    arr[idx] *= arr[idx];
  }
}

// Run with automatically configured launch.
// It suggests a block size that achieves the best theoretical occupancy. 
// But the occupancy can not be translated directly to performance.
int Run(const int count, int *d_arr) {
  int block_size;
  int min_grid_size;
  int grid_size;
  size_t dynamic_smem_usage = 0;

  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
    &min_grid_size,
    &block_size,
    (void*)square,
    dynamic_smem_usage,
    count));

  printf("Minimum grid size for maximum occupancy: %d \n", min_grid_size);
  printf("Suggested block size: %d \n", block_size);
  printf("Blocksize to dynamic shared memory size: %d \n", dynamic_smem_usage);

  // Round up.
  grid_size = (count + block_size - 1) / block_size;

  // Launch and profile.
  square << <grid_size, block_size, dynamic_smem_usage >> > (d_arr, count);

  return 0;
}

int main() {
  const int count = 100000;
  int size = count * sizeof(int);

  // Initialize.
  int *h_arr;
  h_arr = new int[count];
  for (int i = 0; i < count; i += 1) {
    h_arr[i] = i;
  }

  // To prepare data in device.
  int *d_arr;
  CUDA_CHECK(cudaMalloc(&d_arr, size));
  CUDA_CHECK(cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice));

  // The key function.
  Run(count, d_arr);

  // Clear for storing the calculation result.
  for (int i = 0; i < count; i += 1) {
    h_arr[i] = 0;
  }
  // Verify the return data.
  CUDA_CHECK(cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost));
  for (int i = 0; i < count; i += 1) {
    if (h_arr[i] != i * i) {
      printf("index: %d, expected: %d, actual: %d", i, i*i, h_arr[i]);
      return 1;
    }
  }

  // Free.
  delete[] h_arr;  
  CUDA_CHECK(cudaFree(d_arr));  
  CUDA_CHECK(cudaDeviceReset());
  return 0;
}
