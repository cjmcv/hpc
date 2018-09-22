/*!
* \brief Simple demonstration of DeviceScan::ExclusiveSum.
*/
#include "cuda_util.h"

#include <cub/util_allocator.cuh>
#include <cub/device/device_scan.cuh>

using namespace cub;


CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

void Initialize(int *h_in, int num_items) {
  for (int i = 0; i < num_items; ++i)
    h_in[i] = 1;
}

// CPU version.
// Solve exclusive-scan problem.
int SolveInCPU(int *h_in, int *h_reference, int num_items) {
  int sum = 0;
  for (int i = 0; i < num_items; ++i) {
    h_reference[i] = sum;
    sum += h_in[i];
  }
  printf("%d, %d\n", sum, h_reference[num_items - 1]);
  return sum;
}

int main(int argc, char** argv) {

  int ret = cjmcv_cuda_util::InitEnvironment(0);
  if (ret != 0) {
    printf("Failed to initialize the environment for cuda.");
    return -1;
  }

  int num_items = 3;

  printf("cub::DeviceScan::ExclusiveSum %d items (%d-byte elements)\n",
    num_items, (int) sizeof(int));

  // Allocate host arrays
  int *h_in = new int[num_items];
  int *h_reference = new int[num_items];

  // Initialize problem and solution
  Initialize(h_in, num_items);
  SolveInCPU(h_in, h_reference, num_items);

  // Allocate problem device arrays
  int *d_in = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(int) * num_items));

  // Initialize device input
  CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(int) * num_items, cudaMemcpyHostToDevice));

  // Allocate device output array
  int *d_out = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(int) * num_items));

  // Allocate temporary storage
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items));
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  // Run
  CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items));

  int *h_reference4device = new int[num_items];
  CUDA_CHECK(cudaMemcpy(h_reference4device, d_out, sizeof(int) * num_items, cudaMemcpyDeviceToHost));

  bool is_equal = true;
  for (int i = 0; i < num_items; i++) {
    if (h_reference4device[i] != h_reference[i]) {
      is_equal = false;
      break;
    }
  }
  printf("The result is equal or not: %d -> (%d vs %d)\n", is_equal, 
    h_reference[num_items - 1], h_reference4device[num_items - 1]);

  // Cleanup
  if (h_in) delete[] h_in;
  if (h_reference) delete[] h_reference;
  if (h_reference4device) delete[] h_reference4device;
  if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
  if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
  if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

  cjmcv_cuda_util::CleanUpEnvironment();

  return 0;
}



