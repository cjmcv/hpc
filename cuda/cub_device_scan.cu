/*!
* \brief Simple demonstration of DeviceScan::ExclusiveSum.
*        (It is slower than cpu version ?)
*/
#include "cuda_util.h"

#include <cub/util_allocator.cuh>
#include <cub/device/device_scan.cuh>
  
// Caching allocator for device memory
cub::CachingDeviceAllocator  g_allocator(true);

void Initialize(int *h_in, int num_items) {
  for (int i = 0; i < num_items; ++i)
    h_in[i] = i;
}

// CPU version.
// Solve exclusive-scan problem.
void ExclusiveScanCPU(int *h_in, int *h_out, int num_items) {
  int sum = 0;
  for (int i = 0; i < num_items; ++i) {
    h_out[i] = sum;
    sum += h_in[i];
  }
}

int main(int argc, char** argv) {

  int ret = cjmcv_cuda_util::InitEnvironment(0);
  if (ret != 0) {
    printf("Failed to initialize the environment for cuda.");
    return -1;
  }

  int num_items = 1234567;

  printf("cub::DeviceScan::ExclusiveSum %d items (%d-byte elements)\n",
    num_items, (int) sizeof(int));

  /*
  *  CPU version. 4ms.
  */
  // Allocate host arrays
  int *h_in = new int[num_items];
  int *h_out = new int[num_items];

  // Initialize problem and solution
  Initialize(h_in, num_items);
  time_t t = clock();
  ExclusiveScanCPU(h_in, h_out, num_items);
  printf("Time spent in running in cpu: %lld ms\n", clock() - t);

  /*
  *  Device version. 24ms.
  */
  cjmcv_cuda_util::GpuTimer *gpu_timer = new cjmcv_cuda_util::GpuTimer;
  gpu_timer->Start();
  // Initialize device input
  int *d_in = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(int) * num_items));
  CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(int) * num_items, cudaMemcpyHostToDevice));

  // Allocate device output array
  int *d_out = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(int) * num_items));

  // Allocate temporary storage
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items));
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
  gpu_timer->Stop();
  printf("Time spent in creating memory for device: %f ms.\n", gpu_timer->ElapsedMillis());
  
  // Run
  gpu_timer->Start();
  CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items));
  gpu_timer->Stop();
  printf("Time spent in running: %f ms.\n", gpu_timer->ElapsedMillis());

  // Get the output from device.
  int *h_out4cub = new int[num_items];
  CUDA_CHECK(cudaMemcpy(h_out4cub, d_out, sizeof(int) * num_items, cudaMemcpyDeviceToHost));

  // Check the answer.
  bool is_equal = true;
  for (int i = 0; i < num_items; i++) {
    if (h_out4cub[i] != h_out[i]) {
      is_equal = false;
      break;
    }
  }
  printf("The result is equal or not: %d -> (%d vs %d)\n", is_equal, 
    h_out[num_items - 1], h_out4cub[num_items - 1]);

  // Cleanup
  if (gpu_timer) delete gpu_timer;
  if (h_in) delete[] h_in;
  if (h_out) delete[] h_out;
  if (h_out4cub) delete[] h_out4cub;
  if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
  if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
  if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

  cjmcv_cuda_util::CleanUpEnvironment();

  return 0;
}



