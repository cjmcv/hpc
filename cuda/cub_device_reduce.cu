/*!
* \brief Simple demonstration of DeviceScan::Sum.
*        (It is slower than cpu version ?)
*/

#include "cuda_util.h"
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>

//Initialize problem.
void Initialize(int *h_in, int num_items) {
  for (int i = 0; i < num_items; ++i)
    h_in[i] = i;
}

// Compute solution.
void SolveInCPU(int *h_in, int &h_out, int num_items) {
  h_out = 0;
  for (int i = 0; i < num_items; ++i) {
    h_out += h_in[i];
  }
}

int main(int argc, char** argv) {
  int num_items = 1500;

  int ret = cjmcv_cuda_util::InitEnvironment(0);
  if (ret != 0) {
    printf("Failed to initialize the environment for cuda.");
    return -1;
  }

  /*
  *  CPU version. 0 ms.
  */
  // Allocate host arrays
  int* h_in = new int[num_items];
  int h_out;

  // Initialize problem and solution
  Initialize(h_in, num_items);
  time_t t = clock();
  SolveInCPU(h_in, h_out, num_items);
  printf("Time spent in running in cpu: %lld ms\n", clock() - t);
  
  /*
  *  Device version. 4 ms.
  */
  cjmcv_cuda_util::GpuTimer *gpu_timer = new cjmcv_cuda_util::GpuTimer;
  gpu_timer->Start();
  // Initialize device input
  int *d_in = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_in, sizeof(int) * num_items));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, sizeof(int) * num_items, cudaMemcpyHostToDevice));

  // Allocate device output array
  int *d_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_out, sizeof(int) * 1));

  // Request and allocate temporary storage
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  CubDebugExit(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items));
  CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  gpu_timer->Stop();
  printf("Request and allocate temporary storage: %f ms.\n", gpu_timer->ElapsedMillis());

  // Run
  gpu_timer->Start();
  CubDebugExit(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items));
  gpu_timer->Stop();
  printf("Run: %f ms.\n", gpu_timer->ElapsedMillis());

  // Get the output from device.
  int h_out4cub = 0;
  CUDA_CHECK(cudaMemcpy(&h_out4cub, d_out, sizeof(int) * 1, cudaMemcpyDeviceToHost));
  printf("Result: (h_out vs h_out4cub) == (%d vs %d)\n", h_out, h_out4cub);

  // Cleanup
  if (gpu_timer) delete gpu_timer;
  if (h_in) delete[] h_in;
  if (d_in) CUDA_CHECK(cudaFree(d_in));
  if (d_out) CUDA_CHECK(cudaFree(d_out));
  if (d_temp_storage) CUDA_CHECK(cudaFree(d_temp_storage));

  cjmcv_cuda_util::CleanUpEnvironment();

  printf("\n\n");
  return 0;
}



