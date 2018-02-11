/*!
* \brief An experiment on memory access.
* \reference https://www.cnblogs.com/1024incn/p/4573566.html
*/

#include <iostream>
#include <time.h>

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

int InitEnvironment(const int dev_id) {
  CUDA_CHECK(cudaSetDevice(dev_id));
  cudaDeviceProp device_prop;
  cudaError_t error = cudaGetDeviceProperties(&device_prop, dev_id);
  if (device_prop.computeMode == cudaComputeModeProhibited) {
    fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
    return 1;
  }
  if (error != cudaSuccess) {
    printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
  }
  else {
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", dev_id, device_prop.name, device_prop.major, device_prop.minor);
  }
  return 0;
}

// Initialize the input data.
void GenArray(float *arr, const int len) {
  for (int i = 0; i < len; i++) {
    arr[i] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX*RAND_MAX);
  }
}

__global__ void TestMisalignedReadKernel(float *A, float *B, float *C, const int len, int offset) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int k = i + offset;
  if (k < len)
    C[i] = A[k] + B[k];
}

__global__ void TestMisalignedWriteKernel(float *A, float *B, float *C, const int len, int offset) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int k = i + offset;
  if (k < len)
    C[k] = A[i] + B[i];  // Just switch the place between i and k.
}

int main() {
  int dev_id = 0;
  InitEnvironment(dev_id);

  // set up array size
  int len = 1 << 20; // total number of elements to reduce
  printf("len = %d\n", len);
  size_t mem_size = len * sizeof(float);

  // execution configuration
  const int threads_per_block = 512;
  const int blocks_per_grid = (len + threads_per_block - 1) / threads_per_block;

  // Allocate host memory
  float *h_a = (float *)malloc(mem_size);
  float *h_b = (float *)malloc(mem_size);
  float *h_c = (float *)malloc(mem_size);
  // Initialize host array
  GenArray(h_a, len);
  GenArray(h_b, len);

  // Allocate device memory
  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc((float**)&d_a, mem_size));
  CUDA_CHECK(cudaMalloc((float**)&d_b, mem_size));
  CUDA_CHECK(cudaMalloc((float**)&d_c, mem_size));
  // Copy data from host to device
  CUDA_CHECK(cudaMemcpy(d_a, h_a, mem_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, mem_size, cudaMemcpyHostToDevice));

  // Warm up
  TestMisalignedReadKernel << < blocks_per_grid, threads_per_block >> > (d_a, d_b, d_c, len, 256);
  
  // Test.
  float msec_total = 0.0f;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  const int offset_size = 12;
  int offset[12] = {1, 16, 129, 254, 513, 1020, 0, 32, 128, 160, 512, 1024};
  for (int i = 0; i < offset_size; i++) {
    CUDA_CHECK(cudaEventRecord(start, NULL));
    for(int tc = 0; tc < 10; tc++)
      TestMisalignedReadKernel << < blocks_per_grid, threads_per_block >> > (d_a, d_b, d_c, len, offset[i]);
    CUDA_CHECK(cudaEventRecord(stop, NULL));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&msec_total, start, stop));
    printf("Read<<< %4d, %4d >>>, offset %4d, elapsed %f sec, offset%32 == 0 : %d\n",
      blocks_per_grid, threads_per_block,
      offset[i], msec_total, offset[i]%32 == 0);
    if (i == (offset_size-1) / 2)
      printf("\n");
  }

  printf("-------------------------------------------------------------------------------\n");

  for (int i = 0; i < offset_size; i++) {
    CUDA_CHECK(cudaEventRecord(start, NULL));
    for (int tc = 0; tc < 10; tc++)
      TestMisalignedWriteKernel << < blocks_per_grid, threads_per_block >> > (d_a, d_b, d_c, len, offset[i]);
    CUDA_CHECK(cudaEventRecord(stop, NULL));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&msec_total, start, stop));
    printf("Write<<< %4d, %4d >>>, offset %4d, elapsed %f sec, offset%32 == 0 : %d\n",
      blocks_per_grid, threads_per_block,
      offset[i], msec_total, offset[i] % 32 == 0);
    if (i == (offset_size-1) / 2)
      printf("\n");
  }

  // Copy kernel result back to host side and check device results
  CUDA_CHECK(cudaMemcpy(h_c, d_c, mem_size, cudaMemcpyDeviceToHost));

  // Free host and device memory
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  free(h_a);
  free(h_b);
  free(h_c);

  // Reset device
  CUDA_CHECK(cudaDeviceReset());
  return 0;
}