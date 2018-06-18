/*!
 * \brief Vector addition: C = A + B. 
 */

#include <iostream>
#include "util.h"

// Kernel
__global__ void VectorAddKernel(const float *A, const float *B,
                                float *C, int num) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < num) {
    C[i] = A[i] + B[i];
  }
}

int VectorAdd(const float *h_a, const float *h_b, const int num, float *h_c) {
  cjmcv_cuda_util::GpuTimer timer;

  size_t size = num * sizeof(float);
  // Allocate the device input vector
  float *d_a = NULL;
  float *d_b = NULL;
  float *d_c = NULL;

  CUDA_CHECK(cudaMalloc((void **)&d_a, size));
  CUDA_CHECK(cudaMalloc((void **)&d_b, size));
  CUDA_CHECK(cudaMalloc((void **)&d_c, size));

  // Copy the host input vectors in host memory to the device input vectors in
  // device memory
  printf("Copy input data from the host memory to the CUDA device\n");
  CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

  // Launch the Vector Add CUDA Kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (num + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
  timer.Start();
  VectorAddKernel << <blocksPerGrid, threadsPerBlock >> > (d_a, d_b, d_c, num);
  timer.Stop();
  printf("Timer: %f ms.\n", timer.ElapsedMillis());

  CUDA_CHECK(cudaGetLastError());

  // Copy the device result vector in device memory to the host result vector
  // in host memory.
  printf("Copy output data from the CUDA device to the host memory\n");
  CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

  // Verify that the result vector is correct
  for (int i = 0; i < num; ++i) {
    if (fabs(h_a[i] + h_b[i] - h_c[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      return 1;
    }
  }

  // Free device global memory
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  printf("Done\n");
  return 0;
}

int main(void) {
  using namespace cjmcv_cuda_util;

  InitEnvironment(0);

  int num = 50000;
  // Print the vector length to be used, and compute its size
  size_t size = num * sizeof(float);

  // Allocate the host input vector
  float *h_a = (float *)malloc(size);
  float *h_b = (float *)malloc(size);
  float *h_c = (float *)malloc(size);
  if (h_a == NULL || h_b == NULL || h_c == NULL) {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    return 1;
  }

  // Initialize
  for (int i = 0; i < num; ++i) {
    h_a[i] = rand() / (float)RAND_MAX;
    h_b[i] = rand() / (float)RAND_MAX;
  }

  VectorAdd(h_a, h_b, num, h_c);

  // Free host memory
  free(h_a);
  free(h_b);
  free(h_c);

  CleanUpEnvironment();
}
