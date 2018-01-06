/*!
 * \brief Vector addition: C = A + B. 
 */

#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

// Kernel
__global__ void VectorAddKernel(const float *A, const float *B,
                                float *C, int num) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < num) {
    C[i] = A[i] + B[i];
  }
}

int VectorAdd(const float *h_a, const float *h_b, const int num, float *h_c) {
  size_t size = num * sizeof(float);
  // Allocate the device input vector
  cudaError_t cuda_err;
  cudaError_t cuda_err_a, cuda_err_b, cuda_err_c;
  float *d_a = NULL;
  float *d_b = NULL;
  float *d_c = NULL;

  cuda_err_a = cudaMalloc((void **)&d_a, size);
  cuda_err_b = cudaMalloc((void **)&d_b, size);
  cuda_err_c = cudaMalloc((void **)&d_c, size);
  if (cuda_err_a != cudaSuccess || cuda_err_b != cudaSuccess || cuda_err_c != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector (error code %s)!\n");
    return 1;
  }

  // Copy the host input vectors in host memory to the device input vectors in
  // device memory
  printf("Copy input data from the host memory to the CUDA device\n");
  cuda_err_a = cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cuda_err_b = cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
  if (cuda_err_a != cudaSuccess || cuda_err_a != cudaSuccess) {
    fprintf(stderr, "Failed to copy vector from host to device (error code <%s>, <%s>)!\n", 
      cudaGetErrorString(cuda_err_a), cudaGetErrorString(cuda_err_b));
    return 1;
  }

  // Launch the Vector Add CUDA Kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (num + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
  VectorAddKernel << <blocksPerGrid, threadsPerBlock >> > (d_a, d_b, d_c, num);
  cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(cuda_err));
    return 1;
  }

  // Copy the device result vector in device memory to the host result vector
  // in host memory.
  printf("Copy output data from the CUDA device to the host memory\n");
  cuda_err_c = cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

  if (cuda_err_c != cudaSuccess) {
    fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(cuda_err_c));
    return 1;
  }

  // Verify that the result vector is correct
  for (int i = 0; i < num; ++i) {
    if (fabs(h_a[i] + h_b[i] - h_c[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      return 1;
    }
  }

  // Free device global memory
  cuda_err_a = cudaFree(d_a);
  cuda_err_b = cudaFree(d_b);
  cuda_err_c = cudaFree(d_c);
  if (cuda_err_a != cudaSuccess || cuda_err_b != cudaSuccess || cuda_err_c != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector (error code <%s>, <%s>, <%s>)!\n", 
      cudaGetErrorString(cuda_err_a), 
      cudaGetErrorString(cuda_err_b),
      cudaGetErrorString(cuda_err_c));
    return 1;
  }

  // Reset the device and exit
  // cudaDeviceReset causes the driver to clean up all state. While
  // not mandatory in normal operation, it is good practice.  It is also
  // needed to ensure correct operation when the application is being
  // profiled. Calling cudaDeviceReset causes all profile data to be
  // flushed before the application exits
  cuda_err = cudaDeviceReset();
  if (cuda_err != cudaSuccess) {
    fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(cuda_err));
    return 1;
  }

  printf("Done\n");
  return 0;
}

int main(void) {
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
}
