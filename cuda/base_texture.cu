/*!
* \brief Record the basic usage of Texture Memory.
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

// 1D texture
// <class T, int texType = cudaTextureType1D, 
//  enum cudaTextureReadMode mode = cudaReadModeElementType>.
texture<float> gTex1Df;
__global__ void CopyTexture1DKernel(float *dst, int len) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < len; i += blockDim.x * gridDim.x) {
    dst[i] = tex1Dfetch(gTex1Df, i);
  }
}

// 2D texture
texture<float, cudaTextureType2D> gTex2Df;
__global__ void CopyTexture2DKernel(float *dst, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int i = y * blockDim.x * gridDim.x + x;
    dst[i] = tex2D(gTex2Df, x, y);
  }
}

int main() {
  int height = 18;
  int width = 18;
  int num = height * width;
  // Print the vector length to be used, and compute its size
  size_t size = num * sizeof(float);
  
  // Allocate space on the host side.
  float *h_in = (float *)malloc(size);
  float *h_out = (float *)malloc(size);
  if (h_in == NULL || h_out == NULL) {
    printf("Failed to allocate the memory in host!\n");
    return 1;
  }

  // Initialize
  for (int i = 0; i < num; i++) {
    h_in[i] = rand() / (float)RAND_MAX;
  }

  // Show the result of initialization.
  printf(" Original input: \n");
  for (int i = 0; i < num; i++) {
    printf("%f, ", h_in[i]);
  }

  // Allocate space on the device side.
  float *d_out = NULL;
  CUDA_CHECK(cudaMalloc((void **)&d_out, size));

  /// Test 1D texture.
  { 
    float *d_normal = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_normal, size));
    CUDA_CHECK(cudaMemcpy(d_normal, h_in, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaBindTexture(NULL, gTex1Df, d_normal, size)); // Bind

    int threads_per_block = 256;
    int blocks_per_grid = (height * width + threads_per_block - 1) / threads_per_block;
    CopyTexture1DKernel << <blocks_per_grid, threads_per_block >> > (d_out, num);

    memset(h_out, 0, size);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaUnbindTexture(gTex1Df)); // Unbind

    // Verify.
    printf("\n The h_out from 1D texture:\n");
    for (int i = 0; i < num; i++) {
      printf("%f, ", h_out[i]);
    }
    CUDA_CHECK(cudaFree(d_normal));
  }
  /// Finish test 1D texture.

  /// Test 2D texture.
  {
    // The 2D texture needs to be bound to the cudaArray.
    cudaArray *d_array;
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
    CUDA_CHECK(cudaMallocArray(&d_array, &channel_desc, width, height));
    CUDA_CHECK(cudaMemcpyToArray(d_array, 0, 0, h_in, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaBindTextureToArray(gTex2Df, d_array)); // Bind

    const int block_size = 16;
    dim3 threads_per_block(block_size, block_size);
    dim3 blocks_per_grid(width / threads_per_block.x, height / threads_per_block.y);
    CopyTexture2DKernel << <blocks_per_grid, threads_per_block >> > (d_out, width, height);

    memset(h_out, 0, size);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaUnbindTexture(gTex1Df)); // Unbind

    // Verify.
    printf("\n The h_out from 2D texture:\n");
    for (int i = 0; i < num; i++) {
      printf("%f, ", h_out[i]);
    }

    CUDA_CHECK(cudaFreeArray(d_array));
  }
  /// Finish test 1D texture.

  // Free host memory
  free(h_in);
  free(h_out);
  // Free device memory
  CUDA_CHECK(cudaFree(d_out));

  CUDA_CHECK(cudaDeviceReset());
  return 0;
}