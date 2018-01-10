/*!
* \brief gemm: C = A * B.
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

// Initialize the input data.
void GenMatrix(const int height, const int width, float *mat) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      mat[i*width + j] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX*RAND_MAX);
    }
  }
}

// Just for checking the result.
float GetMean(const float* mat, const int height, const int width) {
  int num = height * width;
  float total = 0;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      total += mat[i*width + j];
    }
  }
  return total / num;
}

// Just for checking the result too.
void MatrixPrint(const float* mat, const int height, const int width) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      std::cout << mat[i*width + j] << ",";
    }
    std::cout << std::endl;
  }
}

// Normal version in cpu as a reference
void MatrixMulCPU(const int M, const int N, const int K, const float ALPHA,
  const float *A, const int lda,
  const float *B, const int ldb,
  float *C, const int ldc) {
  int i, j, k;
  memset(C, 0, sizeof(float) * ldc * M);
  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      register float A_PART = ALPHA*A[i*lda + k];
      for (j = 0; j < N; ++j) {
        C[i*ldc + j] += A_PART*B[k*ldb + j];
      }
    }
  }
}

// CUDA version 1.
__global__ void MatrixMulKernelv1(const int M, const int N, const int K, const float ALPHA,
  const float *A, const int lda,
  const float *B, const int ldb,
  float *C, const int ldc) {

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M; i += blockDim.x * gridDim.x) {
    for (int ni = 0; ni < N; ++ni)
      C[i*N + ni] = 0;
    for (int k = 0; k < K; ++k) {
      register float A_PART = ALPHA*A[i*lda + k];
      for (int j = 0; j < N; ++j) {
        C[i*ldc + j] += A_PART*B[k*ldb + j];
      }
    }
  }
}

void MatrixMulCUDAv1(const int M, const int N, const int K, const float ALPHA,
  const float *A, const int lda,
  const float *B, const int ldb,
  float *C, const int ldc) {
  int threads_per_block = 512;
  const int blocks_per_grid = (M + threads_per_block - 1) / threads_per_block;
  MatrixMulKernelv1 << <blocks_per_grid, threads_per_block >> >
    (M, N, K, 1.0, A, lda, B, ldb, C, ldc);
}


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

int main() {
  InitEnvironment(0);

  int height_a = 500, width_a = 405;
  int height_b = 405, width_b = 120;
  if (width_a != height_b) {
    printf("width_a should be equal to height_b.\n");
    return 1;
  }

  const int mem_size_a = sizeof(float) * height_a * width_a;
  const int mem_size_b = sizeof(float) * height_b * width_b;
  const int mem_size_c = sizeof(float) * height_a * width_b;

  float *h_a = (float *)malloc(mem_size_a);
  float *h_b = (float *)malloc(mem_size_b);
  float *h_c = (float *)malloc(mem_size_c);
  if (h_a == NULL || h_b == NULL || h_c == NULL) {
    printf("Fail to malloc.\n");
    return 1;
  }

  // Initialize 
  GenMatrix(height_a, width_a, h_a);
  GenMatrix(height_b, width_b, h_b);

  // CPU
  MatrixMulCPU(height_a, width_b, width_a, 1.0, h_a, width_a,h_b, width_b, h_c, width_b);
  printf("mean in cpu version = %f\n", GetMean(h_c, height_a, width_b));

  // GPU
  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc((void **)&d_a, mem_size_a));
  CUDA_CHECK(cudaMalloc((void **)&d_b, mem_size_b));
  CUDA_CHECK(cudaMalloc((void **)&d_c, mem_size_c));

  // Copy host memory to device
  CUDA_CHECK(cudaMemcpy(d_a, h_a, mem_size_a, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, mem_size_b, cudaMemcpyHostToDevice));

  MatrixMulCUDAv1(height_a, width_b, width_a, 1.0, d_a, width_a, d_b, width_b, d_c, width_b);

  CUDA_CHECK(cudaMemcpy(h_c, d_c, mem_size_c, cudaMemcpyDeviceToHost));
  printf("mean in cpu version = %f\n", GetMean(h_c, height_a, width_b));

  free(h_a);
  free(h_b);
  free(h_c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
