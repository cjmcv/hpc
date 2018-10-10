/*!
* \brief gemm: C = A * B.
*        Use cublas with half.
*/

#include "cuda_util.h"
#include <cublas_v2.h>

// Initialize the input data.
void GenMatrix(const int height, const int width, float *mat) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      mat[i*width + j] = 1;//(float)rand() / RAND_MAX + (float)rand() / (RAND_MAX*RAND_MAX);
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

// CPU version
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

float GemmWithCublas(cublasHandle_t cublas_handle, const bool TransA,
  const bool TransB, const int M, const int N, const int K,
  const float alpha, const float* A, const float* B, const float beta,
  float* C) {
  using namespace cjmcv_cuda_util;

  GpuTimer gpu_timer;

  // Note that cublas follows fortran order.
  int lda = (TransA == false) ? K : M;
  int ldb = (TransB == false) ? N : K;
  cublasOperation_t cuTransA =
    (TransA == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
    (TransB == false) ? CUBLAS_OP_N : CUBLAS_OP_T;

  // Warm up.
  if (cublasSgemm(cublas_handle, cuTransB, cuTransA,
    N, M, K, &alpha, B, ldb, A, lda, &beta, C, N) != CUBLAS_STATUS_SUCCESS) {
    printf("cublasSgemm error.\n");
  }

  gpu_timer.Start();
  if (cublasSgemm(cublas_handle, cuTransB, cuTransA,
    N, M, K, &alpha, B, ldb, A, lda, &beta, C, N) != CUBLAS_STATUS_SUCCESS) {
    printf("cublasSgemm error.\n");
  }
  gpu_timer.Stop();

  return gpu_timer.ElapsedMillis();;
}


__global__ void CvtArrayFloat2Half(const float* in, const int len, half *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < len;
    i += blockDim.x * gridDim.x) {
    out[i] = __float2half(in[i]);
    //printf("o[%d] = %d, ", out[i]);
  }
}

__global__ void CvtArrayHalf2Float(const half* in, const int len, float *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < len;
    i += blockDim.x * gridDim.x) {
    out[i] = __half2float(in[i]);
  }
}

// Even slower than using float in GTX660.
float GemmWithCublasFloat16(cublasHandle_t cublas_handle, const bool TransA,
  const bool TransB, const int M, const int N, const int K,
  const float alpha, const float* A, const float* B, const float beta,
  float* C) {
  using namespace cjmcv_cuda_util;

  // Convert float to half.
  half *A_half, *B_half, *C_half;
  CUDA_CHECK(cudaMalloc((void **)&A_half, sizeof(half) * M * K));
  CUDA_CHECK(cudaMalloc((void **)&B_half, sizeof(half) * K * N));
  CUDA_CHECK(cudaMalloc((void **)&C_half, sizeof(half) * M * N));

  const int threads_per_block = 1024;
  int blocks_per_grid = (M*K + threads_per_block - 1) / threads_per_block;
  CvtArrayFloat2Half << <blocks_per_grid, threads_per_block >> >
    (A, M*K, A_half);

  blocks_per_grid = (K*N + threads_per_block - 1) / threads_per_block;
  CvtArrayFloat2Half << <blocks_per_grid, threads_per_block >> >
    (B, K*N, B_half);

  // Time counter.
  GpuTimer gpu_timer;

  // Note that cublas follows fortran order.
  int lda = (TransA == false) ? K : M;
  int ldb = (TransB == false) ? N : K;
  cublasOperation_t cuTransA =
    (TransA == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
    (TransB == false) ? CUBLAS_OP_N : CUBLAS_OP_T;

  float alpha_f = float(alpha);
  float beta_f = float(beta);
  cudaDataType_t half_datatype = CUDA_R_16F;

  // Warm up.
  if (cublasSgemmEx(cublas_handle, cuTransB, cuTransA,
    N, M, K, &alpha_f, B_half, half_datatype, ldb,
    A_half, half_datatype, lda, &beta_f,
    C_half, half_datatype, N) != CUBLAS_STATUS_SUCCESS) {
    printf("cublasHgemm error.\n");
  }

  gpu_timer.Start();
  if (cublasSgemmEx(cublas_handle, cuTransB, cuTransA,
    N, M, K, &alpha_f, B_half, half_datatype, ldb,
    A_half, half_datatype, lda, &beta_f, 
    C_half, half_datatype, N) != CUBLAS_STATUS_SUCCESS) {
    printf("cublasHgemm error.\n");
  }
  gpu_timer.Stop();

  // Convert half back to float.
  blocks_per_grid = (K*N + threads_per_block - 1) / threads_per_block;
  CvtArrayHalf2Float << <blocks_per_grid, threads_per_block >> >
    (C_half, M*N, C);

  CUDA_CHECK(cudaFree(A_half));
  CUDA_CHECK(cudaFree(B_half));
  CUDA_CHECK(cudaFree(C_half));

  return gpu_timer.ElapsedMillis();
}

int main() {
  int dev_id = 0;
  int ret = cjmcv_cuda_util::InitEnvironment(dev_id);
  if (ret != 0) {
    printf("Failed to initialize the environment for cuda.");
    return -1;
  }

  int height_a = 1024, width_a = 800;
  int height_b = 800, width_b = 2048;
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
  srand(0);
  GenMatrix(height_a, width_a, h_a);
  GenMatrix(height_b, width_b, h_b);

  //// CPU
  time_t t = clock();
  MatrixMulCPU(height_a, width_b, width_a, 1.0, h_a, width_a, h_b, width_b, h_c, width_b);
  printf("In cpu version, msec_total = %lld, mean = %f\n", clock() - t, GetMean(h_c, height_a, width_b));
  //MatrixPrint(h_c, height_a, width_b);

  // GPU
  // Allocate memory in host. 
  float msec_total;
  // Create cublas handle.
  cublasHandle_t cublas_handle;
  if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS) {
    printf("Cannot create Cublas handle. Cublas won't be available.");
  }

  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc((void **)&d_a, mem_size_a));
  CUDA_CHECK(cudaMalloc((void **)&d_b, mem_size_b));
  CUDA_CHECK(cudaMalloc((void **)&d_c, mem_size_c));

  // Copy host memory to device
  CUDA_CHECK(cudaMemcpy(d_a, h_a, mem_size_a, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, mem_size_b, cudaMemcpyHostToDevice));
  
  msec_total = GemmWithCublas(cublas_handle, false, false, height_a, width_b, width_a, 1.0, d_a, d_b, 0.0, d_c);
  CUDA_CHECK(cudaMemcpy(h_c, d_c, mem_size_c, cudaMemcpyDeviceToHost));  // Copy memory back to host.
  printf("In gpu version (float), msec_total = %f, mean = %f\n", msec_total, GetMean(h_c, height_a, width_b));

  memset(h_c, 0, mem_size_c);

  msec_total = GemmWithCublasFloat16(cublas_handle, false, false, height_a, width_b, width_a, 1.0, d_a, d_b, 0.0, d_c);
  CUDA_CHECK(cudaMemcpy(h_c, d_c, mem_size_c, cudaMemcpyDeviceToHost));  // Copy memory back to host.
  printf("In gpu version (half), msec_total = %f, mean = %f\n", msec_total, GetMean(h_c, height_a, width_b));

  //MatrixPrint(h_c, height_a, width_b);
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  if (cublasDestroy(cublas_handle) != CUBLAS_STATUS_SUCCESS) {
    printf("Destory Cublas handle Error.");
  }

  free(h_a);
  free(h_b);
  free(h_c);
  cjmcv_cuda_util::CleanUpEnvironment();

  return 0;
}
