/*!
* \brief gemm: C = A * B.
*/
#include "cuda_util.h"

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

// CPU version 1: 1583 ms
// Normal version in cpu as a reference
void MatrixMulCPUv1(const int M, const int N, const int K, const float ALPHA,
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

// CPU version 2: 3389 ms
// Block based matrix multiplication in cpu.
void MatrixMulCPUv2(const int M, const int N, const int K, const float ALPHA,
  const float *A, const int lda,
  const float *B, const int ldb,
  float *C, const int ldc) {
  int bi, bj, bk;
  int i, j, k;
  const int block_size = 32;
  int block_num_M = M / block_size;
  int block_num_N = N / block_size;
  int block_num_K = K / block_size;
  memset(C, 0, sizeof(float) * ldc * M);

  // Loop over all of the blocks.
  for (bi = 0; bi < block_num_M; ++bi) {
    for (bj = 0; bj < block_num_N; ++bj) {
      for (bk = 0; bk < block_num_K; ++bk) {
        // Loop over all of the elements in a block.
        for (i = bi*block_size; i < (bi + 1)*block_size; ++i) {
          for (k = bk*block_size; k < (bk + 1)*block_size; ++k) {
            for (j = bj*block_size; j < (bj + 1)*block_size; ++j) { 
              C[i*ldc + j] += A[i*lda + k] * B[k*ldb + j];
            }
          }
        }
      }
    }
  }
}

// CUDA version 1: 72 ms
// It is rewrited from MatrixMulCPUv2. 
// bi,bj can be replaced by blockIdx.x,blockIdx.y
// i,j can be replaced by threadIdx.x,threadIdx.y
// so just bk and k left. Grid and block is related to the dst matrix.
//
// \ C[ty, tx] = A[ty, k] * B[k, tx]
// for bk -> bk_num_per_grid
//     for k -> k_num_per_block
//         C[bi*bs + ty, bj*bs + tx] = A[bi*bs + ty, bk*bs + k] * B[k*bs + k, bj*bs + tx]
template <int BLOCK_SIZE>
__global__ void MatrixMulKernelv1(const int M, const int N, const int K, const float ALPHA,
  const float *A, const int lda,
  const float *B, const int ldb,
  float *C, const int ldc) {

  float c_sub_acc = 0;
  for (int bk = 0; bk < K / BLOCK_SIZE; bk++) {
    for (int k = 0;k < BLOCK_SIZE; k++) {
      c_sub_acc += A[(blockIdx.y * BLOCK_SIZE + threadIdx.y) * lda + (bk * BLOCK_SIZE + k)] *
        B[(bk * BLOCK_SIZE + k) * ldb + (blockIdx.x * BLOCK_SIZE + threadIdx.x)];
    }
  }

  C[(blockIdx.y * BLOCK_SIZE + threadIdx.y) * ldc + (blockIdx.x * BLOCK_SIZE + threadIdx.x)] += c_sub_acc;
}

// CUDA version 2.
// Use shared memory.
template <int BLOCK_SIZE>
__global__ void MatrixMulKernelv2(const int M, const int N, const int K, const float ALPHA,
  const float *A, const int lda,
  const float *B, const int ldb,
  float *C, const int ldc) {

  __shared__ float a_shared[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float b_shared[BLOCK_SIZE][BLOCK_SIZE];

  float c_sub_acc = 0;
  // For blocks in grid.
  for (int bk = 0; bk < K / BLOCK_SIZE; bk++) {
    a_shared[threadIdx.y][threadIdx.x] = A[(blockIdx.y * BLOCK_SIZE + threadIdx.y) * lda + (bk * BLOCK_SIZE + threadIdx.x)];
    b_shared[threadIdx.y][threadIdx.x] = B[(bk * BLOCK_SIZE + threadIdx.y) * ldb + (blockIdx.x * BLOCK_SIZE + threadIdx.x)];
    // Wait for data to complete loading to Shared memory.
    __syncthreads();

    // For elements in a block.
    for (int k = 0;k < BLOCK_SIZE; k++) {
      c_sub_acc += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
    }
	  // To prevent the case from happening:
	  // The next round of data is loaded when the data in share memory is not used up.
    __syncthreads();
  }

  C[(blockIdx.y * BLOCK_SIZE + threadIdx.y) * ldc + (blockIdx.x * BLOCK_SIZE + threadIdx.x)] += c_sub_acc;
}

//#define TEST_CUDA_V1
float MatrixMulCUDA(const int M, const int N, const int K, const float ALPHA,
  const float *A, const int lda,
  const float *B, const int ldb,
  float *C, const int ldc) {
  cjmcv_cuda_util::GpuTimer gpu_timer;

  const int block_size = 32;
  dim3 threads_per_block(block_size, block_size);
  dim3 blocks_per_grid(N / threads_per_block.x, M / threads_per_block.y);
  
  // Warm up.
  MatrixMulKernelv1<block_size> << <blocks_per_grid, threads_per_block >> >
    (M, N, K, 1.0, A, lda, B, ldb, C, ldc);
  cudaMemset(C, 0, sizeof(float) * M * N);

  // Record the start event
  gpu_timer.Start();
#ifdef TEST_CUDA_V1
  MatrixMulKernelv1<block_size> << <blocks_per_grid, threads_per_block >> >
    (M, N, K, 1.0, A, lda, B, ldb, C, ldc);
#else
  MatrixMulKernelv2<block_size> << <blocks_per_grid, threads_per_block >> >
    (M, N, K, 1.0, A, lda, B, ldb, C, ldc);
#endif

  // Record the stop event
  gpu_timer.Stop();

  return gpu_timer.ElapsedMillis();
}

int main() {
  int ret = cjmcv_cuda_util::InitEnvironment(0);
  if (ret != 0) {
    printf("Failed to initialize the environment for cuda.");
    return -1;
  }

  int height_a = 2560, width_a = 800;
  int height_b = 800, width_b = 3200;
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

  // CPU
  time_t t = clock();
  MatrixMulCPUv1(height_a, width_b, width_a, 1.0, h_a, width_a,h_b, width_b, h_c, width_b);
  printf("In cpu version 1, msec_total = %lld, mean = %f\n", clock() - t, GetMean(h_c, height_a, width_b));
  //MatrixPrint(h_c, height_a, width_b);

  t = clock();
  MatrixMulCPUv2(height_a, width_b, width_a, 1.0, h_a, width_a, h_b, width_b, h_c, width_b);
  printf("In cpu version 2, msec_total = %lld, mean = %f\n", clock() - t, GetMean(h_c, height_a, width_b));
  //MatrixPrint(h_c, height_a, width_b);

  // GPU
  // Allocate memory in host. 
  float msec_total;
  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc((void **)&d_a, mem_size_a));
  CUDA_CHECK(cudaMalloc((void **)&d_b, mem_size_b));
  CUDA_CHECK(cudaMalloc((void **)&d_c, mem_size_c));

  // Copy host memory to device
  CUDA_CHECK(cudaMemcpy(d_a, h_a, mem_size_a, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, mem_size_b, cudaMemcpyHostToDevice));

  msec_total = MatrixMulCUDA(height_a, width_b, width_a, 1.0, d_a, width_a, d_b, width_b, d_c, width_b);

  // Copy memory back to host.
  CUDA_CHECK(cudaMemcpy(h_c, d_c, mem_size_c, cudaMemcpyDeviceToHost));
  printf("In gpu version 1, msec_total = %f, mean = %f\n", msec_total, GetMean(h_c, height_a, width_b));
  //MatrixPrint(h_c, height_a, width_b);

  free(h_a);
  free(h_b);
  free(h_c);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cjmcv_cuda_util::CleanUpEnvironment();

  return 0;
}
