// Corresponding to CUDA -> MatrixMulKernelv1
__kernel void MatrixMulDeviceV1(const int M, const int N, const int K, const float ALPHA,
  __global const float *A, const int lda,
  __global const float *B, const int ldb,
  __global float *C, const int ldc) {

  for (int gid_x = get_global_id(0), gid_y = get_global_id(1);
    gid_x < N && gid_y < M; 
    gid_x += get_global_size(0), gid_y += get_global_size(1)) {

    // 二维线程布局，每个线程对应结果矩阵中的一个元素
    // 每个线程只需要负责其k方向的遍历即可
    float c_sub_acc = 0;
    for (int k = 0; k < K; k++) {
      c_sub_acc += A[gid_y * lda + k] * B[k * ldb + gid_x];
    }
    C[gid_y * ldc + gid_x] = c_sub_acc;
  }
}

#define BLOCK_SIDE_SIZE 16
__kernel void MatrixMulDeviceV2(const int M, const int N, const int K, const float ALPHA,
  __global const float *A, const int lda,
  __global const float *B, const int ldb,
  __global float *C, const int ldc) {

  // CUDA version.
  //__shared__ float a_shared[BLOCK_SIZE][BLOCK_SIZE];
  //__shared__ float b_shared[BLOCK_SIZE][BLOCK_SIZE];

  //float c_sub_acc = 0;
  //// For blocks in grid.
  //for (int bk = 0; bk < K / BLOCK_SIZE; bk++) {
  //  a_shared[threadIdx.y][threadIdx.x] = A[(blockIdx.y * BLOCK_SIZE + threadIdx.y) * lda + (bk * BLOCK_SIZE + threadIdx.x)];
  //  b_shared[threadIdx.y][threadIdx.x] = B[(bk * BLOCK_SIZE + threadIdx.y) * ldb + (blockIdx.x * BLOCK_SIZE + threadIdx.x)];
  //  // Wait for data to complete loading to Shared memory.
  //  __syncthreads();

  //  // For elements in a block.
  //  for (int k = 0; k < BLOCK_SIZE; k++) {
  //    c_sub_acc += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
  //  }
  //  // To prevent the case from happening:
  //  // The next round of data is loaded when the data in share memory is not used up.
  //  __syncthreads();
  //}

  //C[(blockIdx.y * BLOCK_SIZE + threadIdx.y) * ldc + (blockIdx.x * BLOCK_SIZE + threadIdx.x)] += c_sub_acc;

  //

  // OpenCL version.
  __local float a_shared[BLOCK_SIDE_SIZE][BLOCK_SIDE_SIZE];
  __local float b_shared[BLOCK_SIDE_SIZE][BLOCK_SIDE_SIZE];

  int tid_x = get_local_id(0);
  int tid_y = get_local_id(1);
  int gid_x = get_global_id(0);
  int gid_y = get_global_id(1);

  float c_sub_acc = 0;
  // For blocks in grid.
  for (int bk = 0; bk < K / BLOCK_SIDE_SIZE; bk++) {
    a_shared[tid_y][tid_x] = A[gid_y * lda + (bk * BLOCK_SIDE_SIZE + tid_x)];
    b_shared[tid_y][tid_x] = B[(bk * BLOCK_SIDE_SIZE + tid_y) * ldb + gid_x];
    // Wait for data to complete loading to Shared memory.
    barrier(CLK_LOCAL_MEM_FENCE);

    // For elements in a block.
    for (int k = 0; k < BLOCK_SIDE_SIZE; k++) {
      c_sub_acc += a_shared[tid_y][k] * b_shared[k][tid_x];
    }
    // To prevent the case from happening:
    // The next round of data is loaded when the data in share memory is not used up.
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  C[gid_y * ldc + gid_x] += c_sub_acc;
}