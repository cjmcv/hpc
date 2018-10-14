/*!
* \brief An experiment on Bank Conflict in Shared Memory.
* \Reference https://blog.csdn.net/endlch/article/details/47043069
*/

#include "cuda_util.h"
#include "time.h"

// Kernel. Just copy the data from d_in to d_out through shared memory.
//     And the block size in this experiment is limited to 32 * 32, 
//     because the shared memory is divided into 32 banks.
//
// 145.2 ms, in 10000 times.
// Different threads in a warp will not access the same bank.
//
//          bank0   bank1  bank2 . . . . bank31
//  warp0    t0      t1     t2   . . . .  t31
//  warp1    t32     t33    . . . . . . . t63
//  warp2    t64    . . . . . . . . . . . t95
//   .    .    .    .    .    .    .    .    .
//  warp31   t992   . . . . . . . . . . . t1023 
template <int BLOCK_SIZE>
__global__ void TestCopyKernelOne(const int *d_in, const int width, const int heigh, int *d_out) {
  int x_id = blockIdx.x * blockDim.x + threadIdx.x;
  int y_id = blockIdx.y * blockDim.y + threadIdx.y;
  int data_id = y_id * width + x_id;

  __shared__ int smem[BLOCK_SIZE][BLOCK_SIZE];

  if (x_id < width && y_id < heigh) {
    smem[threadIdx.y][threadIdx.x] = d_in[data_id];
    __syncthreads();
    d_out[data_id] = smem[threadIdx.y][threadIdx.x];
  }
}


// 160.8 ms, in 10000 times.
// Different threads in a warp will access the same bank.
//
//   warp0   warp1  warp2 . . . . warp31
//   bank0   bank1  bank2 . . . . bank31
//     t0     t32    t64   . . . . t992
//     t1     t33    . . . . . . . . .
//     t2    . . . . . . . . . . . . .
//     .    .    .    .    .    .    .   
//    t31   . . . . . . . . . . . t1023 
template <int BLOCK_SIZE>
__global__ void TestCopyKernelTwo(const int *d_in, const int width, const int heigh, int *d_out) {
  int x_id = blockIdx.x * blockDim.x + threadIdx.x;
  int y_id = blockIdx.y * blockDim.y + threadIdx.y;
  int data_id = y_id * width + x_id;

  __shared__ int smem[BLOCK_SIZE][BLOCK_SIZE];

  // Just switch the place between threadIdx.x and threadIdx.y.
  // Causing serious bank conflit.
  if (x_id < width && y_id < heigh) {

    smem[threadIdx.x][threadIdx.y] = d_in[data_id];
    __syncthreads();
    d_out[data_id] = smem[threadIdx.x][threadIdx.y];
  }
}

// 146.8 ms, in 10000 times.
// Avoid bank conflict by adding a column space in shared memory.
//
//    After you adding a column, there are 33 columns in Shared memory. 
// So the last column in row 0 corresponds to bank0, and the first column in row 1 corresponds to bank1.
//    For this : d_out[data_id] = smem[threadIdx.x][threadIdx.y];
//    The thread0 of warp0 will access bank0 on line0, and the thread1 of warp0 will access the first 
// column on line1, but it is corresponds to bank1, since the extral column was added.
//
//   bank0   bank1  bank2 . . . . bank31  bank0
//   bank1   bank2  bank3 . . . . bank0   bank1
//   bank2   bank3  bank4 . . . . bank1   bank2
//   bank3   bank4  bank5 . . . . bank2   bank3
//
//   warp0   warp1  warp2 . . . . warp31
//     t0     t32    t64   . . . . t992
//     t1     t33    . . . . . . . . .
//     t2    . . . . . . . . . . . . .
//     .    .    .    .    .    .    .   
//    t31   . . . . . . . . . . . t1023 
//
//  So: warp0(t0) <=> bank0, warp0(t1) <=> bank1...
//      warp1(t0) <=> bank1, warp1(t1) <=> bank2...
template <int BLOCK_SIZE>
__global__ void TestCopyKernelThree(const int *d_in, const int width, const int heigh, int *d_out) {
  int x_id = blockIdx.x * blockDim.x + threadIdx.x;
  int y_id = blockIdx.y * blockDim.y + threadIdx.y;
  int data_id = y_id * width + x_id;

  // Add a column space.
  __shared__ int smem[BLOCK_SIZE][BLOCK_SIZE + 1];

  if (x_id < width && y_id < heigh) {
    smem[threadIdx.x][threadIdx.y] = d_in[data_id];
    __syncthreads();
    d_out[data_id] = smem[threadIdx.x][threadIdx.y];
  }
}

bool CheckArrayIsEqual(const int *arr1, const int *arr2, const int len) {
  for (int i = 0; i < len; i++) {
    if (arr1[i] != arr2[i]) {
      return false;
    }
  }
  return true;
}

int main() {
  int dev_id = 0;
  int ret = cjmcv_cuda_util::InitEnvironment(dev_id);
  if (ret != 0) {
    printf("Failed to initialize the environment for cuda.");
    return -1;
  }

  const int width = 32;
  const int height = width;
  int len = width * height;
  // Initialize data in host.
  int *h_in = new int[len];
  int *h_out1 = new int[len];
  int *h_out2 = new int[len];
  int *h_out3 = new int[len];
  for (int i = 0; i < len; i++) {
    h_in[i] = i;
  }

  // Initialize data in device.
  int *d_in, *d_out1, *d_out2, *d_out3;
  CUDA_CHECK(cudaMalloc((float**)&d_in, len * sizeof(int)));
  CUDA_CHECK(cudaMalloc((float**)&d_out1, len * sizeof(int)));
  CUDA_CHECK(cudaMalloc((float**)&d_out2, len * sizeof(int)));
  CUDA_CHECK(cudaMalloc((float**)&d_out3, len * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, len * sizeof(int), cudaMemcpyHostToDevice));

  // Layout.
  const int block_size = width;
  dim3 threads_per_block(block_size, block_size);
  dim3 blocks_per_grid((width + threads_per_block.x - 1) / threads_per_block.x,
    (height + threads_per_block.y - 1) / threads_per_block.y);
  // Warn up.
  TestCopyKernelOne<block_size> << < blocks_per_grid, threads_per_block >> > (d_in, width, height, d_out1);
  CUDA_CHECK(cudaMemcpy(h_out1, d_out1, len * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemset(d_out1, 0, len * sizeof(int)));

  cjmcv_cuda_util::GpuTimer *timer = new cjmcv_cuda_util::GpuTimer;
  // The number of iterations.
  int iter_count = 10000;

  ////////// Kernel one ///////////////
  float time_recorder1 = 0;
  for (int i = 0; i < iter_count; i++) {
    timer->Start();
    TestCopyKernelOne<block_size> << < blocks_per_grid, threads_per_block >> > (d_in, width, height, d_out1);
    timer->Stop();
    time_recorder1 += timer->ElapsedMillis();
  }
  // The cudaMemcpy function is synchronized to the host, so you need not call cudaDeviceSynchronize() here.
  CUDA_CHECK(cudaMemcpy(h_out1, d_out1, len * sizeof(int), cudaMemcpyDeviceToHost));
  printf("Kernel One - time_elapsed: %f ms.\n", time_recorder1);

  ////////// Kernel two ///////////////
  float time_recorder2 = 0;
  for (int i = 0; i < iter_count; i++) {
    timer->Start();
    TestCopyKernelTwo<block_size> << < blocks_per_grid, threads_per_block >> > (d_in, width, height, d_out2);
    timer->Stop();
    time_recorder2 += timer->ElapsedMillis();
  }
  CUDA_CHECK(cudaMemcpy(h_out2, d_out2, len * sizeof(int), cudaMemcpyDeviceToHost));
  printf("Kernel Two - time_elapsed: %f ms.\n", time_recorder2);

  ////////// Kernel three ///////////////
  float time_recorder3 = 0;
  for (int i = 0; i < iter_count; i++) {
    timer->Start();
    TestCopyKernelThree<block_size> << < blocks_per_grid, threads_per_block >> > (d_in, width, height, d_out3);
    timer->Stop();
    time_recorder3 += timer->ElapsedMillis();
  }
  CUDA_CHECK(cudaMemcpy(h_out3, d_out3, len * sizeof(int), cudaMemcpyDeviceToHost));
  printf("Kernel Three - time_elapsed: %f ms.\n", time_recorder3);

  // Check results.
  if ((CheckArrayIsEqual(h_in, h_out1, len) == false) ||
      (CheckArrayIsEqual(h_in, h_out2, len) == false) ||
      (CheckArrayIsEqual(h_in, h_out3, len) == false)) {
    printf("FAILED.\n");
  }
  else {
    printf("PASS.\n");
  }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out1));
  CUDA_CHECK(cudaFree(d_out2));
  CUDA_CHECK(cudaFree(d_out3));

  //delete timer;
  if (h_in) delete[] h_in;
  if (h_out1) delete[] h_out1;
  if (h_out2) delete[] h_out2;
  if (h_out3) delete[] h_out3;

  // Reset device
  cjmcv_cuda_util::CleanUpEnvironment();
  return 0;
}