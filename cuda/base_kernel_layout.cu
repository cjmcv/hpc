/*!
* \brief Record the basic execution configuration of kernel.
*/

#include "cuda_util.h"
#include "time.h"

__global__ void KernelOne() {
  int x_id = blockIdx.x * blockDim.x + threadIdx.x;
  int y_id = blockIdx.y * blockDim.y + threadIdx.y;
  int z_id = blockIdx.z * blockDim.z + threadIdx.z;
  //                       z                                        y                         x
  int block_id = (blockIdx.z * (gridDim.x * gridDim.y)) + (blockIdx.y * gridDim.x) + blockIdx.x;
  //                    over block                                                 z                                           y                           x
  int thread_id = (block_id * (blockDim.x * blockDim.y * blockDim.z)) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
  
  printf("<(%d, %d, %d), (%d, %d), %d>\n", x_id, y_id, z_id, block_id, thread_id, warpSize);
}

__global__ void KernelTwo() {
  printf("(%d,%d,%d) | (%d, %d, %d) | (%d, %d, %d) | (%d, %d, %d)\n",
    gridDim.x, gridDim.y, gridDim.z,
    blockDim.x, blockDim.y, blockDim.z,
    blockIdx.x, blockIdx.y, blockIdx.z,
    threadIdx.x, threadIdx.y, threadIdx.z);
}

int main() {
  int dev_id = 0;
  int ret = cjmcv_cuda_util::InitEnvironment(dev_id);
  if (ret != 0) {
    printf("Failed to initialize the environment for cuda.");
    return -1;
  }

  dim3 threads_per_block1(3, 4, 5);
  dim3 blocks_per_grid1(2, 2, 2);
  //  kernel << <Dg, Db, Ns, S >> >
  // -> kernel<<<dim_grid, dim_block, num_bytes_in_SharedMem, stream>>>
  //
  // (dim3) Dg specifies the dimension and size of the grid, 
  //      such that Dg.x * Dg.y * Dg.z equals the number of blocks being launched;
  // (dim3) Db specifies the dimension and size of each block,
  //      such that Db.x * Db.y * Db.z equals the number of threads per block;
  // (size_t) Ns specifies the number of bytes in shared memory that is dynamically 
  //        allocated per block for this call in addition to the statically allocated memory;
  //        this dynamically allocated memory is used by any of the variables declared as an 
  //        external array as mentioned in __shared__; 
  //          Ns is an optional argument which defaults to 0;
  // (cudaStream_t) S specifies the associated stream; 
  //              S is an optional argument which defaults to 0. 
  KernelOne << < blocks_per_grid1, threads_per_block1 >> > ();

  CUDA_CHECK(cudaDeviceSynchronize());

  printf("gridDim | blockDim  | blockIdx  | threadIdx \n");
  dim3 threads_per_block2(3, 3, 3);
  dim3 blocks_per_grid2(2, 2, 2);
  KernelTwo << < threads_per_block2, blocks_per_grid2 >> > ();

  // Reset device
  cjmcv_cuda_util::CleanUpEnvironment();
  return 0;
}