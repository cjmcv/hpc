/*!
* \brief Record the basic usage of Zero Copy.
*/

#include "pocket-ai/engine/cu/common.hpp"

__global__ void SimpleKernel(int *data, int *res) {
    res[threadIdx.x] = data[threadIdx.x];
}

int main() {
    int device_num = 0;
    struct cudaDeviceProp device_prop;
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_num));	// Get the params of device
    int zero_copy_supported = device_prop.canMapHostMemory;	        //0: not support, 1: support
    if (zero_copy_supported == 0) {
        printf("Error Zero Copy not Supported");
    }

    int num = 1000;
    int size = num * sizeof(unsigned int);
    // Ready for Zero Copy: Set the flag to support mapped pinned allocations. 
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

    // Allocate memory in host.
    int *h_data_a, *d_data_a;
    CUDA_CHECK(cudaHostAlloc(&h_data_a, size, cudaHostAllocWriteCombined | cudaHostAllocMapped));
    // Get the corresponding pointer in device by the memory pointer in host.
    // And data in the two pointers are the same. 
    CUDA_CHECK(cudaHostGetDevicePointer(&d_data_a, h_data_a, 0));

    // Initialize the memory in host.
    for (int i = 0;i < num;i++)
        h_data_a[i] = i;

    int *h_data_b, *d_data_b;
    CUDA_CHECK(cudaHostAlloc(&h_data_b, size, cudaHostAllocWriteCombined | cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_data_b, h_data_b, 0));

    // Kernel, Copy data from h_data_a to h_data_b through device.
    // ps: d_data_a can fetch the data in h_data_a directly.
    SimpleKernel << <1, num >> > (d_data_a, d_data_b);

    // Wait for kernel to run to an end.
    cudaDeviceSynchronize();

    // Check the result.
    for (int i = 0;i < num;i++)
      printf("%d ", h_data_b[i]);

    // Attention: The memory used in Zero Copy should be freed by cudaFreeHost and the pointer in host.
    cudaFreeHost(h_data_a);
    cudaFreeHost(h_data_b);
}