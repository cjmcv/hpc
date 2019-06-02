#include "operator/dot_product.h"

// 
int InitEnvironment(const int dev_id) {
  CUDA_CHECK(cudaSetDevice(dev_id));
  cudaDeviceProp device_prop;
  CUDA_CHECK(cudaGetDeviceProperties(&device_prop, dev_id));
  if (device_prop.computeMode == cudaComputeModeProhibited) {
    fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
    return 1;
  }
  fprintf(stderr, "GPU Device %d: \"%s\" with compute capability %d.%d with %d multi-processors.\n\n",
    dev_id, device_prop.name, device_prop.major, device_prop.minor, device_prop.multiProcessorCount);

  return 0;
}

void CleanUpEnvironment() {
  // Reset the device and exit
  // cudaDeviceReset causes the driver to clean up all state. While
  // not mandatory in normal operation, it is good practice.  It is also
  // needed to ensure correct operation when the application is being
  // profiled. Calling cudaDeviceReset causes all profile data to be
  // flushed before the application exits
  CUDA_CHECK(cudaDeviceReset());
}

template <typename T>
void PrintArray(std::string str, T *h_in, int num_items) {
  std::cout << str.c_str();
  for (int i = 0; i < num_items; i++) {
    std::cout << h_in[i] << ",";
  }
  std::cout << std::endl;
}


// Initialize the input data.
void GenArray(const int len, float *arr) {
  for (int i = 0; i < len; i++) {
    arr[i] = 1;//(float)rand() / RAND_MAX + (float)rand() / (RAND_MAX*RAND_MAX);
  }
}

int main() {
  int ret = InitEnvironment(0);
  if (ret != 0) {
    printf("Failed to initialize the environment for cuda.");
    return -1;
  }

  const int loops = 100;
  const int data_len = 10240000; // data_len % threads_per_block == 0
  const int data_mem_size = sizeof(float) * data_len;
  float *h_vector_a = (float *)malloc(data_mem_size);
  float *h_vector_b = (float *)malloc(data_mem_size);
  if (h_vector_a == NULL || h_vector_b == NULL) {
    printf("Fail to malloc.\n");
    return 1;
  }
  
  // Initialize 
  srand(0);
  GenArray(data_len, h_vector_a);
  GenArray(data_len, h_vector_b);

  // CPU
  time_t t = clock();
  float h_result = 0;
  for (int i = 0; i < loops; i++)
    h_result = cux::VectorDotProductCPU(h_vector_a, h_vector_b, data_len);
  printf("\nIn cpu version 1, msec_total = %lld, h_result = %f\n", clock() - t, h_result);

  // GPU
  // Allocate memory in host. 
  float msec_total;
  float *d_vector_a = NULL, *d_vector_b = NULL;
  float *d_result = NULL;
  CUDA_CHECK(cudaMalloc((void **)&d_vector_a, data_mem_size));
  CUDA_CHECK(cudaMalloc((void **)&d_vector_b, data_mem_size));
  CUDA_CHECK(cudaMalloc((void **)&d_result, sizeof(float)));

  // Copy host memory to device
  CUDA_CHECK(cudaMemcpy(d_vector_a, h_vector_a, data_mem_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vector_b, h_vector_b, data_mem_size, cudaMemcpyHostToDevice));

  msec_total = cux::VectorDotProductCUDA(loops, d_vector_a, d_vector_b, data_len, *d_result);
  
  CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
  printf("\nIn gpu version 1, msec_total = %f, h_result = %f\n", msec_total, h_result);

  free(h_vector_a);
  free(h_vector_b);

  cudaFree(d_vector_a);
  cudaFree(d_vector_b);
  cudaFree(d_result);
  CleanUpEnvironment();

  system("pause");
  return 0;
}
