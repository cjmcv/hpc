#include "operator/dot_product.h"
#include "executor.h"
// 
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
  cux::Executor *executor = new cux::Executor();

  int ret = executor->InitEnvironment(0);
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
  executor->Run(cux::RunMode::OnHost, h_vector_a, h_vector_b, data_len, h_result);

  // GPU
  // Allocate memory in host. 
  float *d_vector_a = NULL, *d_vector_b = NULL;
  float *d_result = NULL;
  CUDA_CHECK(cudaMalloc((void **)&d_vector_a, data_mem_size));
  CUDA_CHECK(cudaMalloc((void **)&d_vector_b, data_mem_size));
  CUDA_CHECK(cudaMalloc((void **)&d_result, sizeof(float)));

  // Copy host memory to device
  CUDA_CHECK(cudaMemcpy(d_vector_a, h_vector_a, data_mem_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vector_b, h_vector_b, data_mem_size, cudaMemcpyHostToDevice));

  executor->Run(cux::RunMode::OnDevice, d_vector_a, d_vector_b, data_len, *d_result);
  
  CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
  //printf("\nIn gpu version 1, h_result = %f\n", h_result);

  free(h_vector_a);
  free(h_vector_b);

  cudaFree(d_vector_a);
  cudaFree(d_vector_b);
  cudaFree(d_result);
  executor->CleanUpEnvironment();

  delete executor;

  system("pause");
  return 0;
}
