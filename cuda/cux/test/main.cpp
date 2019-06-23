#include "operator.h"
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

///////////
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

void DotProductTest() {
  cux::Executor *executor = new cux::Executor();

  int ret = executor->InitEnvironment(0);
  if (ret != 0) {
    CUXLOG_ERR("Failed to initialize the environment for cuda.");
    return ;
  }

  executor->SelectOp("dot_product");

  const int loops = 100;
  executor->SetDebugParams(loops);

  const int data_len = 10240000; // data_len % threads_per_block == 0.
  cux::CuxData<float> *in_a = new cux::CuxData<float>(1, 1, 1, data_len);
  cux::CuxData<float> *in_b = new cux::CuxData<float>(1, 1, 1, data_len);
  cux::CuxData<float> *out = new cux::CuxData<float>(1, 1, 1, 1);

  // Initialize 
  srand(0);
  GenArray(data_len, in_a->GetCpuData());
  GenArray(data_len, in_b->GetCpuData());

  std::vector<cux::CuxData<float> *> inputs;
  inputs.push_back(in_a);
  inputs.push_back(in_b);
  std::vector<cux::CuxData<float> *> outputs;
  outputs.push_back(out);

  // TODO: Op selection.
  // TODO: Use a factory to manage params?
  cux::OpParam *params = nullptr;
  executor->SetOpIoParams(inputs, outputs, params);

  // Run.
  executor->Run(cux::RunMode::ON_HOST);
  executor->Run(cux::RunMode::ON_DEVICE);

  delete in_a;
  delete in_b;
  delete out;

  executor->CleanUpEnvironment();
  delete executor;
}

void GemmTest() {
  //int ret = cjmcv_cuda_util::InitEnvironment(0);
  //if (ret != 0) {
  //  printf("Failed to initialize the environment for cuda.");
  //  return -1;
  //}

  //int height_a = 2560, width_a = 800;
  //int height_b = 800, width_b = 3200;
  //if (width_a != height_b) {
  //  printf("width_a should be equal to height_b.\n");
  //  return 1;
  //}

  //const int mem_size_a = sizeof(float) * height_a * width_a;
  //const int mem_size_b = sizeof(float) * height_b * width_b;
  //const int mem_size_c = sizeof(float) * height_a * width_b;

  //float *h_a = (float *)malloc(mem_size_a);
  //float *h_b = (float *)malloc(mem_size_b);
  //float *h_c = (float *)malloc(mem_size_c);
  //if (h_a == NULL || h_b == NULL || h_c == NULL) {
  //  printf("Fail to malloc.\n");
  //  return 1;
  //}

  //// Initialize 
  //srand(0);
  //GenMatrix(height_a, width_a, h_a);
  //GenMatrix(height_b, width_b, h_b);

  //// CPU
  //time_t t = clock();
  //MatrixMulCPUv1(height_a, width_b, width_a, 1.0, h_a, width_a, h_b, width_b, h_c, width_b);
  //printf("In cpu version 1, msec_total = %lld, mean = %f\n", clock() - t, GetMean(h_c, height_a, width_b));
  ////MatrixPrint(h_c, height_a, width_b);

  //t = clock();
  //MatrixMulCPUv2(height_a, width_b, width_a, 1.0, h_a, width_a, h_b, width_b, h_c, width_b);
  //printf("In cpu version 2, msec_total = %lld, mean = %f\n", clock() - t, GetMean(h_c, height_a, width_b));
  ////MatrixPrint(h_c, height_a, width_b);

  //// GPU
  //// Allocate memory in host. 
  //float msec_total;
  //float *d_a, *d_b, *d_c;
  //CUDA_CHECK(cudaMalloc((void **)&d_a, mem_size_a));
  //CUDA_CHECK(cudaMalloc((void **)&d_b, mem_size_b));
  //CUDA_CHECK(cudaMalloc((void **)&d_c, mem_size_c));

  //// Copy host memory to device
  //CUDA_CHECK(cudaMemcpy(d_a, h_a, mem_size_a, cudaMemcpyHostToDevice));
  //CUDA_CHECK(cudaMemcpy(d_b, h_b, mem_size_b, cudaMemcpyHostToDevice));

  //msec_total = MatrixMulCUDA(height_a, width_b, width_a, 1.0, d_a, width_a, d_b, width_b, d_c, width_b);

  //// Copy memory back to host.
  //CUDA_CHECK(cudaMemcpy(h_c, d_c, mem_size_c, cudaMemcpyDeviceToHost));
  //printf("In gpu version 1, msec_total = %f, mean = %f\n", msec_total, GetMean(h_c, height_a, width_b));
  ////MatrixPrint(h_c, height_a, width_b);

  //free(h_a);
  //free(h_b);
  //free(h_c);

  //cudaFree(d_a);
  //cudaFree(d_b);
  //cudaFree(d_c);
  //cjmcv_cuda_util::CleanUpEnvironment();
}
// TODO: µ¥Ôª²âÊÔ.
int main() {
  DotProductTest();

  system("pause");
  return 0;
}
