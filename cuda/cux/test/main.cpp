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
  executor->SetOpIoParams(inputs, outputs, nullptr);

  // Run.
  executor->Run(cux::RunMode::ON_HOST);
  executor->Run(cux::RunMode::ON_DEVICE);

  delete in_a;
  delete in_b;
  delete out;

  executor->CleanUpEnvironment();
  delete executor;

  system("pause");
  return 0;
}
