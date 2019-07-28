#include "operator.h"
#include "executor.h"
#include "util/half.h"
#include "gtest/gtest.h"
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
    arr[i] = i % 2; // (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX*RAND_MAX);
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

void DotProductTest(const bool is_show_info) {
  cux::Executor *executor = new cux::Executor();
  executor->Initialize(0);
  executor->SelectOp("dot_product", "");

  const int loop_cn = 1 ;
  executor->SetOpParams(loop_cn);

  // Too large a value may cause overflow.
  const int data_len = 4096000; // data_len % threads_per_block == 0.
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

  executor->SetOpIoData(inputs, outputs);

  // Run.
  executor->Run(cux::OpRunMode::ON_HOST);
  executor->Run(cux::OpRunMode::ON_DEVICE);

  delete in_a;
  delete in_b;
  delete out;

  executor->Clear();
  delete executor;
}

void GEMMTest(const bool is_show_info) {
  cux::Executor *executor = new cux::Executor();
  executor->Initialize(0);
  executor->SelectOp("gemm", "alpha: 1.0, beta: 3.0");

  const int loop_cn = 1;
  executor->SetOpParams(loop_cn);

  int block_size = 32;
  cux::CuxData<float> *in_a = new cux::CuxData<float>(1, 1, block_size * 30, block_size * 25);
  cux::CuxData<float> *in_b = new cux::CuxData<float>(1, 1, block_size * 25, block_size * 40);
  std::vector<int> shape_a = in_a->shape();
  std::vector<int> shape_b = in_b->shape();
  cux::CuxData<float> *out_c = new cux::CuxData<float>(1, 1,
    shape_a[cux::HEIGHT], shape_b[cux::WIDTH]);

  // Initialize 
  srand(0);
  GenMatrix(shape_a[cux::HEIGHT], shape_a[cux::WIDTH], in_a->GetCpuData());
  GenMatrix(shape_b[cux::HEIGHT], shape_b[cux::WIDTH], in_b->GetCpuData());
  GenMatrix(shape_a[cux::HEIGHT], shape_b[cux::WIDTH], out_c->GetCpuData());

  std::vector<cux::CuxData<float> *> inputs;
  inputs.push_back(in_a);
  inputs.push_back(in_b);
  std::vector<cux::CuxData<float> *> outputs;
  outputs.push_back(out_c);

  executor->SetOpIoData(inputs, outputs);

  // Run.
  executor->Run(cux::OpRunMode::ON_HOST);
  out_c->Restore(cux::ON_HOST); // For beta in gemm.
  executor->Run(cux::OpRunMode::ON_DEVICE);

  delete in_a;
  delete in_b;
  delete out_c;

  executor->Clear();
  delete executor;
}

int main() {
  testing::InitGoogleTest();
  RUN_ALL_TESTS();

  int ret = cux::InitEnvironment();
  if (ret != 0) {
    CUXLOG_ERR("Failed to initialize the environment for cuda.");
    return -1;
  }
  //cux::QueryDevices();
  //////////
  //printf("DotProductTest.\n");
  //DotProductTest(true);

  //printf("\n\nGEMMTest.\n");
  //GEMMTest(false);
  //////////
  //HalfTest();

  cux::CleanUpEnvironment();
  system("pause");
  return 0;
}
