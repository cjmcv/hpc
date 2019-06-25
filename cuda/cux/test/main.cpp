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
  executor->SelectOp("dot_product", "");

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

  executor->SetOpIoData(inputs, outputs);

  // Run.
  executor->Run(cux::RunMode::ON_HOST);
  executor->Run(cux::RunMode::ON_DEVICE);

  delete in_a;
  delete in_b;
  delete out;

  delete executor;
}

void GEMMTest() {
  cux::Executor *executor = new cux::Executor();
  executor->SelectOp("gemm", "alpha: 1.0, beta: 0.0");

  const int loops = 100;
  executor->SetDebugParams(loops);

  cux::CuxData<float> *in_a = new cux::CuxData<float>(1, 1, 256, 800);
  cux::CuxData<float> *in_b = new cux::CuxData<float>(1, 1, 800, 320);
  std::vector<int> shape_a = in_a->shape();
  std::vector<int> shape_b = in_b->shape();
  cux::CuxData<float> *out_c = new cux::CuxData<float>(1, 1, 
    shape_a[cux::HEIGHT], shape_b[cux::WIDTH]);

  // Initialize 
  srand(0);
  GenMatrix(shape_a[cux::HEIGHT], shape_a[cux::WIDTH], in_a->GetCpuData());
  GenMatrix(shape_b[cux::HEIGHT], shape_b[cux::WIDTH], in_b->GetCpuData());

  std::vector<cux::CuxData<float> *> inputs;
  inputs.push_back(in_a);
  inputs.push_back(in_b);
  std::vector<cux::CuxData<float> *> outputs;
  outputs.push_back(out_c);

  executor->SetOpIoData(inputs, outputs);

  // Run.
  executor->Run(cux::RunMode::ON_HOST);
  executor->Run(cux::RunMode::ON_DEVICE);

  delete in_a;
  delete in_b;
  delete out_c;

  delete executor;
}

// TODO: µ¥Ôª²âÊÔ.
int main() {
  int ret = cux::Executor::InitEnvironment(0);
  if (ret != 0) {
    CUXLOG_ERR("Failed to initialize the environment for cuda.");
    return -1;
  }

  //////////
  //DotProductTest();
  GEMMTest();
  //////////
  
  cux::Executor::CleanUpEnvironment();
  system("pause");
  return 0;
}
