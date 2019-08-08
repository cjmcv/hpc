#include "executor.h"

////////////
// Plugins.
extern cux::KernelInterface *DotProductGPUPlugin();
extern cux::KernelInterface *GemmGPUPlugin();

void DotProductTest() {
  cux::Executor *executor = new cux::Executor();
  executor->Initialize(0);
  executor->SelectOp("dot_product", "");
  executor->AddPlugin(DotProductGPUPlugin(), cux::OpRunMode::ON_DEVICE);

  // Too large a value may cause overflow.
  const int data_len = 4096000; // data_len % threads_per_block == 0.
  cux::Array4D *in_a = new cux::Array4D(1, 1, 1, data_len);
  cux::Array4D *in_b = new cux::Array4D(1, 1, 1, data_len);
  cux::Array4D *out = new cux::Array4D(1, 1, 1, 1);

  //in_a->Fill(-2, 3, 0, cux::TypeFlag::FLOAT32, cux::OpRunMode::ON_HOST);
  //in_b->Fill(-2, 3, 0, cux::TypeFlag::FLOAT32, cux::OpRunMode::ON_HOST);

  std::vector<cux::Array4D*> inputs;
  inputs.push_back(in_a);
  inputs.push_back(in_b);
  std::vector<cux::Array4D*> outputs;
  outputs.push_back(out);

  // Run.
  // TODO: Data prepare.
  executor->BindAndFill(inputs, outputs, -2, 3, 0);
  executor->Run(cux::OpRunMode::ON_HOST);
  executor->Run(cux::OpRunMode::ON_DEVICE);

  delete in_a;
  delete in_b;
  delete out;

  executor->Clear();
  delete executor;
}

void GEMMTest() {
  cux::Executor *executor = new cux::Executor();
  executor->Initialize(0);
  executor->SelectOp("gemm", "alpha: 1.0, beta: 3.0");
  executor->AddPlugin(GemmGPUPlugin(), cux::OpRunMode::ON_DEVICE);

  int block_size = 32;
  cux::Array4D *in_a = new cux::Array4D(1, 1, block_size * 6, block_size * 5);
  cux::Array4D *in_b = new cux::Array4D(1, 1, block_size * 5, block_size * 8);
  std::vector<int> shape_a = in_a->shape();
  std::vector<int> shape_b = in_b->shape();
  cux::Array4D *out_c = new cux::Array4D(1, 1, shape_a[cux::HEIGHT], shape_b[cux::WIDTH]);

  //// Initialize 
  //in_a->Fill(-2, 5, 0, cux::TypeFlag::FLOAT32, cux::OpRunMode::ON_HOST);
  //in_b->Fill(-2, 5, 0, cux::TypeFlag::FLOAT32, cux::OpRunMode::ON_HOST);
  //out_c->Fill(0, 5, 0, cux::TypeFlag::FLOAT32, cux::OpRunMode::ON_HOST);

  std::vector<cux::Array4D*> inputs;
  inputs.push_back(in_a);
  inputs.push_back(in_b);
  std::vector<cux::Array4D*> outputs;
  outputs.push_back(out_c);

  // Run.
  executor->BindAndFill(inputs, outputs, -2, 5, 0);
  executor->Run(cux::OpRunMode::ON_HOST);
  out_c->Restore(cux::TypeFlag::FLOAT32, cux::ON_HOST); // For beta in gemm.
  executor->Run(cux::OpRunMode::ON_DEVICE);

  delete in_a;
  delete in_b;
  delete out_c;

  executor->Clear();
  delete executor;
}

int main() {
  int ret = cux::InitEnvironment();
  if (ret != 0) {
    CUXLOG_ERR("Failed to initialize the environment for cuda.");
    return -1;
  }

  cux::QueryDevices();
  ////////
  printf("DotProductTest.\n");
  DotProductTest();

  printf("\n\nGEMMTest.\n");
  GEMMTest();
  //////////

  cux::CleanUpEnvironment();
  system("pause");
  return 0;
}