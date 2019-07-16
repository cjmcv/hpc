/*!
* \brief gemm: C = A * B.
*/
#include "operator/gemm.h"

namespace cux {
  
// CPU version 1: 1583 ms
// Normal version in cpu as a reference
void GEMMHostV0(const int M, const int N, 
                const int K, const float alpha,
                const float *A, const int lda,
                const float *B, const int ldb,
                const float beta,
                float *C, const int ldc) {  
  int i, j, k;
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      C[i*ldc + j] *= beta;
    }
  }
  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      register float A_PART = alpha*A[i*lda + k];
      for (j = 0; j < N; ++j) {
        C[i*ldc + j] += A_PART*B[k*ldb + j];
      }
    }
  }
}

// CPU version 2: 3389 ms
// Block based matrix multiplication in cpu.
void GEMMHostV1(const int M, const int N, 
                const int K, const float alpha,
                const float *A, const int lda,
                const float *B, const int ldb,
                const float beta,
                float *C, const int ldc) {
  int bi, bj, bk;
  int i, j, k;
  const int block_size = 32;
  int block_num_M = M / block_size;
  int block_num_N = N / block_size;
  int block_num_K = K / block_size;

  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      C[i*ldc + j] *= beta;
    }
  }

  // Loop over all of the blocks.
  for (bi = 0; bi < block_num_M; ++bi) {
    for (bj = 0; bj < block_num_N; ++bj) {
      for (bk = 0; bk < block_num_K; ++bk) {
        // Loop over all of the elements in a block.
        for (i = bi*block_size; i < (bi + 1)*block_size; ++i) {
          for (k = bk*block_size; k < (bk + 1)*block_size; ++k) {
            for (j = bj*block_size; j < (bj + 1)*block_size; ++j) { 
              C[i*ldc + j] += alpha * A[i*lda + k] * B[k*ldb + j];
            }
          }
        }
      }
    }
  }
}

//////////
void GEMM::GEMMHost(const int kernel_id, 
                    const int M, const int N,
                    const int K, const float alpha,
                    const float *A, const int lda,
                    const float *B, const int ldb,
                    const float beta,
                    float *C, const int ldc) {
  int shared_memory_size = 0;
  switch (kernel_id) {
  case 0:
    GEMMHostV0(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    break;
  case 1:
    GEMMHostV1(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    break;
  default:
    CUXLOG_ERR("Host Kernel id (%d) not found.", kernel_id);
  }
}

void GEMM::Help() const {
  CUXLOG_COUT("***************** Op Helper ********************");
  CUXLOG_COUT("* Name: GEMM.");
  CUXLOG_COUT("* Function: C(M, N) = A(M, K) * B(K, N) -> (height, width)");
  CUXLOG_COUT("* Inputs:  [Two] CuxData with one matrix each. ");
  CUXLOG_COUT("* Outputs: [One] CuxData with one matrix.");
  CUXLOG_COUT("* Params:  [Two] alpha / beta -> alpha: 1.0, beta: 0.0");
  CUXLOG_COUT("**************************************************");
}

Operator *GEMM::Creator(std::string &params_str) {
  GEMMKernelParam params;
  params.alpha = atoi(StrProcessor::FetchSubStr(params_str, "alpha:", ",").c_str());
  params.beta = atoi(StrProcessor::FetchSubStr(params_str, "beta:", ",").c_str());
  return new GEMM(params);
}

int GEMM::SetIoData(const std::vector< CuxData<float>* > &input,
                    const std::vector< CuxData<float>* > &output) {
  // Check the dimensions.
  if (input.size() != 2 || output.size() != 1) {
    CUXLOG_ERR("Error: The dimensions of the input parameters do not match.");
    Help();
    // TODO: Error code.
    return -1;
  }

  A_ = input[0];
  B_ = input[1];
  C_ = output[0];
  return 0;
}

////////////////////////////////////////////////
// cpp version
// Normal version in cpu as a reference
void GEMM::RunOnHost() {
  CpuTimer cpu_timer;

  // Warp.
  const float *A = A_->GetCpuData(PUSH_IF_EMPTY);
  const float *B = B_->GetCpuData(PUSH_IF_EMPTY);
  float *C = C_->GetCpuData(PUSH_IF_EMPTY);

  const float alpha = kernel_params_.alpha;
  const float beta = kernel_params_.beta;
  const int M = A_->shape()[Shape::HEIGHT];
  const int N = B_->shape()[Shape::WIDTH];
  const int K = B_->shape()[Shape::HEIGHT]; // A_->shape()[Shape::WIDTH];
  const int lda = K;
  const int ldb = N;
  const int ldc = N;

  // Save original data.
  C_->Save(ON_HOST);

  // Run.
  cpu_time_kernel_record_.clear();
  for (int ki = 0; ki < cpu_kernel_cnt_; ki++) {
    cpu_timer.Start();
    for (int i = 0; i < op_params_.loop_cn; i++) {
      //(C, 0, sizeof(float) * M * N);
      C_->Restore(ON_HOST);
      GEMMHost(ki, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    cpu_timer.Stop();
    cpu_time_kernel_record_.push_back(cpu_timer.MilliSeconds() / op_params_.loop_cn);

    checker_.CheckArray(C_->GetCpuData(PUSH), C_->num_element(), ki);
  }

  CUXLOG_COUT("result: %f.", *C_->GetCpuData(PUSH));
}

//////////////////
// cuda version.
void GEMM::RunOnDevice() {
  // Time recorder.
  GpuTimer gpu_timer;

  // Input.
  gpu_timer.Start();
  const float *A = A_->GetGpuData(PUSH_IF_EMPTY);
  const float *B = B_->GetGpuData(PUSH_IF_EMPTY);
  float *C = C_->GetGpuData(PUSH_IF_EMPTY);
  gpu_timer.Stop();
  gpu_time_in_record_ = gpu_timer.MilliSeconds();

  const float alpha = kernel_params_.alpha;
  const float beta = kernel_params_.beta;
  const int M = A_->shape()[Shape::HEIGHT];
  const int N = B_->shape()[Shape::WIDTH];
  const int K = B_->shape()[Shape::HEIGHT]; // A_->shape()[Shape::WIDTH];
  const int lda = K;
  const int ldb = N;
  const int ldc = N;

  // Save original data.
  C_->Save(ON_DEVICE);

  // Prepare launch config for kernels.
  PrepareLaunchConfig(N, M);

  // Warm up.
  gpu_timer.Start();
  GEMMDevice(0, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  gpu_timer.Stop();
  gpu_time_warnup_record_ = gpu_timer.MilliSeconds();

  // Run.
  gpu_time_kernel_record_.clear();
  for (int ki = 0; ki < gpu_kernel_cnt_; ki++) {
    gpu_timer.Start();
    for (int i = 0; i < op_params_.loop_cn; i++) {
      //cudaMemset(C, 0, sizeof(float) * M * N);
      C_->Restore(ON_DEVICE);
      GEMMDevice(ki, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    gpu_timer.Stop();
    gpu_time_kernel_record_.push_back(gpu_timer.MilliSeconds() / op_params_.loop_cn);

    // Output, Only record the first time.
    if (ki == 0) {
      gpu_timer.Start();
      C_->GetCpuData(PUSH);
      gpu_timer.Stop();
      gpu_time_out_record_ = gpu_timer.MilliSeconds();
    }
    checker_.CheckArray(C_->GetCpuData(PUSH), C_->num_element(), ki);
  }
}

}