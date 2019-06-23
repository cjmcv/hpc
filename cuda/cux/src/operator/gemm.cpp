/*!
* \brief gemm: C = A * B.
*/
#include "operator/gemm.h"

namespace cux {
  
// CPU version 1: 1583 ms
// Normal version in cpu as a reference
void GemmlCPUv1(const int M, const int N, const int K, const float ALPHA,
  const float *A, const int lda,
  const float *B, const int ldb,
  float *C, const int ldc) {
  int i, j, k;
  memset(C, 0, sizeof(float) * ldc * M);
  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      register float A_PART = ALPHA*A[i*lda + k];
      for (j = 0; j < N; ++j) {
        C[i*ldc + j] += A_PART*B[k*ldb + j];
      }
    }
  }
}

// CPU version 2: 3389 ms
// Block based matrix multiplication in cpu.
void GemmlCPUv2(const int M, const int N, const int K, const float ALPHA,
  const float *A, const int lda,
  const float *B, const int ldb,
  float *C, const int ldc) {
  int bi, bj, bk;
  int i, j, k;
  const int block_size = 32;
  int block_num_M = M / block_size;
  int block_num_N = N / block_size;
  int block_num_K = K / block_size;
  memset(C, 0, sizeof(float) * ldc * M);

  // Loop over all of the blocks.
  for (bi = 0; bi < block_num_M; ++bi) {
    for (bj = 0; bj < block_num_N; ++bj) {
      for (bk = 0; bk < block_num_K; ++bk) {
        // Loop over all of the elements in a block.
        for (i = bi*block_size; i < (bi + 1)*block_size; ++i) {
          for (k = bk*block_size; k < (bk + 1)*block_size; ++k) {
            for (j = bj*block_size; j < (bj + 1)*block_size; ++j) { 
              C[i*ldc + j] += A[i*lda + k] * B[k*ldb + j];
            }
          }
        }
      }
    }
  }
}

//////////
void GEMM::Help() const {
  CUXLOG_COUT("***************** Op Helper ********************");
  CUXLOG_COUT("* Name: GEMM.");
  CUXLOG_COUT("* Function: C(M, N) = A(M, K) * B(K, N) -> (height, width)");
  CUXLOG_COUT("* Inputs:  [Two] CuxData with one matrix each. ");
  CUXLOG_COUT("* Outputs: [One] CuxData with one matrix.");
  CUXLOG_COUT("* Params:  [Two] alpha and beta.");
  CUXLOG_COUT("**************************************************");
}

int GEMM::SetIoParams(const std::vector< CuxData<float>* > &input,
                      const std::vector< CuxData<float>* > &output,
                      const OpParam *params) {
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

  if (params != nullptr) {
    params_.alpha_ = ((GEMMOpParam *)params)->alpha_;
    params_.beta_ = ((GEMMOpParam *)params)->beta_;
  }

  return 0;
}


////////////////////////////////////////////////
// cpp version
// Normal version in cpu as a reference
void GEMM::RunOnHost() {
  CpuTimer cpu_timer;

  // Warp.
  const float *A = A_->GetCpuData();
  const float *B = B_->GetCpuData();
  float *C = C_->GetCpuData();

  const float ALPHA = params_.alpha_;
  const int M = A_->shape()[CuxShape::HEIGHT];
  const int N = B_->shape()[CuxShape::WIDTH];
  const int K = B_->shape()[CuxShape::HEIGHT]; // A_->shape()[CuxShape::WIDTH];
  const int lda = K;
  const int ldb = N;
  const int ldc = N;

  // Run.
  loops_ = 1;
  cpu_timer.Start();
  for (int i = 0; i < loops_; i++) {
    memset(C, 0, sizeof(float) * M * N);
    GemmlCPUv1(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
  }
  cpu_timer.Stop();
  cpu_time_record_ = cpu_timer.MilliSeconds() / loops_;

  CUXLOG_COUT("result: %f.", *C_->GetCpuData());
}

}