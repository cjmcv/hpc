/*!
* \brief gemm: C = A * B.
*/
#include "operator/gemm.h"

namespace cux {
  
// CPU version 0: 3718 ms
// Normal version in cpu as a reference
void GemmHostV0(const int M, const int N,
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
    for (j = 0; j < N; ++j) {
      float temp = C[i*ldc + j];
      for (k = 0; k < K; ++k) {
        temp += alpha * A[i*lda + k] * B[k*ldb + j];
      }
      C[i*ldc + j] = temp;
    }
  }
}

// CPU version 1: 383 ms
// Normal version in cpu as a reference
void GemmHostV1(const int M, const int N, 
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

// CPU version 2: 1400 ms
// Block-based matrix multiplication in cpu.
void GemmHostV2(const int M, const int N, 
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
void Gemm::CpuKernelsSetup() {
  cpu_kernels_.clear();
  // Kernel v0.
  {
    auto func = [&](const int M, const int N,
                    const int K, const float alpha,
                    const void *A, const int lda,
                    const void *B, const int ldb,
                    const float beta,
                    void *C, const int ldc) -> void {
      GemmHostV0(M, N, K, alpha, (float *)A, lda, (float *)B, ldb, beta, (float *)C, ldc);
    };

    GemmCpuKernelIF *kernel = new GemmCpuKernelIF();
    kernel->type_flag = TypeFlag::FLOAT32;
    kernel->func = func;
    kernel->describe_info = "Normal";

    cpu_kernels_.push_back(kernel);
  }
  // Kernel v1.
  {
    auto func = [&](const int M, const int N,
      const int K, const float alpha,
      const void *A, const int lda,
      const void *B, const int ldb,
      const float beta,
      void *C, const int ldc) -> void {
      GemmHostV1(M, N, K, alpha, (float *)A, lda, (float *)B, ldb, beta, (float *)C, ldc);
    };

    GemmCpuKernelIF *kernel = new GemmCpuKernelIF();
    kernel->type_flag = TypeFlag::FLOAT32;
    kernel->func = func;
    kernel->describe_info = "Adjust iteration order";

    cpu_kernels_.push_back(kernel);
  }
  // Kernel v2.
  {
    auto func = [&](const int M, const int N,
      const int K, const float alpha,
      const void *A, const int lda,
      const void *B, const int ldb,
      const float beta,
      void *C, const int ldc) -> void {
      GemmHostV2(M, N, K, alpha, (float *)A, lda, (float *)B, ldb, beta, (float *)C, ldc);
    };

    GemmCpuKernelIF *kernel = new GemmCpuKernelIF();
    kernel->type_flag = TypeFlag::FLOAT32;
    kernel->func = func;
    kernel->describe_info = "Block-based";

    cpu_kernels_.push_back(kernel);
  }
}

//////////////////////
Operator *Gemm::Creator(OpAssistor *assistor, std::string &params_str) {
  GemmKernelParam params;
  params.alpha = atoi(StrProcessor::FetchSubStr(params_str, "alpha:", ",").c_str());
  params.beta = atoi(StrProcessor::FetchSubStr(params_str, "beta:", ",").c_str());
  return new Gemm(assistor, params);
}

void Gemm::Help() const {
  CUXLOG_COUT("***************** Op Helper ********************");
  CUXLOG_COUT("* Name: Gemm.");
  CUXLOG_COUT("* Function: C(M, N) = A(M, K) * B(K, N) -> (height, width)");
  CUXLOG_COUT("* Inputs:  [Two] Array4D with one matrix each. ");
  CUXLOG_COUT("* Outputs: [One] Array4D with one matrix.");
  CUXLOG_COUT("* Params:  [Two] alpha / beta -> alpha: 1.0, beta: 0.0");
  CUXLOG_COUT("**************************************************");
}

int Gemm::SetIoData(const std::vector< Array4D* > &input,
                    const std::vector< Array4D* > &output) {
  // Check the dimensions.
  if (input.size() != 2 || output.size() != 1) {
    Help();
    CUXLOG_ERR("Error: The dimensions of the input parameters do not match.");
  }

  A_ = input[0];
  B_ = input[1];
  C_ = output[0];
  return 0;
}

void Gemm::AddPlugin(KernelInterface *kernel_if, OpRunMode mode) {
  if (mode == OpRunMode::ON_HOST)
    cpu_kernels_.push_back((GemmCpuKernelIF*)kernel_if);
  else
    gpu_kernels_.push_back((GemmGpuKernelIF*)kernel_if);

  ResetKernelNum(cpu_kernels_.size(), gpu_kernels_.size());
}

////////////////////////////////////////////////
// cpp version
void Gemm::RunOnHost() {
  CUXLOG_COUT("Gemm -> CPU: ");
  const int M = A_->shape()[Shape::HEIGHT];
  const int N = B_->shape()[Shape::WIDTH];
  const int K = B_->shape()[Shape::HEIGHT]; // A_->shape()[Shape::WIDTH];
  const int lda = K;
  const int ldb = N;
  const int ldc = N;

  for (int ki = 0; ki < cpu_kernels_.size(); ki++) {
    GemmCpuKernelIF *kernel = cpu_kernels_[ki];

    const void *A, *B;
    void *C;
    cpu_timer_record_[ki].input = GET_TIME_DIFF(cpu_timer_,
      TYPE_SWITCH(kernel->type_flag, T, {
        A = A_->GetCpuData<T>(PUSH_IF_EMPTY);
        B = B_->GetCpuData<T>(PUSH_IF_EMPTY);
        C = C_->GetCpuData<T>(PUSH_IF_EMPTY);
      };);
    );
    // Save original data.
    if (ki == 0) {
      C_->Save(kernel->type_flag, ON_HOST);
    }
    // Run.
    C_->Restore(kernel->type_flag, ON_HOST);
    cpu_timer_record_[ki].run = GET_TIME_DIFF(cpu_timer_,
      kernel->func(M, N, K, params_.alpha, A, lda, B, ldb, params_.beta, C, ldc);
    );
    TYPE_SWITCH(kernel->type_flag, T,
      assistor_->checker()->CheckArray(C_->GetCpuData<T>(PUSH), C_->num_element(), ki);
    );
  }
  // Show.
  for (int ki = 0; ki < cpu_kernels_.size(); ki++) {
    PrintRecordedInfo(OpRunMode::ON_HOST, ki, cpu_kernels_[ki]);
  }
}

//////////////////
// cuda version.
void Gemm::RunOnDevice() {
  CUXLOG_COUT("Gemm -> GPU: ");
  const int M = A_->shape()[Shape::HEIGHT];
  const int N = B_->shape()[Shape::WIDTH];
  const int K = B_->shape()[Shape::HEIGHT]; // A_->shape()[Shape::WIDTH];
  const int lda = K;
  const int ldb = N;
  const int ldc = N;

  for (int ki = 0; ki < gpu_kernels_.size(); ki++) {
    GemmGpuKernelIF *kernel = gpu_kernels_[ki];
    Config2D config = kernel->get_config(M, N);

    // Record the occupancy for profiling.
    QueryPotentialOccupancy(kernel->kernel_address, ki,
                            config.threads_per_block.x * config.threads_per_block.y,
                            config.shared_memory_size);
    // Check and convert precision.
    TYPE_SWITCH(kernel->type_flag, T, { 
      A_->CheckPrecsCpuCvt<T>();
      B_->CheckPrecsCpuCvt<T>();
      C_->CheckPrecsCpuCvt<T>();
    });
    
    // Input.
    const void *A, *B;
    void *C;
    gpu_timer_record_[ki].input = GET_TIME_DIFF(gpu_timer_,
      TYPE_SWITCH(kernel->type_flag, T, {
        A = A_->GetGpuData<T>(PUSH_IF_EMPTY);
        B = B_->GetGpuData<T>(PUSH_IF_EMPTY);
        C = C_->GetGpuData<T>(PUSH_IF_EMPTY);
      });
    ); 
    // Save original data if backup is empty.
    bool is_save_if_empty = true;
    C_->Save(kernel->type_flag, ON_DEVICE, is_save_if_empty);
    // Warm up.
    gpu_timer_record_[ki].warnup = GET_TIME_DIFF(gpu_timer_,
      kernel->func(config, M, N, K, params_.alpha, A, lda, B, ldb, params_.beta, C, ldc);
    );
    // Run.
    C_->Restore(kernel->type_flag, ON_DEVICE);
    gpu_timer_record_[ki].run = GET_TIME_DIFF(gpu_timer_,
      kernel->func(config, M, N, K, params_.alpha, A, lda, B, ldb, params_.beta, C, ldc);
    );
    // Output.
    gpu_timer_record_[ki].output = GET_TIME_DIFF(gpu_timer_,
      TYPE_SWITCH(kernel->type_flag, T, { C_->GetCpuData<T>(PUSH); });
    );
    // Check.
    TYPE_SWITCH(kernel->type_flag, T, 
      assistor_->checker()->CheckArray(C_->GetCpuData<T>(PUSH), C_->num_element(), ki);
    );
  }
  // Show.
  for (int ki = 0; ki < gpu_kernels_.size(); ki++) {
    PrintRecordedInfo(OpRunMode::ON_DEVICE, ki, gpu_kernels_[ki]);
  }
}

}