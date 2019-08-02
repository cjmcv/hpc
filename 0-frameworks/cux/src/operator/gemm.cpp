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

    GemmCpuKernel *kernel = new GemmCpuKernel();
    kernel->type_flag = TypeFlag::kFloat32;
    kernel->func = func;
    kernel->describe_info = "Normal";
    kernel->params.alpha = 1.0;
    kernel->params.beta = 0.0;

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

    GemmCpuKernel *kernel = new GemmCpuKernel();
    kernel->type_flag = TypeFlag::kFloat32;
    kernel->func = func;
    kernel->describe_info = "Adjust iteration order";
    kernel->params.alpha = 1.0;
    kernel->params.beta = 0.0;

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

    GemmCpuKernel *kernel = new GemmCpuKernel();
    kernel->type_flag = TypeFlag::kFloat32;
    kernel->func = func;
    kernel->describe_info = "Block-based";
    kernel->params.alpha = 1.0;
    kernel->params.beta = 0.0;

    cpu_kernels_.push_back(kernel);
  }
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

Operator *Gemm::Creator(OpAssistor *assistor, std::string &params_str) {
  GemmKernelParam params;
  params.alpha = atoi(StrProcessor::FetchSubStr(params_str, "alpha:", ",").c_str());
  params.beta = atoi(StrProcessor::FetchSubStr(params_str, "beta:", ",").c_str());
  return new Gemm(assistor, params);
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
    GemmCpuKernel *kernel = cpu_kernels_[ki];

    const void *A, *B;
    void *C;
    kernel->time_record.input = GET_TIME_DIFF(cpu_timer_,
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
    kernel->time_record.run = GET_TIME_DIFF(cpu_timer_,
      kernel->func(M, N, K, kernel->params.alpha, A, lda, B, ldb, kernel->params.beta, C, ldc);
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

  // TODO: 记录第一个kernel的类型，切换到其他类型的核时，需要拷贝数据;还是在准备数据时，先完成拷贝？
  for (int ki = 0; ki < gpu_kernels_.size(); ki++) {
    GemmGpuKernel *kernel = gpu_kernels_[ki];
    Config2D config = kernel->get_config(M, N);

    // Record the occupancy for profiling.
    QueryPotentialOccupancy(kernel->kernel_address, ki,
                            config.threads_per_block.x * config.threads_per_block.y,
                            config.shared_memory_size);
    // Input.
    const void *A, *B;
    void *C;
    kernel->time_record.input = GET_TIME_DIFF(gpu_timer_,
      TYPE_SWITCH(kernel->type_flag, T, {
        A = A_->GetGpuData<T>(PUSH_IF_EMPTY);
        B = B_->GetGpuData<T>(PUSH_IF_EMPTY);
        C = C_->GetGpuData<T>(PUSH_IF_EMPTY);
      };);
    ); 
    // Save original data.
    if (ki == 0) {
      C_->Save(kernel->type_flag, ON_DEVICE);
    }
    // Warm up.
    kernel->time_record.warnup = GET_TIME_DIFF(gpu_timer_,
      kernel->func(config, M, N, K, kernel->params.alpha, A, lda, B, ldb, kernel->params.beta, C, ldc);
    );
    // Run.
    C_->Restore(kernel->type_flag, ON_DEVICE);
    kernel->time_record.run = GET_TIME_DIFF(gpu_timer_,
      kernel->func(config, M, N, K, kernel->params.alpha, A, lda, B, ldb, kernel->params.beta, C, ldc);
    );
    // Output.
    kernel->time_record.output = GET_TIME_DIFF(gpu_timer_, 
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