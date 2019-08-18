/*!
* \brief gemm: C = A * B.
*/
#include "operator/gemm.h"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace cux {
  
// Kernel V0
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

// Kernel V1
// Adjust iteration order based on V0.
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
      register float aA = alpha*A[i*lda + k];
      for (j = 0; j < N; ++j) {
        C[i*ldc + j] += aA*B[k*ldb + j];
      }
    }
  }
}

// Kernel V2
// A simple version using simd based on V1.
void GemmHostV2(const int M, const int N,
                const int K, const float alpha,
                const float *A, const int lda,
                const float *B, const int ldb,
                const float beta,
                float *C, const int ldc) {
  int i, j, k;
  __m256 beta8 = _mm256_set1_ps(beta);
  for (i = 0; i < M*N - 7; i += 8) {
    __m256 c = _mm256_loadu_ps(C + i);
    c = _mm256_mul_ps(c, beta8);
    _mm256_storeu_ps(C + i, c);
  }
  for (; i < N; i++) {
    C[i] *= beta;
  }

  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      const float aA0 = alpha*A[i*lda + k];
      __m256 aA8 = _mm256_set1_ps(aA0);
      for (j = 0; j < N - 7; j += 8) {
        __m256 b = _mm256_loadu_ps(B + k*ldb + j); // load
        __m256 c = _mm256_loadu_ps(C + i*ldc + j);
        c = _mm256_fmadd_ps(aA8, b, c);            // run: aA8 * b + c
        _mm256_storeu_ps(C + i*ldc + j, c);        // store
      }
      for (; j < N; j++) {
        C[i*ldc + j] += aA0 * B[k*ldb + j];
      }
    }
  }
}

// Kernel V3
// 1. Expand the loop based on V2.
//    Reduce the use of  __m256 c8 = _mm256_loadu_ps(C + i*ldc + j);
//                  and  _mm256_storeu_ps(C + i*ldc + j, c8);
// 2. Use openmp.
void GemmHostV3(const int M, const int N,
                const int K, const float alpha,
                const float *A, const int lda,
                const float *B, const int ldb,
                const float beta,
                float *C, const int ldc) {
  int i, j, k;

  __m256 beta8 = _mm256_set1_ps(beta);
  for (i = 0; i < M*N - 7; i += 8) {
    __m256 c = _mm256_loadu_ps(C + i);
    c = _mm256_mul_ps(c, beta8);
    _mm256_storeu_ps(C + i, c);
  }
  for (; i < M*N; i++) {
    C[i] *= beta;
  }

  for (i = 0; i < M; ++i) {
    for (k = 0; k < K - 3; k += 4) {
      const float aA0 = alpha*A[i*lda + k];
      const float aA1 = alpha*A[i*lda + k + 1];
      const float aA2 = alpha*A[i*lda + k + 2];
      const float aA3 = alpha*A[i*lda + k + 3];
      __m256 aA08 = _mm256_set1_ps(aA0);
      __m256 aA18 = _mm256_set1_ps(aA1);
      __m256 aA28 = _mm256_set1_ps(aA2);
      __m256 aA38 = _mm256_set1_ps(aA3);
      for (j = 0; j < N - 7; j += 8) {
        __m256 b08 = _mm256_loadu_ps(B + k*ldb + j);    // load
        __m256 b18 = _mm256_loadu_ps(B + (k + 1)*ldb + j);
        __m256 b28 = _mm256_loadu_ps(B + (k + 2)*ldb + j);
        __m256 b38 = _mm256_loadu_ps(B + (k + 3)*ldb + j);

        __m256 c8 = _mm256_loadu_ps(C + i*ldc + j);

        c8 = _mm256_fmadd_ps(aA08, b08, c8);            // run: aA8 * b + c
        c8 = _mm256_fmadd_ps(aA18, b18, c8);
        c8 = _mm256_fmadd_ps(aA28, b28, c8);
        c8 = _mm256_fmadd_ps(aA38, b38, c8);

        _mm256_storeu_ps(C + i*ldc + j, c8);            // store
      }
      // Rest j.
      for (int jr = j; jr < N; jr++) { C[i*ldc + jr] += aA0 * B[k*ldb + jr]; }
      for (int jr = j, k1 = k + 1; jr < N; jr++) { C[i*ldc + jr] += aA1 * B[k1*ldb + jr]; }
      for (int jr = j, k2 = k + 2; jr < N; jr++) { C[i*ldc + jr] += aA2 * B[k2*ldb + jr]; }
      for (int jr = j, k3 = k + 3; jr < N; jr++) { C[i*ldc + jr] += aA3 * B[k3*ldb + jr]; }
    }
    // Rest k.
    for (; k < K; k++) {
      register float aA = alpha*A[i*lda + k];
      for (j = 0; j < N; ++j) {
        C[i*ldc + j] += aA*B[k*ldb + j];
      }
    }
  }
}

// Kernel V4
void GemmHostV4(const int M, const int N,
                const int K, const float alpha,
                const float *A, const int lda,
                const float *B, const int ldb,
                const float beta,
                float *C, const int ldc) {
  __m256 beta8 = _mm256_set1_ps(beta);
  for (int i = 0; i < M*N - 7; i += 8) {
    _mm256_storeu_ps(C + i, _mm256_mul_ps(_mm256_loadu_ps(C + i), beta8));
  }
  for (int i = (M*N) - (M*N%4); i < M*N; i++) {
    C[i] *= beta;
  }

#ifdef _OPENMP
#include <omp.h>
#pragma omp parallel for num_threads(6)
#endif
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K - 3; k += 4) {
      const float aA0 = alpha*A[i*lda + k];
      const float aA1 = alpha*A[i*lda + k + 1];
      const float aA2 = alpha*A[i*lda + k + 2];
      const float aA3 = alpha*A[i*lda + k + 3];
      __m256 aA08 = _mm256_set1_ps(aA0);
      __m256 aA18 = _mm256_set1_ps(aA1);
      __m256 aA28 = _mm256_set1_ps(aA2);
      __m256 aA38 = _mm256_set1_ps(aA3);
      for (int j = 0; j < N - 7; j += 8) {
        __m256 b08 = _mm256_loadu_ps(B + k*ldb + j);    // load
        __m256 b18 = _mm256_loadu_ps(B + (k + 1)*ldb + j);
        __m256 b28 = _mm256_loadu_ps(B + (k + 2)*ldb + j);
        __m256 b38 = _mm256_loadu_ps(B + (k + 3)*ldb + j);

        __m256 c8 = _mm256_loadu_ps(C + i*ldc + j);

        c8 = _mm256_fmadd_ps(aA08, b08, c8);            // run: aA8 * b + c
        c8 = _mm256_fmadd_ps(aA18, b18, c8);
        c8 = _mm256_fmadd_ps(aA28, b28, c8);
        c8 = _mm256_fmadd_ps(aA38, b38, c8);

        _mm256_storeu_ps(C + i*ldc + j, c8);            // store
      }
      // Rest j.
      for (int jr = N - (N % 8); jr < N; jr++) { C[i*ldc + jr] += aA0 * B[k*ldb + jr]; }
      for (int jr = N - (N % 8), k1 = k + 1; jr < N; jr++) { C[i*ldc + jr] += aA1 * B[k1*ldb + jr]; }
      for (int jr = N - (N % 8), k2 = k + 2; jr < N; jr++) { C[i*ldc + jr] += aA2 * B[k2*ldb + jr]; }
      for (int jr = N - (N % 8), k3 = k + 3; jr < N; jr++) { C[i*ldc + jr] += aA3 * B[k3*ldb + jr]; }
    }
    // Rest k.
    for (int k = K - K%4; k < K; k++) {
      register float aA = alpha*A[i*lda + k];
      for (int j = 0; j < N; ++j) {
        C[i*ldc + j] += aA*B[k*ldb + j];
      }
    }
  }
}

// Kernel V5
// Block-based matrix multiplication in cpu.
void GemmHostV5(const int M, const int N, 
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
    kernel->describe_info = "V0 + Adjust iteration order";

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
      //CUXLOG_WARN("V2> SSE/AVX acceleration is not enabled.");
    };

    GemmCpuKernelIF *kernel = new GemmCpuKernelIF();
    kernel->type_flag = TypeFlag::FLOAT32;
    kernel->func = func;
    kernel->describe_info = "V1 + SIMD";

    cpu_kernels_.push_back(kernel);
  }
  // Kernel v3.
  {
    auto func = [&](const int M, const int N,
                    const int K, const float alpha,
                    const void *A, const int lda,
                    const void *B, const int ldb,
                    const float beta,
                    void *C, const int ldc) -> void {
      GemmHostV3(M, N, K, alpha, (float *)A, lda, (float *)B, ldb, beta, (float *)C, ldc);
      //CUXLOG_WARN("V2> SSE/AVX acceleration is not enabled.");
    };

    GemmCpuKernelIF *kernel = new GemmCpuKernelIF();
    kernel->type_flag = TypeFlag::FLOAT32;
    kernel->func = func;
    kernel->describe_info = "V2 + Loop unrolling";

    cpu_kernels_.push_back(kernel);
  }
  // Kernel v4.
  {
    auto func = [&](const int M, const int N,
      const int K, const float alpha,
      const void *A, const int lda,
      const void *B, const int ldb,
      const float beta,
      void *C, const int ldc) -> void {
      GemmHostV4(M, N, K, alpha, (float *)A, lda, (float *)B, ldb, beta, (float *)C, ldc);
    };

    GemmCpuKernelIF *kernel = new GemmCpuKernelIF();
    kernel->type_flag = TypeFlag::FLOAT32;
    kernel->func = func;
    kernel->describe_info = "V3 + OpenMP with 6 threads.";

    cpu_kernels_.push_back(kernel);
  }
  // Kernel v5.
  {
    auto func = [&](const int M, const int N,
                    const int K, const float alpha,
                    const void *A, const int lda,
                    const void *B, const int ldb,
                    const float beta,
                    void *C, const int ldc) -> void {
      GemmHostV5(M, N, K, alpha, (float *)A, lda, (float *)B, ldb, beta, (float *)C, ldc);
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

void Gemm::IoCheckAndSet(const std::vector< Array4D* > &input,
                         const std::vector< Array4D* > &output) {
  // Check the dimensions.
  if (input.size() != 2 || output.size() != 1) {
    Help();
    CUXLOG_ERR("The dimensions of the data do not match.");
  }
  if (input[0]->shape()[Shape::WIDTH] != input[1]->shape()[Shape::HEIGHT] ||
    input[0]->shape()[Shape::HEIGHT] != output[0]->shape()[Shape::HEIGHT] ||
    input[1]->shape()[Shape::WIDTH] != output[0]->shape()[Shape::WIDTH]) {
    Help();
    CUXLOG_ERR("The dimensions of the data do not match.");
  }

  A_ = input[0];
  B_ = input[1];
  C_ = output[0];
}

void Gemm::AddPlugin(KernelInterface *kernel_if, OpRunMode mode) {
  if (mode == OpRunMode::ON_HOST)
    cpu_kernels_.push_back((GemmCpuKernelIF*)kernel_if);
  else
    gpu_kernels_.push_back((GemmGpuKernelIF*)kernel_if);

  ResetByKernelNum(cpu_kernels_.size(), gpu_kernels_.size());
}

void Gemm::ExtractDataTypes(std::vector<int>& type_flags) {
  type_flags.clear();
  type_flags.resize(TYPES_NUM);
  for (int i = 0; i < type_flags.size(); i++) {
    type_flags[i] = 0;
  }
  for (int i = 0; i < cpu_kernels_.size(); i++) {
    type_flags[cpu_kernels_[i]->type_flag] = 1;
  }
  for (int i = 0; i < gpu_kernels_.size(); i++) {
    type_flags[gpu_kernels_[i]->type_flag] = 1;
  }
}
////////////////////////////////////////////////
// cpp version
void Gemm::RunOnHost(const std::vector< Array4D* > &input,
                     const std::vector< Array4D* > &output) {
  CUXLOG_COUT("Gemm -> CPU: ");
  IoCheckAndSet(input, output);

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
      assistor_->checker()->CheckArray(C_->GetCpuData<T>(PUSH), 
        C_->num_element(), 1.0 / C_->num_element(), ki);
    );
  }
  // Show.
  for (int ki = 0; ki < cpu_kernels_.size(); ki++) {
    PrintRecordedInfo(OpRunMode::ON_HOST, ki, cpu_kernels_[ki]);
  }
}

//////////////////
// cuda version.
void Gemm::RunOnDevice(const std::vector< Array4D* > &input,
                       const std::vector< Array4D* > &output) {
  CUXLOG_COUT("Gemm -> GPU: ");
  IoCheckAndSet(input, output);

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
    QueryPotentialOccupancy(kernel->config_kernel, ki,
                            config.threads_per_block.x * config.threads_per_block.y,
                            config.shared_memory_size);

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
      assistor_->checker()->CheckArray(C_->GetCpuData<T>(PUSH), 
        C_->num_element(), 1.0 / C_->num_element(), ki);
    );
  }
  // Show.
  for (int ki = 0; ki < gpu_kernels_.size(); ki++) {
    PrintRecordedInfo(OpRunMode::ON_DEVICE, ki, gpu_kernels_[ki]);
  }
}

}