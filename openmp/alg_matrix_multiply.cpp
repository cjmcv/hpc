/*!
* \brief gemm: C = A * B.
*/
#include <omp.h>

#include <stdio.h>
#include <memory.h>
#include <stdlib.h>

void InitializeArray(const int len, float *data) {
  for (int i = 0; i < len; i++) {
    data[i] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX*RAND_MAX);
  }
}

// C(M,N) = A(M,K) * B(K,N)
void MatrixMulSerial(const int M, const int N, const int K, const float ALPHA,
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

void MatrixMulMP(const int M, const int N, const int K, const float ALPHA,
  const float *A, const int lda,
  const float *B, const int ldb,
  float *C, const int ldc) {

  memset(C, 0, sizeof(float) * ldc * M);
  int i, j, k;
  register float A_PART;

  #pragma omp parallel for private(A_PART, i, j, k)  
  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      A_PART = ALPHA*A[i*lda + k];
      for (j = 0; j < N; ++j) {
        C[i*ldc + j] += A_PART*B[k*ldb + j];
      }
    }
  }
}

int main(int argc, char *argv[]) {
 
  int M=1000, N=1500, K=2000; // C(M,N) = A(M,K) * B(K,N)  
  int i, j, k;

  float *A, *B, *C1, *C2;
  A = (float *)malloc(M * K * sizeof(float));
  B = (float *)malloc(K * N * sizeof(float));
  C1 = (float *)malloc(M * N * sizeof(float));
  C2 = (float *)malloc(M * N * sizeof(float));

  InitializeArray(M * K, A);
  InitializeArray(K * N, B);

  const int fops = 2 * N * M * K;
  double start_time, run_time;

  // Process.
  start_time = omp_get_wtime();
  MatrixMulSerial(M, N, K, 1.0, A, K, B, N, C1, N);
  run_time = omp_get_wtime() - start_time;
  printf("Serial: %f seconds.\n", run_time);
  printf("mflops: %f.\n", fops / (1000000.0* run_time));

  start_time = omp_get_wtime();
  MatrixMulMP(M, N, K, 1.0, A, K, B, N, C2, N);
  run_time = omp_get_wtime() - start_time;
  printf("OpenMP: %f seconds <%d threads>.\n", 
    run_time, omp_get_max_threads());
  printf("mflops: %f.\n", fops / (1000000.0* run_time));

  // Check.
  float err_acc = 0;
  for (int i = 0; i < M*N; i++)
    err_acc += C1[i] - C2[i];

  if (err_acc >= 0.01)
    printf("Accumulative error: %f", err_acc);
  else
    printf("Pass.\n");

  return 0;
}
