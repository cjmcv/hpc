/*!
* \brief gemm: C = A * B.
* \reference: https://computing.llnl.gov/tutorials/mpi/samples/C/mpi_mm.c
*/

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define MASTER 0                   /* rank of first task */
#define TAG_FROM_MASTER 1          /* setting A message type */
#define TAG_FROM_WORKER 2          /* setting A message type */

int main(int argc, char *argv[]) {

  //     K           N         N
  //  M (A)   x   K (B)  =  M (C)
  int M = 62;
  int K = 15;
  int N = 7;
  int lda = K, ldb = N, ldc = N;
  double *A = (double *)malloc(M * lda * sizeof(double));
  double *B = (double *)malloc(K * ldb * sizeof(double));
  double *C = (double *)malloc(M * ldc * sizeof(double));
  
  int num_tasks, rank, rc;  
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
  if (num_tasks < 2) {
    printf("Need at least two MPI tasks. Quitting...\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
    exit(1);
  }

  /**************************** master task ************************************/
  if (rank == MASTER) {
    int num_workers = num_tasks - 1;
    printf("Start with %d tasks.\n", num_tasks);
    printf("Initializing arrays...\n");
    for (int i = 0; i < M; i++)
      for (int j = 0; j < K; j++)
        A[i*lda+j] = 2;
    for (int i = 0; i < K; i++)
      for (int j = 0; j < N; j++)
        B[i*ldb+j] = 2;

    /* Send matrix data to the worker tasks */
    // averow, extra and offset are used to determine rows sent to each worker
    int averow = M / num_workers;
    int extra = M % num_workers;
    int offset = 0;
    // rows of matrix A sent to each worker.
    int rows; 
    
    double t = MPI_Wtime();
    for (int dest = 1; dest <= num_workers; dest++) {
      rows = (dest <= extra) ? averow + 1 : averow;
      printf("Sending %d rows to task %d offset=%d\n", rows, dest, offset);
      MPI_Send(&offset, 1, MPI_INT, dest, TAG_FROM_MASTER, MPI_COMM_WORLD);
      MPI_Send(&rows, 1, MPI_INT, dest, TAG_FROM_MASTER, MPI_COMM_WORLD);
      MPI_Send(A+offset*lda, rows*lda, MPI_DOUBLE, dest, TAG_FROM_MASTER, MPI_COMM_WORLD);
      MPI_Send(B, K*ldb, MPI_DOUBLE, dest, TAG_FROM_MASTER, MPI_COMM_WORLD);
      offset = offset + rows;
    }

    /* Receive results from worker tasks */
    for (int i = 1; i <= num_workers; i++) {
      int source = i;
      MPI_Recv(&offset, 1, MPI_INT, source, TAG_FROM_WORKER, MPI_COMM_WORLD, &status);
      MPI_Recv(&rows, 1, MPI_INT, source, TAG_FROM_WORKER, MPI_COMM_WORLD, &status);
      MPI_Recv(C+offset*ldc, rows*ldc, MPI_DOUBLE, source, TAG_FROM_WORKER, MPI_COMM_WORLD, &status);
      printf("Received results from task %d\n", source);
    }
    
    /* Print results */
    printf("******************************************************\n");
    printf("That tooks %f seconds\n", MPI_Wtime() - t);
    printf("Result Matrix:\n");
    for (int i = 0; i < M; i++) {
      printf("\n");
      for (int j = 0; j < N; j++)
        printf("%6.2f   ", C[i*ldc+j]);
    }
    printf("\n******************************************************\n");
    printf("Done.\n");
  }

  /**************************** worker task ************************************/
  if (rank != MASTER) {
    int offset, rows;
    MPI_Recv(&offset, 1, MPI_INT, MASTER, TAG_FROM_MASTER, MPI_COMM_WORLD, &status);
    MPI_Recv(&rows, 1, MPI_INT, MASTER, TAG_FROM_MASTER, MPI_COMM_WORLD, &status);
    MPI_Recv(A, rows*lda, MPI_DOUBLE, MASTER, TAG_FROM_MASTER, MPI_COMM_WORLD, &status);
    MPI_Recv(B, K*ldb, MPI_DOUBLE, MASTER, TAG_FROM_MASTER, MPI_COMM_WORLD, &status);

    for (int k = 0; k < N; k++) {
      for (int i = 0; i < rows; i++) {
        C[i*ldc+k] = 0.0;
        for (int j = 0; j < K; j++)
          C[i*ldc+k] = C[i*ldc+k] + A[i*lda+j] * B[j*ldb+k];
      }
    }
    MPI_Send(&offset, 1, MPI_INT, MASTER, TAG_FROM_WORKER, MPI_COMM_WORLD);
    MPI_Send(&rows, 1, MPI_INT, MASTER, TAG_FROM_WORKER, MPI_COMM_WORLD);
    MPI_Send(C, rows*ldc, MPI_DOUBLE, MASTER, TAG_FROM_WORKER, MPI_COMM_WORLD);
  }

  MPI_Finalize();
  
  if(A) free(A);
  if(B) free(B);
  if(C) free(C);
}