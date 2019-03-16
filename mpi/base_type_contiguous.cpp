/*!
* \brief Send and receive custom types of data 
*     by using MPI_Type_contiguous.
*/

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

template <class T>
void PrintArrayInRankOrder(std::string prefix, T *data, int len, 
                           int rank, int num_tasks) {
  for (int i=0; i<num_tasks; i++) {
    if (rank == i) {
      std::cout << prefix << "-[rank = " << rank << "] : ";
      for (int i=0; i<len; i++) {
        std::cout << data[i] << ", ";
      }
      std::cout << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

int main (int argc, char *argv[]) {
    
  int num_tasks, rank, rc;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
  
  // Need two tasks.
  if (num_tasks != 2) {
    if (rank == 0) {
      printf("Quitting. Only need two tasks: num_tasks=%d\n", num_tasks);
      MPI_Abort(MPI_COMM_WORLD, rc);
      exit(1);
    }
  }
  
  MPI_Status status;

  /////////////////////////////////////////////////////////////////
  // Create a new type for TRIPLE_FLOAT.
  MPI_Datatype TRIPLE_FLOAT;  
  MPI_Type_contiguous(3, MPI_FLOAT, &TRIPLE_FLOAT);
  MPI_Type_commit(&TRIPLE_FLOAT);

  // Prepare data;
  int len = 9;
  float *buffer = (float *)malloc(sizeof(float) * len);
  if (rank == 0)
    for(int i=0; i<len; i++)
      buffer[i] = i;
  else
    memset(buffer, 0, sizeof(float) * len);

  // Send and receive.
  if (rank == 0)
    MPI_Send(buffer, len/3, TRIPLE_FLOAT, 1, 123, MPI_COMM_WORLD);
  else
    MPI_Recv(buffer, len/3, TRIPLE_FLOAT, 0, 123, MPI_COMM_WORLD, &status);
  
  PrintArrayInRankOrder<float>("MPI_Type_contiguous", 
                                 buffer, len, rank, num_tasks);
  free(buffer);
    
  MPI_Type_free(&TRIPLE_FLOAT);  
  
  MPI_Finalize();
}