/*!
* \brief Environment Management Routines.
*/

#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
  
  int  num_tasks, rank, len; 
  char hostname[MPI_MAX_PROCESSOR_NAME];

  // Initialize MPI environment.
  MPI_Init(&argc,&argv);

  // Get number of tasks.
  MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

  // Get the rank of this processor.
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Get the hostname of each process.
  MPI_Get_processor_name(hostname, &len);
  printf ("Number of tasks= %d My rank= %d Running on %s\n", num_tasks, rank, hostname);

  // Clean up the environment.   
  MPI_Finalize();
}