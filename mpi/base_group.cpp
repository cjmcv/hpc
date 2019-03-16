/*!
* \brief Group communication.
*/

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define NPROCS 8

int main (int argc, char *argv[]) {
  
  int num_tasks, rank, rc;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
  
  if (num_tasks != NPROCS) {
    if (rank == 0) {
      printf("Quitting. Must specify %d tasks\n", NPROCS);
      MPI_Abort(MPI_COMM_WORLD, rc);
      exit(1);
    }
  }

  int sendbuf=rank, recvbuf;

  // Extract the original group handle.
  MPI_Group orig_group;
  MPI_Comm_group(MPI_COMM_WORLD, &orig_group);
  
  int ranks1[4]={0,1,2,3}, ranks2[4]={4,5,6,7};
  // Divide tasks into two distinct groups based upon rank.
  // int MPI_Group_incl(MPI_Group group, int n, int *ranks, MPI_Group new group)
  // n: The number of elements in the ranks parameter and the size of the new group.
  // ranks: The processes to be included in the new group
  MPI_Group new_group;
  if (rank < NPROCS/2)
    MPI_Group_incl(orig_group, NPROCS/2, ranks1, &new_group);
  else
    MPI_Group_incl(orig_group, NPROCS/2, ranks2, &new_group);

  // Create new new communicator and then perform collective communications.
  MPI_Comm new_comm;
  MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);
  MPI_Allreduce(&sendbuf, &recvbuf, 1, MPI_INT, MPI_SUM, new_comm);
  
  int new_rank = -1;
  MPI_Group_rank (new_group, &new_rank);
  printf("[new_comm] rank= %d newrank= %d recvbuf= %d\n", rank, new_rank, recvbuf); 

  MPI_Barrier(MPI_COMM_WORLD);
  
  MPI_Allreduce(&sendbuf, &recvbuf, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  printf("[MPI_COMM_WORLD] rank= %d newrank= %d recvbuf= %d\n", rank, new_rank, recvbuf); 

  MPI_Finalize();
}