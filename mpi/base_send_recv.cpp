/*!
* \brief Record the basic usage of MPI_Send/MPI_Recv and MPI_ISend/MPI_IRecv.
*/

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define MASTER 0

int main (int argc, char *argv[]) {
    
  int num_tasks, rank, rc;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
  
  // Need an even number of tasks
  if (num_tasks % 2 != 0) {
    if (rank == MASTER) {
      printf("Quitting. Need an even number of tasks: num_tasks=%d\n", num_tasks);
      MPI_Abort(MPI_COMM_WORLD, rc);
      exit(1);
    }
  }
  if (rank == MASTER)
    printf("MASTER: Number of MPI tasks is: %d\n",num_tasks);
  
  // Test nonblocking MPI_Isend and MPI_Irecv.
  {
    int partner, message;
    MPI_Status stats[2];
    MPI_Request reqs[2];  
    
    // Determine partner and then send/receive with partne.
    if (rank < num_tasks/2) 
      partner = num_tasks/2 + rank;
    else if (rank >= num_tasks/2) 
      partner = rank - num_tasks/2;

    // The tag of sending and receiving shoule be the same.
    // MPI_Irecv (&buf, count, datatype, source, tag, comm, &request) 
    MPI_Irecv(&message, 1, MPI_INT, partner, 123, MPI_COMM_WORLD, &reqs[0]);
    // MPI_Isend (&buf, count, datatype, dest, tag, comm, &request) 
    MPI_Isend(&rank, 1, MPI_INT, partner, 123, MPI_COMM_WORLD, &reqs[1]);

    // Now block until requests are complete.
    // MPI_Waitall (count,&array_of_requests,&array_of_statuses) 
    MPI_Waitall(2, reqs, stats);
    
    // Print partner info and exit.
    printf("[nonblocking] Task %d is partner with %d\n", rank, message);
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  // Test blocking MPI_Send and MPI_Recv.
  {
    int partner, message;
    MPI_Status status;
    // Determine partner and then send/receive with partner.
    if (rank < num_tasks/2) {
      partner = num_tasks/2 + rank;
      // MPI_Send (&buf, count, datatype, dest, tag, comm) 
      MPI_Send(&rank, 1, MPI_INT, partner, 123, MPI_COMM_WORLD);
      MPI_Recv(&message, 1, MPI_INT, partner, 123, MPI_COMM_WORLD, &status);
    }
    else if (rank >= num_tasks/2) {
      partner = rank - num_tasks/2;
      // MPI_Recv (&buf, count, datatype, source, tag, comm, &status) 
      MPI_Recv(&message, 1, MPI_INT, partner, 123, MPI_COMM_WORLD, &status);
      MPI_Send(&rank, 1, MPI_INT, partner, 123, MPI_COMM_WORLD);
    }

    int count;
    rc = MPI_Get_count(&status, MPI_INT, &count);
    printf("[blocking] Task %d is partner with %d, received %d int(s) with tag %d\n",
           rank, message, count, status.MPI_TAG);
  }
  
  MPI_Finalize();
}
