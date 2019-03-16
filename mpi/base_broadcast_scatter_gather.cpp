/*!
* \brief Record the basic usage of Bcast, Scatter, Gather and Allgather.
*/

#include "mpi.h"
#include <stdio.h>
#include<stdlib.h>

#define MASTER 0

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

int main(int argc, char *argv[]) {
  
  int num_tasks, rank; 
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
  
  // MPI_Bcast
  {
    int len = 8;
    float *buffer = (float *)malloc(sizeof(float) * len);
    if (rank == MASTER) {
      for(int i=0; i<len; i++)
		buffer[i] = i;
    }
    else
      memset(buffer, 0, sizeof(float) * len);
  
    PrintArrayInRankOrder<float>("MPI_Bcast(before)", buffer, len, 
                                 rank, num_tasks);
    
    // MPI_Bcast (&buffer, count, datatype, root, comm) 
    MPI_Bcast(buffer, len, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
    
    PrintArrayInRankOrder<float>("MPI_Bcast(after)", buffer, len, 
                                 rank, num_tasks);
    
    free(buffer);
  }
  
  // MPI_Scatter. 
  {   
    // Note the len of the send_buffer and recv_buffer.
    int recv_len = 4;  
    int send_len = num_tasks * recv_len;  
    
    float *send_buffer = (float *)malloc(sizeof(float) * send_len);
    if (rank == MASTER) {
      for (int i=0; i<send_len; i++) {
        send_buffer[i] = i;
      }
    }
    else
      memset(send_buffer, 0, sizeof(float) * send_len);
   
    float *recv_buffer = (float *)malloc(sizeof(float) * recv_len);
    memset(recv_buffer, 0, sizeof(float) * recv_len);
  
    PrintArrayInRankOrder<float>("MPI_Scatter(send)", send_buffer, send_len,
                                 rank, num_tasks);
    // MPI_Scatter (&sendbuf, sendcnt, sendtype, &recvbuf, recvcnt,
    //              recvtype, root, comm)
    // Note: The sendcnt and recvcnt should pick the shorter one between 
    //       send_len and recv_len.
    MPI_Scatter(send_buffer, recv_len, MPI_FLOAT, recv_buffer, recv_len,
                MPI_FLOAT, MASTER, MPI_COMM_WORLD);
    
    PrintArrayInRankOrder<float>("MPI_Scatter(recv)", recv_buffer, recv_len, 
                                 rank, num_tasks);
    
    free(send_buffer);
    free(recv_buffer);
  }
  
  // MPI_Gather and MPI_Allgather
  {
    // Note the len of the send_buffer and recv_buffer.
    int send_len = 4;
    int recv_len = num_tasks * send_len;
    
    float *send_buffer = (float *)malloc(sizeof(float) * send_len);
    for (int i=0; i<send_len; i++)
      send_buffer[i] = i + rank * send_len;
  
    float *recv_buffer = (float *)malloc(sizeof(float) * recv_len);
    memset(recv_buffer, 0, sizeof(float) * recv_len);
    
    PrintArrayInRankOrder<float>("MPI_Gather(send)", send_buffer, send_len, 
                                 rank, num_tasks);
    // MPI_Gather (&sendbuf, sendcnt, sendtype, &recvbuf, recvcnt, 
    //              recvtype, root, comm) 
    MPI_Gather(send_buffer, send_len, MPI_FLOAT, recv_buffer, send_len,
               MPI_FLOAT, MASTER, MPI_COMM_WORLD);
    
    PrintArrayInRankOrder<float>("MPI_Gather(recv)", recv_buffer, recv_len, 
                                 rank, num_tasks);
    
    ////////////////
    
    PrintArrayInRankOrder<float>("MPI_Allgather(send)", send_buffer, send_len, 
                                 rank, num_tasks);
    // MPI_Allgather (&sendbuf, sendcnt, sendtype, &recvbuf, recvcnt, 
    //                recvtype, comm) 
    MPI_Allgather(send_buffer, send_len, MPI_FLOAT, recv_buffer, send_len,
                  MPI_FLOAT, MPI_COMM_WORLD);
    
    PrintArrayInRankOrder<float>("MPI_Allgather(recv)", recv_buffer, recv_len, 
                                 rank, num_tasks);
    
    free(send_buffer);
    free(recv_buffer);
  }
  MPI_Finalize();
}