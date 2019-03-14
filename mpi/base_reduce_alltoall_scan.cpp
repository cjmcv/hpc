/*!
* \brief Record the basic usage of Reduce, Allreduce, Alltoall, Scan and Exscan.
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
  
  // Prepare data
  int send_len = 4;
  int recv_len = send_len;
    
  float *send_buffer = (float *)malloc(sizeof(float) * send_len);
  for (int i=0; i<send_len; i++) 
    send_buffer[i] = rank * send_len + i;
  
  float *recv_buffer = (float *)malloc(sizeof(float) * recv_len);
    
  // Reduce.
  {   
    memset(recv_buffer, 0, sizeof(float) * recv_len);
    
    // MPI_Reduce (&sendbuf, &recvbuf, count, datatype, op, root, comm)   
    PrintArrayInRankOrder<float>("MPI_Reduce(send)", send_buffer, send_len, 
                                 rank, num_tasks);
    MPI_Reduce(send_buffer, recv_buffer, send_len, 
               MPI_FLOAT, MPI_SUM, MASTER, MPI_COMM_WORLD);
    PrintArrayInRankOrder<float>("MPI_Reduce(recv)", recv_buffer, recv_len, 
                                 rank, num_tasks);
    
    // Clear.
    memset(recv_buffer, 0, sizeof(float) * recv_len);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // MPI_Allreduce (&sendbuf, &recvbuf, count, datatype, op, comm)
    PrintArrayInRankOrder<float>("MPI_Allreduce(send)", send_buffer, send_len, 
                                 rank, num_tasks);
    MPI_Allreduce(send_buffer, recv_buffer, send_len, 
                  MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    PrintArrayInRankOrder<float>("MPI_Allreduce(recv)", recv_buffer, recv_len, 
                                 rank, num_tasks);
  }
  
  // MPI_Alltoall: Scatter data from all tasks to all tasks in communicator.
  {
    memset(recv_buffer, 0, sizeof(float) * recv_len);
    
    // MPI_Alltoall (&sendbuf, sendcount, sendtype, &recvbuf, recvcnt, recvtype, comm) 
    // Note: sendcount and recvcnt means segment length.
    PrintArrayInRankOrder<float>("MPI_Alltoall(send)", send_buffer, send_len, 
                                 rank, num_tasks);
    MPI_Alltoall(send_buffer, send_len / num_tasks, MPI_FLOAT, 
                 recv_buffer, send_len / num_tasks, MPI_FLOAT, MPI_COMM_WORLD);
    PrintArrayInRankOrder<float>("MPI_Alltoall(recv)", recv_buffer, recv_len, 
                                 rank, num_tasks);
  }
  
  // Scan
  {
    memset(recv_buffer, 0, sizeof(float) * recv_len);
    
    // MPI_Scan (&sendbuf, &recvbuf, count, datatype, op, comm) 
    PrintArrayInRankOrder<float>("MPI_Scan(send)", send_buffer, send_len, 
                                 rank, num_tasks);
    MPI_Scan(send_buffer, recv_buffer, send_len, MPI_FLOAT, 
             MPI_SUM, MPI_COMM_WORLD);
    PrintArrayInRankOrder<float>("MPI_Scan(recv)", recv_buffer, recv_len, 
                                 rank, num_tasks);
    
    ////////////////////
    
    memset(recv_buffer, 0, sizeof(float) * recv_len);
    
    // MPI_Exscan (&sendbuf, &recvbuf, count, datatype, op, comm) 
    PrintArrayInRankOrder<float>("MPI_Exscan(send)", send_buffer, send_len, 
                                 rank, num_tasks);
    MPI_Exscan(send_buffer, recv_buffer, send_len, MPI_FLOAT, 
             MPI_SUM, MPI_COMM_WORLD);
    PrintArrayInRankOrder<float>("MPI_Exscan(recv)", recv_buffer, recv_len, 
                                 rank, num_tasks);                             
  }
      
  free(send_buffer);
  free(recv_buffer);
    
  MPI_Finalize();
}