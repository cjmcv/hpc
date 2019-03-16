/*!
* \brief Send and receive custom types of data 
*     by using MPI_Type_struct.
*/

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct{
  char a;
  int b;
  float c[2];
  double d[4];
} MyStruct;
    
void CheckMyStruct(std::string prefix, MyStruct data, int rank, int num_tasks) {
  for (int i=0; i<num_tasks; i++) {
    if (rank == i) {
      std::cout << prefix << "-[rank = " << rank << "] : ";
      printf("<%c, %d, (%f, %f), (%f, %f, %f, %f)>", 
             data.a, data.b, data.c[0], data.c[1], 
             data.d[0], data.d[1], data.d[2], data.d[3]);  
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
  // Create a new type for MyStruct.
  MPI_Datatype old_type[4]= {MPI_CHAR, MPI_INT, MPI_FLOAT, MPI_DOUBLE};
  int block_count[4]= {1,1,2,4};
    
  MPI_Aint head_address;
  MPI_Aint offset[4];
    
  MyStruct temp_struct;
  MPI_Address(&temp_struct, &head_address);
  MPI_Address(&(temp_struct.a), &offset[0]);
  MPI_Address(&(temp_struct.b), &offset[1]);
  MPI_Address(temp_struct.c, &offset[2]);
  MPI_Address(temp_struct.d, &offset[3]);

  for (int i=0; i<4; i++)
    offset[i] -= head_address;
  
  MPI_Datatype MY_STRUCT;
  MPI_Type_struct(4, block_count, offset, old_type, &MY_STRUCT);
  MPI_Type_commit(&MY_STRUCT);
  /////////////////////////////////////////////////////////////////
  
  // Prepare data.
  MyStruct data;    
  if (rank == 0) {
    data.a = 'a';
    data.b = 1;
    for (int i=0; i<block_count[2]; i++) 
      data.c[i] = i;
    for (int i=0; i<block_count[3]; i++)
      data.d[i] = i;    
  }
    
  // Check.
  CheckMyStruct("MPI_Type_struct(before)", data, rank, num_tasks);
    
  if (rank == 0)
    MPI_Send(&data, 1, MY_STRUCT, 1, 123, MPI_COMM_WORLD);
  else
    MPI_Recv(&data, 1, MY_STRUCT, 0, 123, MPI_COMM_WORLD, &status);
    
  // Check the received data.
  CheckMyStruct("MPI_Type_struct(after)", data, rank, num_tasks);
    
  MPI_Type_free(&MY_STRUCT); 

  MPI_Finalize();
}