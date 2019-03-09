#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Record the basic usage of Bcast, Scatter, Gather and Allgather.

import numpy as np
from mpi4py import MPI
    
def PrintInMPIRank(comm, print_prefix, subject):
    for r in range(comm.size):
        if comm.rank == r:
            print("%s: [%d] %s" % (print_prefix, comm.rank, subject))
        comm.Barrier()

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    # comm.rank is the same as comm.Get_rank().
    #     Most functions in mpi4py come in both lowercase and uppercase 
    # versions, such as send()/recv() and Send()/Recv(), rank and Get_rank().
    #     The uppercase version has buffer, which is closer to C++ 
    # and more efficient. And the lowercase version is easier to use.
    rank = comm.Get_rank() 
    if rank == 0:
        print("Running in %d processes" % comm.size)

    # Wait for everybody to synchronize _here_.
    comm.Barrier()

    # Prepare a vector to be broadcasted.
    Bn = 3
    An = Bn * comm.size
    if rank == 0:
        A = np.arange(An, dtype=np.float64)    # rank 0 has proper data
    else:
        A = np.empty(An, dtype=np.float64)     # all other just an empty array

    PrintInMPIRank(comm, "Init", A)

    # ----------------------------------------------------------
    # Broadcast A from rank 0 to everybody, means root=0.
    # Default: comm.Bcast( [A, MPI.DOUBLE], root=0 )
    comm.Bcast( [A, MPI.DOUBLE] )
    PrintInMPIRank(comm, "Broadcast from rank0", A)
    
    # ----------------------------------------------------------
    B = np.empty(Bn, dtype=np.float64)    
    # Scatter data into B arrays.
    comm.Scatter( [A, MPI.DOUBLE], [B, MPI.DOUBLE] )
    PrintInMPIRank(comm, "Scatter from rank0", B)

    # Everybody is multiplying by 2.
    B *= 2
    
    # ----------------------------------------------------------
    # Gather B data into A rank0.
    # Default root=0.
    comm.Gather( [B, MPI.DOUBLE], [A, MPI.DOUBLE], root=1 )
    PrintInMPIRank(comm, "Gather", A)
    
    # ----------------------------------------------------------
    # Allgather B data into A.
    # Allgather = Gather + Bcast.
    comm.Allgather( [B, MPI.DOUBLE], [A, MPI.DOUBLE] )
    PrintInMPIRank(comm, "Allgather", A)