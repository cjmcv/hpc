#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Record the basic usage of Reduce and Scan.
# Reduce(self, sendbuf, recvbuf, Op op=SUM, int root=0)
# Scan(self, sendbuf, recvbuf, Op op=SUM)

import numpy as np
from mpi4py import MPI
    
def PrintInMPIRank(comm, print_prefix, subject):
    for r in range(comm.size):
        if comm.rank == r:
            print("%s[%d] %s" % (print_prefix, comm.rank, subject))
        comm.Barrier()

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() 
    if rank == 0:
        print("Running in %d processes" % comm.size)
    comm.Barrier()

    # ------------------------------------------------------------------------------
    # Prepare data.
    A = np.array([0, 1], dtype='i') + 2 * rank 
    PrintInMPIRank(comm, "A", A)

    if rank == 0:
        # Initialize B with [-1, -1] of rank0
        B = np.zeros(2, dtype='i') - 1
    else:
        B = None
    PrintInMPIRank(comm, "B", B)
    
    # ------------------------------------------------------------------------------
    # Reduce(comm.size == 5): [-1, -1] + [2, 3] + [4, 5] + [6, 7] + [8 9] = [19, 23]
    if rank == 0:
        # MPI.IN_PLACE: Use the data of B, and ignore A in rank0.
        # root=0: The result will be saved in rank0. 
        comm.Reduce(MPI.IN_PLACE, B, op=MPI.SUM, root=0)
    else:
        # Ignore the data of B.
        comm.Reduce(A, B, op=MPI.SUM, root=0)
    PrintInMPIRank(comm, "Reduce with IN_PLACE", B)
    
    # ------------------------------------------------------------------------------
    # Reduce(comm.size == 5): [0, 1] + [2, 3] + [4, 5] + [6, 7] + [8 9] = [20, 25]
    if rank == 1:
        B = np.zeros(2, dtype='i')
    comm.Reduce(A, B, op=MPI.SUM, root=1)
    PrintInMPIRank(comm, "Reduce", B)
    
    # ------------------------------------------------------------------------------
    # Scan
    C = np.zeros(2, dtype='i') - 1
    comm.Scan(A, C, op=MPI.SUM)
    PrintInMPIRank(comm, "Scan", C)
    
    # ------------------------------------------------------------------------------
    # Exscan
    D = np.zeros(2, dtype='i') - 1
    comm.Exscan(A, D, op=MPI.SUM)
    PrintInMPIRank(comm, "Exscan", D)