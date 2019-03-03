#!/usr/bin/python
# -*- coding : utf-8 -*-
#
# Record the basic usage of Send and Recv.

import numpy as np
from mpi4py import MPI
    
if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numprocs = comm.Get_size()
    if numprocs < 2:
        print("ERROR: Running in %d processes, \
        But this demo should be run in more than 2 processes!" % numprocs)
    if rank == 0:
        print("Running in %d processes" % numprocs)
    comm.Barrier()
    
    N = 10
    source_process = 0
    dest_process = 1  
    # ------------------------------------------------------------------
    # Version1: Passing MPI datatypes explicitly -> [data, MPI.INT]
    if rank == source_process:
        data = np.arange(N, dtype='i')
        print 'process %d sends %s' % (rank, data)
        # The tag of sending and receiving shoule be the same.
        comm.Send([data, MPI.INT], dest=dest_process, tag=123)
    elif rank == dest_process:
        data = np.empty(N, dtype='i')
        comm.Recv([data, MPI.INT], source=source_process, tag=123)
        print 'process %d receives %s' % (rank, data)
    
    # ------------------------------------------------------------------
    # Version2: Automatic MPI datatype discovery
    if rank == source_process:
        # dtype can be 'f', 'i' or np.float64, etc.
        data = np.arange(N, dtype=np.float64)
        print 'process %d sends %s' % (rank, data)
        comm.Send(data, dest=dest_process, tag=345)
    elif rank == dest_process:
        data = np.empty(N, dtype=np.float64)
        comm.Recv(data, source=source_process, tag=345)
        print 'process %d receives %s' % (rank, data)