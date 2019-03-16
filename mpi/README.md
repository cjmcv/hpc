# MPI - Message Passing Interface Standard
MPI is a specification for the developers and users of message passing libraries. By itself, it is NOT a library - but rather the specification of what such a library should be.
MPI primarily addresses the message-passing parallel programming model: data is moved from the address space of one process to that of another process through cooperative operations on each process.
* Tutorials - [link](https://computing.llnl.gov/tutorials/mpi/)
* MPICH - [link](http://www.mpich.org/)
* OpenMPI - [link](https://www.open-mpi.org/software/ompi/v4.0/)

## Install
sudo apt-get install libopenmpi-dev openmpi-bin openmpi-doc

## Run
Compile: mpicxx mpi_hello.cpp -o mpi_hello  OR  mpicc mpi_hello.c -o mpi_hello

Execute: mpirun -allow-run-as-root -np 5 mpi_hello 

# mpi4py - MPI for Python
mpi4py provides Python bindings for the Message Passing Interface (MPI) standard. It is implemented on top of the MPI-1/2/3 specification and exposes an API which grounds on the standard MPI-2 C++ bindings.
* Github - [link](https://github.com/mpi4py/mpi4py)

## Install based on openmpi
Install OpenMPI: sudo apt-get install libopenmpi-dev openmpi-bin openmpi-doc

Install mpi4py : sudo pip install mpi4py

## Run 
mpirun -allow-run-as-root -n 5 python base_reduce_scan.py
---
