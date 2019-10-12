# CUDA - Compute Unified Device Architecture
CUDA® is a parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs). 
* [Zone](https://developer.nvidia.com/cuda-zone)
* [Toolkit](https://developer.nvidia.com/cuda-toolkit)
* [GPU Specs Database](https://www.techpowerup.com/gpu-specs/)

## cuBLAS
The NVIDIA cuBLAS library is a fast GPU-accelerated implementation of the standard basic linear algebra subroutines (BLAS). Using cuBLAS APIs, you can speed up your applications by deploying compute-intensive operations to a single GPU or scale up and distribute work across multi-GPU configurations efficiently.
* [Main Page](https://developer.nvidia.com/cublas)
* [Doc](https://docs.nvidia.com/cuda/index.html)

## cuDNN
cuDNN is a GPU-accelerated library of primitives for deep neural networks, provides highly tuned implementations for standard routines such as forward and backward convolution, pooling, normalization, and activation layers.
* [Main Page](https://developer.nvidia.com/cudnn)

## cuSOLVER
The NVIDIA cuSOLVER library provides a collection of dense and sparse direct solvers which deliver significant acceleration for Computer Vision, CFD, Computational Chemistry, and Linear Optimization applications.
* [Main Page](https://developer.nvidia.com/cusolver)
* [Doc](https://docs.nvidia.com/cuda/cusolver/)

## Thrust
Thrust is a powerful library of parallel algorithms and data structures. Thrust provides a flexible, high-level interface for GPU programming that greatly enhances developer productivity. Using Thrust, C++ developers can write just a few lines of code to perform GPU-accelerated sort, scan, transform, and reduction operations orders of magnitude faster than the latest multi-core CPUs. 
* [Main Page](https://developer.nvidia.com/thrust)
* [Github](https://github.com/thrust/thrust)
* [Doc](https://docs.nvidia.com/cuda/thrust/)

## CUB
CUB provides state-of-the-art, reusable software components for every layer of the CUDA programming model: Parallel primitives(Warp-wide, Block-wide, Device-wide) and Utilities.
* [Main Page](http://nvlabs.github.io/cub/)
* [Github](https://github.com/NVlabs/cub)

## NCCL
NCCL (pronounced "Nickel") is a stand-alone library of standard collective communication routines for GPUs, implementing all-reduce, all-gather, reduce, broadcast, and reduce-scatter. It has been optimized to achieve high bandwidth on platforms using PCIe, NVLink, NVswitch, as well as networking using InfiniBand Verbs or TCP/IP sockets. NCCL supports an arbitrary number of GPUs installed in a single node or across multiple nodes, and can be used in either single- or multi-process (e.g., MPI) applications.
* [Main Page](https://developer.nvidia.com/taxonomy/term/784)
* [Github](https://github.com/NVIDIA/nccl)
* [Demo](https://github.com/NVIDIA/nccl-tests)
---
