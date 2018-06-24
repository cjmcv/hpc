/*!
 * \brief A simple task consumer using threads and streams 
 *        with all data in Unified Memory.
 */

#include <iostream>
#include <time.h>
#include <vector>
#include <algorithm>
#include <thread>

#include "cuda_util.h"
#include <cublas_v2.h>

 // Simple host dgemv: assume data_ is in row-major format and square
template <typename T>
void gemv(int m, int n, T alpha, T *A, T *x, T beta, T *result) {
  // rows
  for (int i = 0; i < n; i++) {
    result[i] *= beta;

    for (int j = 0; j < n; j++) {
      result[i] += A[i*n + j] * x[j];
    }
  }
}

template <typename T>
class Task {
public:
  Task() : size_(0), id_(0), data_(NULL), result_(NULL), vector_(NULL) {};
  ~Task() {}
    
  // Perform on host 
  void ExecuteOnHost(cudaStream_t stream) {
    // attach managed memory to a (dummy) stream to allow host access while the device is running
    CUDA_CHECK(cudaStreamAttachMemAsync(stream, data_, 0, cudaMemAttachHost));
    CUDA_CHECK(cudaStreamAttachMemAsync(stream, vector_, 0, cudaMemAttachHost));
    CUDA_CHECK(cudaStreamAttachMemAsync(stream, result_, 0, cudaMemAttachHost));
    // necessary to ensure Async cudaStreamAttachMemAsync calls have finished
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // call the host operation
    gemv(size_, size_, 1.0, data_, vector_, 0.0, result_);
  }

  // Perform on device
  void ExecuteOnDevice(cublasHandle_t handle, cudaStream_t stream) {
    double one = 1.0;
    double zero = 0.0;

    // Attach managed memory to my stream
    cublasStatus_t status = cublasSetStream(handle, stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
      printf("cublasSetStream failed. \n ");
    }
    CUDA_CHECK(cudaStreamAttachMemAsync(stream, data_, 0, cudaMemAttachSingle));
    CUDA_CHECK(cudaStreamAttachMemAsync(stream, vector_, 0, cudaMemAttachSingle));
    CUDA_CHECK(cudaStreamAttachMemAsync(stream, result_, 0, cudaMemAttachSingle));
    // Call the device operation
    status = cublasDgemv(handle, CUBLAS_OP_N, size_, size_, &one, data_, size_, vector_, 1, &zero, result_, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
      printf("cublasSetStream failed. \n ");
    }
  }

  // Allocate unified memory.
  void Allocate(const unsigned int size, const unsigned int unique_id) {
    id_ = unique_id;
    size_ = size;
    CUDA_CHECK(cudaMallocManaged(&data_, sizeof(T)*size_*size_));
    CUDA_CHECK(cudaMallocManaged(&result_, sizeof(T)*size_));
    CUDA_CHECK(cudaMallocManaged(&vector_, sizeof(T)*size_));
    CUDA_CHECK(cudaDeviceSynchronize());

    // populate data_ with random elements
    for (int i = 0; i < size_*size_; i++) {
      data_[i] = double(rand()) / RAND_MAX;
    }

    for (int i = 0; i < size_; i++) {
      result_[i] = 0.;
      vector_[i] = double(rand()) / RAND_MAX;
    }
  }

  void Deallocate() {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(data_));
    CUDA_CHECK(cudaFree(result_));
    CUDA_CHECK(cudaFree(vector_));
  }

  inline int get_size() { return size_; }

private:
  T *data_;
  T *result_;
  T *vector_;  
  unsigned int size_, id_;
};

template <typename T>
void TaskAssignment(std::vector< Task<T> > &task_list, cublasHandle_t *handle, 
  cudaStream_t *stream, int tid, int num_per_thread) {
  for (int i = tid*num_per_thread;
    i < (tid + 1)*num_per_thread && i < task_list.size();
    i++) {
    int size = task_list[i].get_size();
    if (size < 100) {
      printf("Task [%d], thread [%d] executing on host (%d)\n", i, tid, size);
      task_list[i].ExecuteOnHost(stream[0]);
    }
    else {
      printf("Task [%d], thread [%d] executing on device (%d)\n", i, tid, size);
      task_list[i].ExecuteOnDevice(handle[tid], stream[tid]);
    }
  }
}


int main() {
  int ret = cjmcv_cuda_util::InitEnvironment(0);
  if (ret != 0) {
    printf("Failed to initialize the environment for cuda.");
    return -1;
  }
  srand(time(NULL));

  // Number of threads
  const int nthreads = 4;

  // Create a cuda stream and a cublas handle for each thread.
  cudaStream_t *streams = new cudaStream_t[nthreads];
  cublasHandle_t *handles = new cublasHandle_t[nthreads];

  for (int i = 0; i < nthreads; i++) {
    CUDA_CHECK(cudaStreamCreate(&streams[i]));
    cublasStatus_t status = cublasCreate(&handles[i]);
    if (status != CUBLAS_STATUS_SUCCESS) {
      printf("Failed to create cublas handle. \n ");
    }
  }

  // Create list of tasks
  std::vector<Task<double> > task_list(40);
  for (int i = 0; i < task_list.size(); i++) {
    // Allocate with random sizes.
    int size = std::max((int)((double(rand()) / RAND_MAX)*1000.0), 64);
    task_list[i].Allocate(size, i);
  }

  printf("Executing tasks on host / device\n");
  std::thread *thread_list = new std::thread[nthreads];
  int num_per_thread = (task_list.size() + nthreads - 1) / nthreads;
  for (int tid = 0; tid < nthreads; tid++) {
    // Can not release memory in the destructor of Task ?
    thread_list[tid] = std::thread(TaskAssignment<double>, task_list, handles, streams, tid, num_per_thread);
  }
  for (int tid = 0; tid < nthreads; tid++) {
    thread_list[tid].join();
  }
  delete[] thread_list;
  printf("\nFinish excuting tasks. \n");

  cudaDeviceSynchronize();
  // Destroy CUDA Streams and cuBlas handles.
  for (int i = 0; i < nthreads; i++) {
    cudaStreamDestroy(streams[i]);
    cublasDestroy(handles[i]);
  }

  // Release tasks.
  for (int i = 0; i < task_list.size(); i++) {
    task_list[i].Deallocate();
  }
  task_list.swap(std::vector<Task<double> >());

  cjmcv_cuda_util::CleanUpEnvironment();
  return 0;
}