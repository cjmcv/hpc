/*!
 * \brief A simple task consumer using threads and streams 
 *        with all data in Unified Memory.
 */

#include <iostream>
#include <time.h>
#include <vector>
#include <algorithm>
#include <thread>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cublas_v2.h>

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      fprintf(stderr, "CUDA_CHECK error in line %d of file %s \
              : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
      exit(EXIT_FAILURE); \
    } \
  } while(0);

template <typename T>
struct Task {
  unsigned int size_, id_;
  T *data_;
  T *result_;
  T *vector_;

  Task() : size_(0), id_(0), data_(NULL), result_(NULL), vector_(NULL) {};
  Task(unsigned int s) : size_(s), id_(0), data_(NULL), result_(NULL) {}
  ~Task() {}
    
  // allocate unified memory outside of constructor
  void allocate(const unsigned int size, const unsigned int unique_id) {
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

  void deallocate() {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(data_));
    CUDA_CHECK(cudaFree(result_));
    CUDA_CHECK(cudaFree(vector_));
  }
};

// simple host dgemv: assume data_ is in row-major format and square
template <typename T>
void gemv(int m, int n, T alpha, T *A, T *x, T beta, T *result_) {
  // rows
  for (int i = 0; i < n; i++) {
    result_[i] *= beta;

    for (int j = 0; j < n; j++) {
      result_[i] += A[i*n + j] * x[j];
    }
  }
}

template <typename T>
void Execute(Task<T> &t, cublasHandle_t *handle, cudaStream_t *stream, int tid) {
  if (t.size_ < 100) {
    // perform on host
    printf("Task [%d], thread [%d] executing on host (%d)\n", t.id_, tid, t.size_);

    // attach managed memory to a (dummy) stream to allow host access while the device is running
    CUDA_CHECK(cudaStreamAttachMemAsync(stream[0], t.data_, 0, cudaMemAttachHost));
    CUDA_CHECK(cudaStreamAttachMemAsync(stream[0], t.vector_, 0, cudaMemAttachHost));
    CUDA_CHECK(cudaStreamAttachMemAsync(stream[0], t.result_, 0, cudaMemAttachHost));
    // necessary to ensure Async cudaStreamAttachMemAsync calls have finished
    CUDA_CHECK(cudaStreamSynchronize(stream[0]));
    // call the host operation
    gemv(t.size_, t.size_, 1.0, t.data_, t.vector_, 0.0, t.result_);
  }
  else {
    // perform on device
    printf("Task [%d], thread [%d] executing on device (%d)\n", t.id_, tid, t.size_);
    double one = 1.0;
    double zero = 0.0;

    // attach managed memory to my stream
    cublasStatus_t status = cublasSetStream(handle[tid], stream[tid]);
    if (status != CUBLAS_STATUS_SUCCESS) {
      printf("cublasSetStream failed. \n ");
    }
    CUDA_CHECK(cudaStreamAttachMemAsync(stream[tid], t.data_, 0, cudaMemAttachSingle));
    CUDA_CHECK(cudaStreamAttachMemAsync(stream[tid], t.vector_, 0, cudaMemAttachSingle));
    CUDA_CHECK(cudaStreamAttachMemAsync(stream[tid], t.result_, 0, cudaMemAttachSingle));
    // call the device operation
    status = cublasDgemv(handle[tid], CUBLAS_OP_N, t.size_, t.size_, &one, t.data_, t.size_, t.vector_, 1, &zero, t.result_, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
      printf("cublasSetStream failed. \n ");
    }
  }
}

template <typename T>
void TaskAssignment(std::vector< Task<T> > &task_list, cublasHandle_t *handle, cudaStream_t *stream, int tid, int num_per_thread) {
  for (int i = tid*num_per_thread; i < (tid + 1)*num_per_thread && i < task_list.size(); i++) {
    printf("process: %d, ", i);
    Execute(task_list[i], handle, stream, tid);
  }
}

// populate a list of tasks with random sizes
template <typename T>
void InitialiseTasks(std::vector< Task<T> > &task_list) {
  for (unsigned int i = 0; i < task_list.size(); i++) {
    // generate random size_
    int size_ = std::max((int)((double(rand()) / RAND_MAX)*1000.0), 64);
    task_list[i].allocate(size_, i);
  }
}

template <typename T>
void ReleaseTasks(std::vector< Task<T> > &task_list) {
  printf("release task:\n");
  for (unsigned int i = 0; i < task_list.size(); i++) {
    task_list[i].deallocate();
    printf("%d, ", i);
  }
}

int InitEnvironment(const int dev_id) {
  CUDA_CHECK(cudaSetDevice(dev_id));
  cudaDeviceProp device_prop;
  cudaError_t error = cudaGetDeviceProperties(&device_prop, dev_id);
  if (device_prop.computeMode == cudaComputeModeProhibited) {
    fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
    return 1;
  }
  if (error != cudaSuccess) {
    printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
  }
  else {
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", dev_id, device_prop.name, device_prop.major, device_prop.minor);
  }
  return 0;
}

int main() {
  InitEnvironment(0);
  // randomise task sizes
  srand(time(NULL));

  // set number of threads
  const int nthreads = 4;

  // number of streams = number of threads.
  cudaStream_t *streams = new cudaStream_t[nthreads];
  cublasHandle_t *handles = new cublasHandle_t[nthreads];

  for (int i = 0; i < nthreads; i++) {
    CUDA_CHECK(cudaStreamCreate(&streams[i]));
    cublasStatus_t status = cublasCreate(&handles[i]);
    if (status != CUBLAS_STATUS_SUCCESS) {
      printf("Failed to create cublas handle. \n ");
    }
  }

  // Create list of N tasks
  unsigned int N = 40;
  std::vector<Task<double> > task_list(N);
  InitialiseTasks(task_list);

  printf("Executing tasks on host / device\n");
  std::thread *p = new std::thread[nthreads];
  int num_per_thread = (task_list.size() + nthreads - 1) / nthreads;
  for (int tid = 0; tid < nthreads; tid++) {
    p[tid] = std::thread(TaskAssignment<double>, task_list, handles, streams, tid, num_per_thread);
  }
  for (int tid = 0; tid < nthreads; tid++) {
    p[tid].join();
  }
  printf("\nFinish join() \n");

  cudaDeviceSynchronize();

  // Destroy CUDA Streams, cuBlas handles
  for (int i = 0; i < nthreads; i++) {
    cudaStreamDestroy(streams[i]);
    cublasDestroy(handles[i]);
  }

  printf("task_list.size() = %d \n", task_list.size());
  // Release tasks.
  ReleaseTasks(task_list);
  task_list.swap(std::vector<Task<double> >());

  return 0;
}