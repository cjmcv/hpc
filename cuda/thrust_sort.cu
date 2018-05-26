/*!
 * \brief Sort.
 */

#include <iostream>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      fprintf(stderr, "CUDA_CHECK error in line %d of file %s \
              : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
      exit(EXIT_FAILURE); \
    } \
  } while(0);

void TestDataInCPU() {
  /////////////////////////////////////////////////////
  {
    const int N = 6;
    int arr[N] = { 1, 4, 2, 8, 5, 7 };

    thrust::sort(arr, arr + N);

    for (int i = 0; i < N; i++)
      std::cout << arr[i] << ", ";
  }
  std::cout << std::endl << "Finish Test1." << std::endl;

  /////////////////////////////////////////////////////
  {
    const int N = 6;
    int arr[N] = { 1, 4, 2, 8, 5, 7 };

    thrust::stable_sort(arr, arr + N, thrust::greater<int>());

    for (int i = 0; i < N; i++)
      std::cout << arr[i] << ", ";
  }
  std::cout << std::endl << "Finish Test2." << std::endl;

  /////////////////////////////////////////////////////
  {
    const int N = 6;
    int    keys[N] = { 1,   4,   2,   8,   5,   7 };
    char values[N] = { 'a', 'b', 'c', 'd', 'e', 'f' };
    thrust::sort_by_key(keys, keys + N, values);
    for (int i = 0; i < N; i++)
      std::cout << values[i] << "(" << keys[i] << "), ";
    // keys is now   {  1,   2,   4,   5,   7,   8}
    // values is now {'a', 'c', 'b', 'e', 'f', 'd'}
  }
  std::cout << std::endl << "Finish Test3." << std::endl;
}

void TestDataInGPU() {
  ////////////////////////////////////////
  {
    const int N = 6;
    int arr[N] = { 1, 4, 2, 8, 5, 7 };

    thrust::device_vector<int> d_x(6, 1);
    for (int i = 0; i < N; i++)
      d_x[i] = arr[i];

    thrust::sort(d_x.begin(), d_x.end());
    for (int i = 0; i < N; i++)
      std::cout << d_x[i] << ", ";
  }
  std::cout << std::endl << "Finish Test4." << std::endl;

  /////////////////////////////////////////
  {
    const int N = 6;
    int arr[N] = { 1, 4, 2, 8, 5, 7 };

    int *d_x;
    CUDA_CHECK(cudaMalloc((void **)&d_x, sizeof(int) * N));
    CUDA_CHECK(cudaMemcpy(d_x, arr, sizeof(int) * N, cudaMemcpyHostToDevice));

    thrust::device_ptr <int > dev_ptr(d_x);
    thrust::sort(dev_ptr, dev_ptr +N);

    CUDA_CHECK(cudaMemcpy(arr, d_x, sizeof(int) * N, cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++)
      std::cout << arr[i] << ", ";

    CUDA_CHECK(cudaFree(d_x));
  }
  std::cout << std::endl << "Finish Test5." << std::endl;
}

int main(void) {
  TestDataInCPU();

  TestDataInGPU();
  
  return 0;
}
