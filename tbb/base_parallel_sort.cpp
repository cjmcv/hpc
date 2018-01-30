/*!
* \brief The basic use of parallel_sort.
*/
#include <iostream>
#include <tbb/tbb.h>

int main() {
  const int N = 100;
  float *arr = new float[N];
  for (int i = 0; i < N; i++) {
    arr[i] = i % 10;
  }
  tbb::parallel_sort(arr, arr + N); // Use std::less< T >() by default, small -> big
  //tbb::parallel_sort(arr, arr + N, std::greater<float>()); // big -> small

  for (int i = 0; i < N; i++) {
    printf("%f, ", arr[i]);
  }

  delete[] arr;
}