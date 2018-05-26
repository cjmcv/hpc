/*!
* \brief Record the basic usage of Transformations in Thrust.
*        https://docs.nvidia.com/cuda/thrust/
* \Functor plus: +; 
*          minus: -;
*          multiplies: *; 
*          equal_to: ==;
*          less: <;
*/

#include <iostream>
#include "time.h"
// Thrust related
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

////////// Test Saxpy ////////////
struct saxpy_functor {
  const float alpha_;

  saxpy_functor(float a) : alpha_(a) {}

  __host__ __device__
    float operator()(const float& x, const float& y) const {
      return alpha_ * x + y;
  }
};

void saxpy_fast(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y) {
  // Y <- A * X + Y
  thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}

void saxpy_slow(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y) {
  thrust::device_vector<float> temp(X.size());

  // temp <- A
  thrust::fill(temp.begin(), temp.end(), A);

  // temp <- A * X
  thrust::transform(X.begin(), X.end(), temp.begin(), temp.begin(), thrust::multiplies<float>());

  // Y <- A * X + Y
  thrust::transform(temp.begin(), temp.end(), Y.begin(), Y.begin(), thrust::plus<float>());
}

void TestSaxpy() {
  // allocate three device_vectors with 10 elements
  thrust::device_vector<float> d_x(10);
  thrust::device_vector<float> d_y(10);

  // initialize X to 0,1,2,3, ....
  thrust::sequence(d_x.begin(), d_x.end(), 0);
  thrust::fill(d_y.begin(), d_y.end(), 1);

  float alpha = 2.0;
  
  time_t t = clock();
  saxpy_slow(alpha, d_x, d_y);
  printf("saxpy_slow, msec_total = %lld\n", clock() - t);

  for (int i = 0; i < d_y.size(); i++)
    std::cout << "y[" << i << "] = " << d_y[i] << std::endl;

  thrust::fill(d_y.begin(), d_y.end(), 1);

  t = clock();
  saxpy_fast(alpha, d_x, d_y);
  printf("saxpy_fast, msec_total = %lld\n", clock() - t);
  for (int i = 0; i < d_y.size(); i++)
    std::cout << "y2[" << i << "] = " << d_y[i] << std::endl;
}

////////// Test Reduce ////////////
// square<T> computes the square of a number f(x) -> x*x
template <typename T>
struct square {
  __host__ __device__
    T operator()(const T& x) const {
    return x * x;
  }
};

void TestReduce() {
  float x[4] = { 1.0, 2.0, 3.0, 4.0 };

  // transfer to device
  thrust::device_vector<float> d_x(x, x + 4);

  // setup arguments
  float init = 0;
  float sum = thrust::transform_reduce(d_x.begin(), d_x.end(), square<float>(), init, thrust::plus<float>());
  std::cout << "sum: " << sum << std::endl;
}

////////// Test Scan ////////////
void TestScan() {
  const int LEN = 12;
  int data[LEN] = { 1, 0, 2, 2, 1, 3, 7, 8, 12, 36, 15, 6 };
  thrust::inclusive_scan(data, data + LEN, data);

  std::cout << "Scan Result: " << std::endl;
  for (int i = 0; i < LEN; i++) {
    std::cout << data[i] << ", ";
  }
  std::cout << std::endl;
}

int main() {

  std::cout << "Start TestSaxpy()." << std::endl;
  TestSaxpy();

  std::cout << "Start TestReduce()." << std::endl;
  TestReduce();

  std::cout << "Start TestScan()." << std::endl;
  TestScan();

  system("pause");
  return 0;
}
