/*!
 * \brief Record the basic usage of Iterators in Thrust.
 * \Source https://docs.nvidia.com/cuda/thrust/
 * \iterator constant_iterator, 
 *           counting_iterator,
 *           transform_iterator,  
 *           permutation_iterator,
 *           zip_iterator.
 */

#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

void TestConstantIterator() {
  std::cout << "TestConstantIterator: "<< std::endl;
  thrust::constant_iterator<int> first(10);
  thrust::constant_iterator<int> last = first + 3;

  std::cout << "first[0] = " << first[0] << std::endl;  // returns 10
  std::cout << "first[1] = " << first[1] << std::endl;  // returns 10
  std::cout << "first[100] = " << first[100] << std::endl;  // returns 10

  // sum of [first, last)
  int sum = thrust::reduce(first, last);
  std::cout << "sum = " << sum << std::endl; // returns 10 * 3
}

void TestCountingIterator() {
  std::cout << "TestCountingIterator: " << std::endl;
  thrust::counting_iterator<int> first(10);
  thrust::counting_iterator<int> last = first + 3;

  std::cout << "first[0] = " << first[0] << std::endl;  // returns 10
  std::cout << "first[1] = " << first[1] << std::endl;  // returns 11=10+1
  std::cout << "first[100] = " << first[100] << std::endl;  // returns 110=10+100

  int sum = thrust::reduce(first, last);
  std::cout << "sum = " << sum << std::endl; // returns 33=10+11+12
}

void TestTransformIterator() {
  std::cout << "TestTransformIterator: " << std::endl;
  thrust::device_vector<int> vec(3);
  vec[0] = 10; vec[1] = 20; vec[2] = 30;

  auto first = thrust::make_transform_iterator(vec.begin(), thrust::negate<int>());
  auto last = thrust::make_transform_iterator(vec.end(), thrust::negate<int>());

  std::cout << "first[0] = " << first[0] << std::endl;  // returns -10
  std::cout << "first[1] = " << first[1] << std::endl;  // returns -20
  std::cout << "first[2] = " << first[2] << std::endl;  // returns -30

  int sum = thrust::reduce(first, last);
  std::cout << "sum = " << sum << std::endl; // returns -60 (-10 + -20 + -30)
}

void TestPermutationIterator() {
  std::cout << "TestPermutationIterator: " << std::endl;
  // Gather locations
  thrust::device_vector<int> map(4);
  map[0] = 3;
  map[1] = 1;
  map[2] = 0;
  map[3] = 5;

  // Array to gather from
  thrust::device_vector<int> source(6);
  source[0] = 10;
  source[1] = 20;
  source[2] = 30;
  source[3] = 40;
  source[4] = 50;
  source[5] = 60;

  // Fuse gather with reduction: 
  // sum = source[map[0]] + source[map[1]] + ...
  int sum = thrust::reduce(thrust::make_permutation_iterator(source.begin(), map.begin()),
    thrust::make_permutation_iterator(source.begin(), map.end()));
  std::cout << "sum = " << sum << std::endl;
}

void TestZipIterator() {
  std::cout << "TestZipIterator: " << std::endl;

  thrust::device_vector<int>  A(3);
  thrust::device_vector<char> B(3);
  A[0] = 10;  A[1] = 20;  A[2] = 30;
  B[0] = 'x'; B[1] = 'y'; B[2] = 'z';

  auto first = thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin()));
  auto last = thrust::make_zip_iterator(thrust::make_tuple(A.end(), B.end()));

  std::cout << "first[0] = (" << thrust::get<0>(first[0]) << ", " << thrust::get<1>(first[0]) << ")" << std::endl;  // returns 10,x
  std::cout << "first[1] = (" << thrust::get<0>(first[1]) << ", " << thrust::get<1>(first[1]) << ")" << std::endl;  // returns 20,y
  std::cout << "first[2] = (" << thrust::get<0>(first[2]) << ", " << thrust::get<1>(first[2]) << ")" << std::endl;  // returns 30,z

  // maximum of [first, last)
  thrust::maximum< thrust::tuple<int, char> > binary_op;
  thrust::tuple<int, char> init = first[0];
  auto res = thrust::reduce(first, last, init, binary_op); // returns tuple(30, 'z')
  std::cout << "res = (" << thrust::get<0>(res) << ", " << thrust::get<1>(res) << ")" << std::endl;  // returns 30,z
}

int main(void) {
  TestConstantIterator();
  std::cout << "##############################" << std::endl << std::endl;
  
  TestCountingIterator();
  std::cout << "##############################" << std::endl << std::endl;

  TestTransformIterator();
  std::cout << "##############################" << std::endl << std::endl;

  TestPermutationIterator();
  std::cout << "##############################" << std::endl << std::endl;

  TestZipIterator();
  std::cout << "##############################" << std::endl << std::endl;
  return 0;
}
