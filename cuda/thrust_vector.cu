/*!
* \brief Record the basic usage of Vector in Thrust.
*/

#include <iostream>
#include "time.h"
#include <list>
// Thrust related
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

void VectorTest1() {
  // H has storage for 4 integers
  thrust::host_vector<int> H(4);

  // initialize individual elements
  for (int i = 0; i < H.size(); i++)
    H[i] = i * 10 + 1;

  // print contents of H
  for (int i = 0; i < H.size(); i++)
    std::cout << "H[" << i << "] = " << H[i] << std::endl;

  H.resize(2);

  // Copy host_vector H to device_vector D
  thrust::device_vector<int> D = H;

  // elements of D can be modified
  for (int i = 0; i < D.size(); i++)
    D[i] = i * 100 + 1;

  // print contents of D
  for (int i = 0; i < D.size(); i++) {
    std::cout << "D[" << i << "] = " << D[i] << std::endl;
    //printf("Dp[%d] = %d.\n", i, D[i]);  //Can not print the right number in this way.
  }

  // H and D are automatically deleted when the function returns.
}

void VectorTest2() {
  // initialize all ten integers of a device_vector to 1
  thrust::device_vector<int> D(10, 1);

  // set the first seven elements of a vector to 9
  thrust::fill(D.begin(), D.begin() + 7, 9);

  // initialize a host_vector with the first five elements of D
  thrust::host_vector<int> H(D.begin(), D.begin() + 5);

  // set the elements of H to 0, 1, 2, 3, ...
  thrust::sequence(H.begin(), H.end());

  // copy all of H back to the beginning of D
  thrust::copy(H.begin(), H.end(), D.begin());

  for (int i = 0; i < D.size(); i++)
    std::cout << "D[" << i << "] = " << D[i] << std::endl;

  // H and D are automatically deleted when the function returns.
}

int main() {
  VectorTest1();

  std::cout << "Finish Test1." << std::endl << std::endl;

  VectorTest2();

  std::cout << "Finish Test2." << std::endl << std::endl;

  system("pause");
  return 0;
}
