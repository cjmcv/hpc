/*!
* \brief Parallel and For.
*/

#include <stdio.h>
#include <omp.h>

int main() {
  ////////////////////////////////////////
  // parallel
  ////////////////////////////////////////
  printf("Test parallel.\n");
  omp_set_num_threads(10);
  #pragma omp parallel 
  {
    printf("%d(%d), ", omp_get_thread_num(), omp_get_num_threads());
  }
  // 0(10), 8(10), 5(10), 7(10), 3(10), 9(10), 4(10), 1(10), 2(10), 6(10),

  printf("\nTest parallel if false.\n");
  #pragma omp parallel if(false) num_threads(6)
  {
    printf("<%d>", omp_get_thread_num());
  }
  // <0>

  printf("\nTest parallel if true.\n");
  #pragma omp parallel if(true) num_threads(6)
  {
    printf("<%d>", omp_get_thread_num());
  }
  // <1><2><5><0><3><4>

  ////////////////////////////////////////
  // for
  ////////////////////////////////////////
  printf("\nTest for in parallel.\n");
  #pragma omp parallel num_threads(6)
  {
    // Each thread executes once.
    for (int i = 0; i < 10; ++i) {
      printf("%d(%d), ", i, omp_get_num_threads());
    }
  }
  // 0(6), 0(6), 0(6), 1(6), 1(6), 0(6), 2(6), 0(6), 1(6), 3(6), 
  // 2(6), 4(6), 2(6), 3(6), 1(6), 4(6), 5(6), 6(6), 1(6), 7(6),
  // 3(6), 2(6), 8(6), 3(6), 9(6), 4(6), 0(6), 5(6), 1(6), 6(6),
  // 2(6), 7(6), 4(6), 8(6), 5(6), 2(6), 6(6), 5(6), 3(6), 7(6), 
  // 3(6), 8(6), 6(6), 9(6), 4(6), 7(6), 5(6), 8(6), 4(6), 9(6), 
  // 5(6), 6(6), 6(6), 9(6), 7(6), 7(6), 8(6), 8(6), 9(6), 9(6), 

  printf("\nTest for.\n");
  #pragma omp for
  for (int i = 0; i < 10; ++i) {
    printf("%d(%d), ", i, omp_get_num_threads());
  }
  // 0(1), 1(1), 2(1), 3(1), 4(1), 5(1), 6(1), 7(1), 8(1), 9(1),

  printf("\nTest parallel and for.\n");
  #pragma omp parallel num_threads(3)
  {
    #pragma omp for
    for (int i = 0; i < 10; ++i) {
      printf("%d(%d), ", i, omp_get_num_threads());
    }
  }
  // 0(3), 4(3), 1(3), 7(3), 2(3), 5(3), 3(3), 6(3), 8(3), 9(3)

  printf("\nTest parallel for.\n");
  #pragma omp parallel for num_threads(5)
  for (int i = 0; i < 10; ++i) {
    printf("%d(%d), ", i, omp_get_num_threads());
  }
  // 4(5), 6(5), 2(5), 8(5), 0(5), 9(5), 5(5), 1(5), 3(5), 7(5),


}