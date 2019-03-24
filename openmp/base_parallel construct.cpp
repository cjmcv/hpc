/*!
* \brief Parallel construct.
*/

#include <stdio.h>
#include <omp.h>

int main() {
  int nthreads = 4;
  omp_set_num_threads(nthreads);

  printf("Test parallel.\n");

#pragma omp parallel 
  {
    printf("Hello World from thread = %d with %d threads\n", 
      omp_get_thread_num(), omp_get_num_threads());
  }

  printf("\nTest parallel if false.\n");

#pragma omp parallel if(false) num_threads(6)
  {
    printf("<%d>", omp_get_thread_num());
  }

  printf("\nTest parallel if true.\n");

#pragma omp parallel if(true) num_threads(6)
  {
    printf("<%d>", omp_get_thread_num());
  }
}