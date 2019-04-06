/*!
* \brief Synchronous operation in openmp, including
*    barrier, ordered and master.
*/

#include <omp.h>
#include <stdio.h>

int main() {
  ////////////////////////////////////////
  // barrier
  ////////////////////////////////////////
  printf("\nbarrier: \n");
  #pragma omp parallel 
  {
    int sumb = 0;
    for (int i = 0; i < 100000; i++)
      sumb++;

    // Wait for all threads to execute here.
    #pragma omp barrier
    printf("sum=%d, thread_id=%d\n", sumb, omp_get_thread_num());
  }

  ////////////////////////////////////////
  // ordered
  ////////////////////////////////////////
  // Execute the threads sequentially.
  // "ordered" must be bound to "for".
  printf("\nordered: \n");
  #pragma omp parallel for ordered num_threads(6)
  for (int i = 0; i < 10; i++) {
    #pragma omp ordered
    printf("i=%d, thread_id=%d\n", i, omp_get_thread_num());
  }

  ////////////////////////////////////////
  // master
  ////////////////////////////////////////
  printf("\nmaster: \n");
  #pragma omp parallel
  {
    // Specify the master thread to execute.
    #pragma omp master
    for (int i = 0; i<5; i++) {
      printf("i=%d, thread_id=%d\n", i, omp_get_thread_num());
    }
  }
}