/*!
* \brief Sections and Single.
*/

#include <stdio.h>
#include <omp.h>

int main() {
  ////////////////////////////////////////
  // Sections
  ////////////////////////////////////////
  printf("\nTest sections: \n");
  #pragma omp parallel sections num_threads(3)
  {  
    // Each section has a thread to call.
    printf("thread_num = %d, num_threads = %d.\n",
      omp_get_thread_num(), omp_get_num_threads());

    #pragma omp section
    {
      printf("section 0,tid=%ld\n", omp_get_thread_num());
    }
    #pragma omp section
    {
      printf("section 1,tid=%ld\n", omp_get_thread_num());
    }
    #pragma omp section
    {
      printf("section 2,tid=%ld\n", omp_get_thread_num());
    }
  }
  ////////////////////////////////////////
  // Single
  //////////////////////////////////////// 
  printf("\nTest single: \n");
  #pragma omp parallel num_threads(3)
  {
    // It will only take a thread to execute it.
    #pragma omp single
    printf("Beginning work1.\n");

    printf("Running work1 with thread %d\n", omp_get_thread_num());

    // It also play the role of a barrier.
    #pragma omp single
    {
      printf("Finishing work1.\n"); 
      for (int i = 0; i < 20; i++)
        printf("-");
    }

    // nowait.
    #pragma omp single nowait
    {
      printf("Running work2 with thread %d\n", omp_get_thread_num());    
      for (int i = 0; i < 20; i++)
        printf(".");
    }

    printf("work on 2 parallelly.%d\n", omp_get_thread_num());
  }
  return 0;
}