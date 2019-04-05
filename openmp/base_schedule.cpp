/*!
* \brief Records the basic usage of schedule.
*
* \operation schedule: Mainly for the parallel for loop, when the computation 
* amount of each iteration in the loop is not equal, if you simply assign the 
* same number of iterations to each thread, it may cause the imbalance of 
* computing load of each thread and affect the overall performance of the program.
*
* \params static  -> statically preassign iterations to threads
*         dynamic -> each thread gets more work when its done at runtime
*         guided  -> similar to dynamic with automatically adjusted chunk size
*         auto    -> let the compiler decide!
*/

#include <omp.h>
#include <stdio.h>

int main() {

  #pragma omp parallel for num_threads(6)
  for (int i = 0; i < 10; i++) {
    printf("normal: iter %d on thread %d\n",
      i, omp_get_thread_num());
  }

  int chunk = 3;

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < 10; i++) {
    printf("schedule(static): iter %d on thread %d\n", 
      i, omp_get_thread_num());
  }

  #pragma omp parallel for schedule(static, chunk)
	for (int i = 0; i < 10; i++) {
		printf("schedule(static, %d): iter %d on thread %d\n", 
      chunk, i, omp_get_thread_num());
	}

  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < 10; i++) {
    printf("schedule(dynamic): iter %d on thread %d\n",
      i, omp_get_thread_num());
  }

  #pragma omp parallel for schedule(dynamic, chunk)
  for (int i = 0; i < 10; i++) {
    printf("schedule(dynamic, %d): iter %d on thread %d\n",
      chunk, i, omp_get_thread_num());
  }

	return 0;
}