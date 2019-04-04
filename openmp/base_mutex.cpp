/*!
* \brief Mutex operation in openmp, including 
*    critical, atomic, lock.
*/

#include <omp.h>
#include <stdio.h>

int main() {
  ////////////////////////////////////////
  // critical
  ////////////////////////////////////////
  printf("critical: \n");
  int sum = 0;
  #pragma omp parallel for
  for (int i = 0; i < 100000; i++) {
    // critical corresponds to any size code block.
    #pragma omp critical
    {
      sum += 1;
    }
  }
  printf("sum = %d.\n", sum);

  ////////////////////////////////////////
  // atomic
  ////////////////////////////////////////
  printf("\natomic: \n");
  sum = 0;
  #pragma omp parallel for
  for (int i = 0; i < 100000; i++) {
    // atomic corresponds to a single assignment statement
    #pragma omp atomic
      sum += 1;
  }
  printf("sum = %d.\n", sum);

  ////////////////////////////////////////
  // lock -> init / destory , set / unset
  ////////////////////////////////////////
  printf("\nlock: \n");
  sum = 0;

  static omp_lock_t lock;
  omp_init_lock(&lock);
    
  #pragma omp parallel for
  for (int i = 0; i < 100000; i++) {
    omp_set_lock(&lock);
    sum += 1;
    omp_unset_lock(&lock);
  }

  omp_destroy_lock(&lock);

  printf("sum = %d.\n", sum);
}