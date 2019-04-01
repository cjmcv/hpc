/*!
* \brief Calculate PI using parallel and for reduction.
*/

#include <omp.h>

#include <stdio.h>
#include <stdlib.h>
#include <ctime>

#define random(x) ((rand()%(x*2))*1.0 - x) / x;

int main() {

  long num_trials = 10000000;
  double r = 1.0;   // radius of circle. Side of squrare is 2*r   
  double x, y, test, circ_count;
 
  printf("num_trials = %d\n", num_trials);

  printf("Serial version: \n");
  {
    circ_count = 0;
    srand(time(0));

    clock_t wtime = clock();
    for (long i = 0; i < num_trials; i++) {
      // Output[-1, 1]
      x = random(1000);
      y = random(1000);
      test = x*x + y*y;
      if (test <= r*r) circ_count++;
    }
    double pi = 4.0 * ((double)circ_count / (double)num_trials);

    printf("Result: pi is %lf in %lf seconds.\n", 
      pi, (clock() - wtime) * 1.0 / CLOCKS_PER_SEC);
  }
    
  printf("\nOpenmp: \n");
  {
    circ_count = 0;  
    srand(time(0));

    double wtime = omp_get_wtime();
    #pragma omp parallel
    {
      #pragma omp single
      printf("<Using %d threads>\n", omp_get_num_threads());

      #pragma omp for reduction(+:circ_count) private(x,y,test)
      for (long i = 0; i < num_trials; i++) {
        // Output[-1, 1]
        x = random(1000);
        y = random(1000);
        test = x*x + y*y;
        if (test <= r*r) circ_count++;
      }
    }
    double pi = 4.0 * ((double)circ_count / (double)num_trials);

    printf("Result: pi is %lf in %lf seconds.\n", 
      pi, omp_get_wtime() - wtime);
  }

  return 0;
}
