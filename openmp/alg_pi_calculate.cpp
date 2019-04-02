/*!
* \brief Calculate PI using parallel, for and reduction.
*/

#include <omp.h>

#include <stdio.h>
#include <stdlib.h>
#include <ctime>

#include <random>

// C: Output[-1, 1]
#define random() ((rand()%(200))*1.0 - 100) / 100;

int main() {

  long num_trials = 10000000;
  double r = 1.0;   // radius of circle. Side of squrare is 2*r   
  double x, y, test, circ_count;
 
  /// Generate random numbers using C++, but it is slower than using C.
  //std::uniform_real_distribution<double> uniform_rand(-1.0, 1.0);
  //std::mt19937 rng;
  //rng.seed(std::random_device{}());
  //double generated_number = uniform_rand(rng);

  printf("num_trials = %d\n", num_trials);

  printf("Serial version: \n");
  {
    circ_count = 0;
    srand(time(0));

    clock_t wtime = clock();
    for (long i = 0; i < num_trials; i++) {
      x = random(); // Or use uniform_rand(rng).
      y = random();
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
        x = random();
        y = random();
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
