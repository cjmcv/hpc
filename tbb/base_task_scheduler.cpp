/*!
* \brief task scheduler
*/

#include "iostream"
#include "time.h"

#include <tbb/tbb.h>  

long SerialFib(long n) {
  if (n < 2)
    return n;
  else
    return SerialFib(n - 1) + SerialFib(n - 2);
}

class FibTask : public tbb::task {
public:
  FibTask(long n, long* sum) : n_(n), sum_(sum) {}
  // Overrides virtual function task::execute
  tbb::task* execute() {
    if (n_ < 10) {
      *sum_ = SerialFib(n_);
    }
    else {
      long x, y;
      // Create child task.
      FibTask& a = *new(tbb::task::allocate_child()) FibTask(n_ - 1, &x);
      FibTask& b = *new(tbb::task::allocate_child()) FibTask(n_ - 2, &y);
      // Set reference count (2+1) ->(task a + task b + the waiting task which is created by sapwn_and_wait_for_all)
      tbb::task::set_ref_count(3);
      // Start b running.
      tbb::task::spawn(b);
      // Start a running and wait for all children (a and b).
      tbb::task::spawn_and_wait_for_all(a);
      // Do the sum.
      *sum_ = x + y;
    }
    return NULL;
  }

private:
  const long n_;
  long* const sum_;
};

long ParallelFib(long n) {
  long sum;
  // Create the root task.
  FibTask& a = *new(tbb::task::allocate_root()) FibTask(n, &sum);
  // Start a running and wait for all children.
  tbb::task::spawn_root_and_wait(a);
  return sum;
}

int main(int argc, char** argv) {
  int loops = 100;
  long reslut;

  time_t stime = clock();
  for(int i=0; i<loops; i++)
    reslut = SerialFib(30);
   printf("Serial result = %d, time: %f \n", reslut, double(clock() - stime));

  // It is much slower than the serial one because the complexity of the child tasks are very low.
  stime = clock();
  for (int i = 0; i<loops; i++)
    reslut = ParallelFib(30);
  printf("TBB result = %d, time: %f \n", reslut, double(clock() - stime));

  return 0;
}