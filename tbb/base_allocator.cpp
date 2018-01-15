/*!
* \brief The basic use of allocator.
*/
#include <iostream>
#include <vector>

#include <tbb/tbb.h>
#include <tbb/scalable_allocator.h>

int main() {
  tbb::tick_count start_time, end_time;

  const int num = 1000;
  const int len = 10000000;
  float **mem_container = (float **)malloc(num * sizeof(float *));

  // Serial.
  start_time = tbb::tick_count::now();
  for (int i = 0; i < num; i++) {
    mem_container[i] = (float *)malloc(len * sizeof(float));
  }
  for (int i = 0; i < num; i++) {
    free(mem_container[i]);
  }
  end_time = tbb::tick_count::now();
  std::cout << (end_time - start_time).seconds() << std::endl;

  // TBB. scalable_calloc, scalable_realloc.
  start_time = tbb::tick_count::now();
  for (int i = 0; i < num; i++) {
    mem_container[i] = (float *)scalable_malloc(len * sizeof(float)); // Faster.
  }
  for (int i = 0; i < num; i++) {
    scalable_free(mem_container[i]);
  }
  end_time = tbb::tick_count::now();
  std::cout << (end_time - start_time).seconds() << std::endl;

  return 0;
}