/*!
* \brief The basic use of parallel_for.
*/

#include <iostream>  
#include <vector>  
#include <tbb/tbb.h>  
#include <time.h>

// Target function.
void ArrayPrint(const int *arr, const int index) {
  std::cout << arr[index] << ", ";
}

// Note: The operator in parallel_for is const.
// Use a Class to encapsulate the target function.
class ArrayPrintBody {
public:
  ArrayPrintBody(int *arr) : arr_(arr) {}
  void operator () (const tbb::blocked_range<size_t> & r) const {
    for (size_t i = r.begin(); i != r.end(); ++i) {
      ArrayPrint(arr_, i);
    }
  }

private:
  int *arr_;
};

int main() {
  // Initialize the input array.
  int num = 100;
  int *arr = new int[num];
  for (int i = 0; i < num; i++)
    arr[i] = i;

  time_t stime;

  // Serial mode.
  std::cout << "Serial mode: " << std::endl;
  stime = clock();
  for (int i = 0; i < num; i++) {
    ArrayPrint(arr, i);
  }
  std::cout << "Serial ->  time: " << clock() - stime << std::endl;

  // Note: 1, If you know the size of task, specify the grain size by youself 
  //       is better. Otherwise using tbb::auto_partitioner() is an option.
  //       2, If the task is small, there is no need to use TBB.

  // TBB example one.
  std::cout << std::endl <<"TBB example one: " << std::endl;
  stime = clock();
  tbb::parallel_for(tbb::blocked_range<size_t>(0, num), ArrayPrintBody(arr), tbb::auto_partitioner());
  //tbb::parallel_for(tbb::blocked_range<size_t>(0, num, num / 16), ArrayPrintBody(arr));
  std::cout << "TBB Parallel a ->  time: " << clock() - stime << std::endl;

  // TBB example two.
  std::cout << std::endl << "TBB example two: " << std::endl;
  stime = clock();
  tbb::parallel_for(tbb::blocked_range<size_t>(0, num), [&](tbb::blocked_range<size_t> &r) {
    for (size_t i = r.begin(); i < r.end();i++) {
      ArrayPrint(arr, i);
    }
  });
  std::cout << "TBB Parallel a ->  time: " << clock() - stime << std::endl;

  std::cout << std::endl;
  return 0;
}