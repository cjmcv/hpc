/*!
* \brief The basic use of parallel_for.
*/

#include <iostream>  
#include <vector>  
#include <tbb/tbb.h>  

// Target function.
void ArrayPrint(const int *arr, const int index) {
  std::cout << arr[index] << ", ";
}

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

  // Normal call.
  for (int i = 0; i < num; i++) {
    ArrayPrint(arr, i);
  }

  // Note: blocked_range is used to control partitioner and grainsize in a parallel task.
  //       So the range in blocked_range should be suitable for cpu cores to get the best performance.
  //       And not just set it to (0, num) for all cases.

  // TBB example one.
  std::cout << "Example one: " << std::endl;
  tbb::parallel_for(tbb::blocked_range<size_t>(0, num), ArrayPrintBody(arr));

  // TBB example two.
  std::cout << std::endl << "Example two: " << std::endl;
  tbb::parallel_for(tbb::blocked_range<size_t>(0, num), [&](tbb::blocked_range<size_t> &r) {
    for (size_t i = r.begin(); i < r.end();i++) {
      ArrayPrint(arr, i);
    }
  });

  std::cout << std::endl;
  return 0;
}