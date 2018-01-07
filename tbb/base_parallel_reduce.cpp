/*!
* \brief The basic use of parallel_reduce.
* \features 1, The operator in it is not const 
*              which is different from parallel_for.
*           2, Contains a constructor with tbb::split.
*           3, Have a join() function to merge result. 
*/
#include <iostream>
#include <tbb/tbb.h>
#include <time.h>

// Serial mode.
float GetSumSerial(float *arr, size_t n) {
  float temp = 0;
  for (int i = 0; i < n; i++) {
    temp += arr[i];
  }
  return temp;
}

// TBB mode.
struct GetSumBody {
  float value;
  GetSumBody() : value(0) {}
  GetSumBody(GetSumBody& s, tbb::split) { value = 0; }
  void operator()(const tbb::blocked_range<float*>& range) {
    float temp = value;
    for (float *pval = range.begin(); pval != range.end(); ++pval) {
      temp += *pval;
    }
    value = temp;
  }
  // Do the corresponding merges.
  void join(GetSumBody& obj) { value += obj.value; }
};

float GetSumParallel(float *arr, size_t n) {
  GetSumBody sum_obj;
  // Call tbb without specifying the Grain Size.
  tbb::parallel_reduce(tbb::blocked_range<float*>(arr, arr + n), sum_obj, tbb::auto_partitioner());
  //tbb::parallel_reduce(tbb::blocked_range<float*>(arr, arr + n, n / 16), sum_obj);
  return sum_obj.value;
}

int main() {
  // Initialize the input array.
  int num = 10000000;
  float *arr = new float[num];
  for (int i = 0; i < num; i++)
    arr[i] = 2;

  time_t stime;
  float res = 0;

  stime = clock();
  res = GetSumSerial(arr, num);
  std::cout << "Serial ->  result: " << res << ", time: " << clock() - stime << std::endl;

  stime = clock();
  res = GetSumParallel(arr, num);
  std::cout << "TBB Parallel ->  result: " << res << ", time: " << clock() - stime << std::endl;

  return 0;
}