/*!
* \brief The basic use of parallel_scan.
* \example: input: 1,2,3,4
*           operation: Add
*           ouput: 1,3,6,10 (out[i]=sum(in[0:i]))
*/

#include <iostream>  
#include <time.h>
#include <tbb/tbb.h>

template<typename DType>
class Body {
public:
  Body(DType y[], const DType x[]) :sum_(0), x_(x), y_(y) {}
  DType get_sum() const {
    return sum_;
  }

  template<typename Tag>
  void operator()(const tbb::blocked_range<int>& r, Tag) {
    DType temp = sum_;
    for (int i = r.begin(); i < r.end(); i++) {
      temp += x_[i];
      if (Tag::is_final_scan())	// Used to indicate that the initial scan is being performed
        y_[i] = temp;
    }
    sum_ = temp;
  }

  // Split b so that this and b can accumulate separately.
  Body(Body&b, tbb::split) :x_(b.x_), y_(b.y_), sum_(0) {}
  // Merge preprocessing state of a into this, where a was created earlier
  // from b by b¡¯s splitting constructor.
  // The operation reverse_join is similar to the operation join 
  // used by parallel_reduce, except that the arguments are reversed
  void reverse_join(Body& a) {
    sum_ += a.sum_;
  }
  // Assign state of b to this.
  void assign(Body& b) {
    sum_ = b.sum_;
  }
private:
  DType sum_;
  DType* const y_;
  const DType* const x_;
};

int main() {
  const int num = 100;
  int *input = new int[num];
  int *output = new int[num];
  for (int i = 0; i < num; i++) {
    input[i] = i;
    output[i] = 0;
  }
  
  time_t stime;
  // Serial
  stime = clock();
  output[0] = input[0];
  for (int i = 1; i < num; i++) {
    output[i] = input[i] + output[i-1];
  }
  std::cout << "Serial ->  time: " << clock() - stime << ", result: " << output[num - 1] << std::endl;

  memset(output, 0, sizeof(int) * num);

  // TBB
  stime = clock();
  Body<int> body(output, input);
  tbb::parallel_scan(tbb::blocked_range<int>(0, num), body, tbb::auto_partitioner());
  std::cout << "TBB Parallel ->  time: " << clock() - stime << ", result: " << body.get_sum() << std::endl;
  for (int i = 0; i < num; i++) {
    std::cout << output[i] << ", ";
  }
  return 0;
} 