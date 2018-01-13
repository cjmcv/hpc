/*!
* \brief The basic use of parallel_scan.
*/

#include <iostream>  
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
  int x[10] = { 0,1,2,3,4,5,6,7,8,9 };
  int y[10];
  Body<int> body(y, x);
  tbb::parallel_scan(tbb::blocked_range<int>(0, 10), body, tbb::auto_partitioner());
  std::cout << "sum:" << body.get_sum() << std::endl;
  return 0;
}