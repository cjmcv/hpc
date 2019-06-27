/*!
* \brief Operator.
*/

#ifndef CUX_OPERATOR_HPP_
#define CUX_OPERATOR_HPP_

#include <vector>
#include "util/util.h"
#include "data.h"

namespace cux {

template <typename Dtype>
class ResultChecker {
public:
  ResultChecker() :prev_data_(nullptr), len_(0) {}
  ~ResultChecker() {
    if (prev_data_ != nullptr) {
      delete[]prev_data_;
      len_ = 0;
    }
  }
  bool CheckArray(const Dtype *in, const int len, const int id);

private:
  void SetBenchmarkData(const Dtype *in, const int len);
private:
  Dtype *prev_data_;
  int len_;
};

class PerformanceEvaluator {
public:
  static double GetPotentialOccupancy(const void *kernel,
    const int block_size, const size_t dynamic_shared_mem);
};

class Operator {
public:
  Operator(): loops_(1) {}
  inline void SetLoops(const int loop) { loops_ = loop; }
  void PrintElapsedTime(const OpRunMode mode) const;
  
  virtual void Help() const {};
  virtual int SetIoData(const std::vector< CuxData<float>* > &input,
                        const std::vector< CuxData<float>* > &output) { return -1; };
  virtual void RunOnHost() {};
  virtual void RunOnDevice() {};

public: 
  int loops_;
  
  GpuTimer gpu_timer_;
  std::vector<float> gpu_time_kernel_record_;
  float gpu_time_in_record_;
  float gpu_time_out_record_;
  float gpu_time_warnup_record_;

  CpuTimer cpu_timer_;
  std::vector<float> cpu_time_kernel_record_;
  
  ResultChecker<float> checker_;
  //PerformanceEvaluator evaluator_;
  std::vector<float> gpu_kernel_occupancys_;
};
} // cux.

#endif //CUX_OPERATOR_HPP_