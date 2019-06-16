/*!
* \brief Operator.
*/

#ifndef CUX_OPERATOR_HPP_
#define CUX_OPERATOR_HPP_

#include <vector>
#include "util.h"
#include "data.h"

namespace cux {

// TODO: 1. Layout推荐
//       2. Kernel手动选择/自动遍历所有kernel?
class Operator {
public:
  Operator(): loops_(1) {}
  inline void SetLoops(int loop) { loops_ = loop; }
  inline void PrintCpuRunTime() const {
    CUXLOG_COUT("CPU: %f ms for %d loops.", cpu_time_record_, loops_);
  }
  inline void PrintGpuRunTime() const {
    CUXLOG_COUT("GPU: %f ms for %d loops.", gpu_time_record_, loops_);
  }  
  
  virtual void Help() const {};
  virtual void PrintResult() const {};

  virtual int SetIoParams(const std::vector< CuxData<float>* > &input,
                          const std::vector< CuxData<float>* > &output,
                          const void *params) { return -1; };
  virtual void RunOnHost() {};
  virtual void RunOnDevice() {};

public:
  GpuTimer gpu_timer_;
  float gpu_time_record_;

  CpuTimer cpu_timer_;
  float cpu_time_record_;
  
  int loops_;
};
} // cux.

#endif //CUX_OPERATOR_HPP_