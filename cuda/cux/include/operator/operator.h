/*!
* \brief Operator.
*/

#ifndef CUX_OPERATOR_HPP_
#define CUX_OPERATOR_HPP_

#include <vector>
#include "util.h"
#include "data.h"

namespace cux {

// TODO: 1¡¢LayoutÍÆ¼ö
class Operator {
public:
  Operator(): loops_(1) {}
  void SetLoops(int loop) { loops_ = loop; }
  void PrintCpuRunTime() const {
    std::cout << cpu_time_record_ << " ms for " << loops_ << " loops." << std::endl;
  }
  void PrintGpuRunTime() const {
    std::cout << gpu_time_record_ << " ms for " << loops_ << " loops." << std::endl;
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