/*!
* \brief Operator.
*/

#ifndef CUX_OPERATOR_HPP_
#define CUX_OPERATOR_HPP_

#include <vector>
#include "util.h"
#include "data.h"

namespace cux {

// TODO: 1. LayoutÍÆ¼ö

struct OpParam {};

class Operator {
public:
  Operator(): loops_(1) {}
  inline void SetLoops(const int loop) { loops_ = loop; }
  inline void PrintElapsedTime(const RunMode mode) const {
    if (mode == RunMode::ON_HOST) {
      CUXLOG_COUT("CPU: %f ms.", cpu_time_record_);
    }
    else if (mode == RunMode::ON_DEVICE) { 
      CUXLOG_COUT("GPU: %f ms for input.", gpu_time_in_record_);
      CUXLOG_COUT("     %f ms for output.", gpu_time_out_record_);
      CUXLOG_COUT("     %f ms for warnup v0.", gpu_time_warnup_record_);
      for (int ki = 0; ki < gpu_time_kernel_record_.size(); ki++) {
        CUXLOG_COUT("     %f ms for kernel v%d.", gpu_time_kernel_record_[ki], ki);
      }
    }
  }
  
  virtual void Help() const {};

  virtual int SetIoParams(const std::vector< CuxData<float>* > &input,
                          const std::vector< CuxData<float>* > &output,
                          const OpParam *params) { return -1; };
  virtual void RunOnHost() {};
  virtual void RunOnDevice() {};

public:
  GpuTimer gpu_timer_;
  std::vector<float> gpu_time_kernel_record_;
  float gpu_time_in_record_;
  float gpu_time_out_record_;
  float gpu_time_warnup_record_;

  CpuTimer cpu_timer_;
  float cpu_time_record_;
  
  int loops_;
};
} // cux.

#endif //CUX_OPERATOR_HPP_