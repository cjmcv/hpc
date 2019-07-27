/*!
* \brief Operator.
*/

#include "operator.h"

namespace cux {

///////////////////
// ResultChecker
template <typename Dtype>
bool ResultChecker<Dtype>::CheckArray(const Dtype *in, const int len, const int id) {
  if (id == 0) {
    SetBenchmarkData(in, len);
    return true;
  }
  float diff = 0.0;
  for (int i = 0; i < len; i++) {
    Dtype t = prev_data_[i] - in[i];
    diff += (t >= 0 ? t : -t);
  }
  if (diff < DBL_MIN) {
    CUXLOG_INFO("Pass: V0 vs V%d -> (diff: %f, first number: %f, %f)",
      id, diff, (float)prev_data_[0], (float)in[0]);
    return true;
  }
  else {
    CUXLOG_WARN("Fail: V0 vs V%d -> (diff: %f, first number: %f, %f)",
      id, diff, (float)prev_data_[0], (float)in[0]);
    return false;
  }
}

template <typename Dtype>
void ResultChecker<Dtype>::SetBenchmarkData(const Dtype *in, const int len) {
  if (prev_data_ == nullptr) {
    prev_data_ = new Dtype[len];
    len_ = len;
  }
  else if (len_ != len) {
    delete[]prev_data_;
    prev_data_ = new Dtype[len];
    len_ = len;
  }
  memcpy(prev_data_, in, sizeof(Dtype) * len);
}
INSTANTIATE_CLASS(ResultChecker);


///////////////
// Operator
template <typename Dtype>
void Operator<Dtype>::PrintElapsedTime(const OpRunMode mode) {
  if (mode == OpRunMode::ON_HOST) {
    for (int ki = 0; ki < cpu_time_kernel_record_.size(); ki++) {
      CUXLOG_COUT("CPU: %f ms for kernel V%d : %s.", cpu_time_kernel_record_[ki], ki, GetHostKernelsInfo(ki).c_str());
    }
  }
  else if (mode == OpRunMode::ON_DEVICE) {
    CUXLOG_COUT("GPU: %f ms for input.", gpu_time_in_record_);
    CUXLOG_COUT("GPU: %f ms for output.", gpu_time_out_record_);
    CUXLOG_COUT("GPU: %f ms for warnup V0.", gpu_time_warnup_record_);
    if (gpu_time_kernel_record_.size() != gpu_kernel_occupancys_.size()) {
      CUXLOG_ERR("GPU: gpu_time_kernel_record_.size() != gpu_kernel_occupancys_.size()");
      return;
    }
    for (int ki = 0; ki < gpu_time_kernel_record_.size(); ki++) {
      CUXLOG_COUT("GPU: %f ms for kernel V%d (Occuancys :%f, active_blocks: %d) : %s.", 
        gpu_time_kernel_record_[ki], ki, gpu_kernel_occupancys_[ki], gpu_kernel_active_blocks_[ki], GetDeviceKernelsInfo(ki).c_str());
    }
  }
}

INSTANTIATE_CLASS(Operator);
} // cux.
