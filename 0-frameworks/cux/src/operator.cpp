/*!
* \brief Operator.
*/

#include "operator.h"

namespace cux {

///////////////////
// ResultChecker
template <typename DType>
bool ResultChecker::CheckArray(DType *in, int len, int id) {
  if (id == 0) {
    SetBenchmarkData(in, len);
    return true;
  }
  float diff = 0.0;
  for (int i = 0; i < len; i++) {
    float t = prev_data_[i] - (float)in[i];
    diff += (t >= 0 ? t : -t);
  }
  if (diff < DBL_MIN) {
    CUXLOG_COUT("Pass: V0 vs V%d -> (diff: %f, first number: %f, %f)",
      id, diff, (float)prev_data_[0], (float)in[0]);
    return true;
  }
  else {
    CUXLOG_COUT("Fail: V0 vs V%d -> (diff: %f, first number: %f, %f)",
      id, diff, (float)prev_data_[0], (float)in[0]);
    return false;
  }
}

template <typename DType>
void ResultChecker::SetBenchmarkData(DType *in, int len) {
  if (prev_data_ == nullptr) {
    prev_data_ = new float[len];
    len_ = len;
  }
  else if (len_ != len) {
    delete[]prev_data_;
    prev_data_ = new float[len];
    len_ = len;
  }
  for (int i = 0; i < len; i++) {
    prev_data_[i] = in[i];
  }
}

template bool ResultChecker::CheckArray<float>(float *in, int len, int id);
template bool ResultChecker::CheckArray<half>(half *in, int len, int id);
template bool ResultChecker::CheckArray<int>(int *in, int len, int id);
template bool ResultChecker::CheckArray<signed char>(signed char *in, int len, int id);

template void ResultChecker::SetBenchmarkData<float>(float *in, int len);
template void ResultChecker::SetBenchmarkData<half>(half *in, int len);
template void ResultChecker::SetBenchmarkData<int>(int *in, int len);
template void ResultChecker::SetBenchmarkData<signed char>(signed char *in, int len);

///////////////
// Operator
void Operator::QueryPotentialOccupancy(const void *kernel_address, int kernel_id, 
                                       int threads_per_block, int shared_memory_size) {
  if (kernel_address == nullptr 
    || kernel_id < 0 
    || kernel_id > gpu_kernel_occupancys_.size()) {
    return;
  }
  op_params_.launch_config->QueryPotentialOccupancy(
    kernel_address, threads_per_block, shared_memory_size,
    gpu_kernel_active_blocks_[kernel_id], gpu_kernel_occupancys_[kernel_id]);
}

void Operator::PrintRecordedInfo(const OpRunMode &mode, int kernel_id, const OpKernel *kernel_info) {
  // TODO: Show config and occupancy.
  if (mode == OpRunMode::ON_HOST) {
    CUXLOG_COUT("V%d ) I: %f, R: %f -> [ %s ]",
      kernel_id,
      kernel_info->time_record.input, 
      kernel_info->time_record.run,
      kernel_info->describe_info.c_str());
  }
  else {
    CUXLOG_COUT("V%d ) I: %f, W: %f, R: %f, O: %f -> %f [ %s ]",
      kernel_id,
      kernel_info->time_record.input, 
      kernel_info->time_record.warnup,
      kernel_info->time_record.run,
      kernel_info->time_record.output,
      gpu_kernel_occupancys_[kernel_id], 
      kernel_info->describe_info.c_str());
  }
}

} // cux.
