/*!
* \brief Operator.
*/

#include "operator/operator.h"

namespace cux {

///////////////
// Operator
void Operator::QueryPotentialOccupancy(const void *kernel_address, int kernel_id, 
                                       int threads_per_block, int shared_memory_size) {
  if (kernel_address == nullptr 
    || kernel_id < 0 
    || kernel_id > gpu_kernel_occupancys_.size()) {
    return;
  }
  assistor_->launch_config()->QueryPotentialOccupancy(
    kernel_address, threads_per_block, shared_memory_size,
    gpu_kernel_active_blocks_[kernel_id], gpu_kernel_occupancys_[kernel_id]);
}

void Operator::PrintRecordedInfo(const OpRunMode &mode, int kernel_id, const KernelInterface *kernel_info) {
  // TODO: Show config and occupancy.
  if (mode == OpRunMode::ON_HOST) {
    CUXLOG_COUT("V%d ) I: %f, R: %f -> [ %s ]",
      kernel_id,
      cpu_timer_record_[kernel_id].input,
      cpu_timer_record_[kernel_id].run,
      kernel_info->describe_info.c_str());
  }
  else {
    CUXLOG_COUT("V%d ) I: %f, W: %f, R: %f, O: %f -> %f [ %s ]",
      kernel_id,
      gpu_timer_record_[kernel_id].input,
      gpu_timer_record_[kernel_id].warnup,
      gpu_timer_record_[kernel_id].run,
      gpu_timer_record_[kernel_id].output,
      gpu_kernel_occupancys_[kernel_id], 
      kernel_info->describe_info.c_str());
  }
}

void Operator::ResetKernelNum(int cpu_kernel_num, int gpu_kernel_num) {  
  cpu_timer_record_.resize(cpu_kernel_num);

  gpu_kernel_occupancys_.resize(gpu_kernel_num);
  gpu_kernel_active_blocks_.resize(gpu_kernel_num);
  gpu_timer_record_.resize(gpu_kernel_num);
}

} // cux.
