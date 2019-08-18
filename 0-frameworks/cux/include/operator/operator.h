/*!
* \brief Operator.
*/

#ifndef CUX_OPERATOR_H_
#define CUX_OPERATOR_H_

#include <vector>
#include "util/util.h"
#include "util/timer.h"
#include "util/launch_config.h"
#include "operator/op_assistor.h"
#include "operator/kernel_interface.h"
#include "array.h"

namespace cux {

struct KernelTimeRecord {
  float run = 0.0;

  float warnup = 0.0;
  float input = 0.0;
  float output = 0.0;
};

class Operator {
public:
  Operator(OpAssistor *assistor) : assistor_(assistor) {}

  void QueryPotentialOccupancy(const void *config_kernel, int kernel_id, 
                               int threads_per_block, int shared_memory_size);

  void PrintRecordedInfo(const OpRunMode &mode, int kernel_id, const KernelInterface *kernel_info);
  void ResetByKernelNum(int cpu_kernel_num, int gpu_kernel_num);

  // Show relevant prompts.
  virtual void Help() const = 0;
  virtual void AddPlugin(KernelInterface *kernel_if, OpRunMode mode) = 0;
  virtual void ExtractDataTypes(std::vector<int>& type_flags) = 0;

  virtual void RunOnHost(const std::vector< Array4D* > &input,
                         const std::vector< Array4D* > &output) = 0;
  virtual void RunOnDevice(const std::vector< Array4D* > &input,
                           const std::vector< Array4D* > &output) = 0;

private:
  // Set the input and output data.
  virtual void IoCheckAndSet(const std::vector< Array4D* > &input,
                             const std::vector< Array4D* > &output) = 0;

public:   
  OpAssistor *assistor_;
  
  GpuTimer gpu_timer_;
  CpuTimer cpu_timer_;
  std::vector<KernelTimeRecord> cpu_timer_record_;
  std::vector<KernelTimeRecord> gpu_timer_record_;

  // occupancys for each kernel.
  // An element corresponds to a kernel.
  std::vector<double> gpu_kernel_occupancys_;
  std::vector<int> gpu_kernel_active_blocks_;
};
} // cux.

#endif //CUX_OPERATOR_H_
