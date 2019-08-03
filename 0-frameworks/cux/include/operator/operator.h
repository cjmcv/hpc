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
#include "array.h"

namespace cux {

struct KernelTimeRecord {
  float run = 0.0;

  float warnup = 0.0;
  float input = 0.0;
  float output = 0.0;
};

struct OpKernel {
  TypeFlag type_flag;
  std::string describe_info;
  KernelTimeRecord time_record;
};

class Operator {
public:
  Operator(OpAssistor *assistor) : assistor_(assistor) {}

  void QueryPotentialOccupancy(const void *kernel_address, int kernel_id, 
                               int threads_per_block, int shared_memory_size);

  void PrintRecordedInfo(const OpRunMode &mode, int kernel_id, const OpKernel *kernel_info);
  
  // Show relevant prompts.
  virtual void Help() const = 0;
  // Set the input and output data.
  virtual int SetIoData(const std::vector< Array4D* > &input,
                        const std::vector< Array4D* > &output) = 0;
  virtual void RunOnHost() = 0;
  virtual void RunOnDevice() = 0;

public:   
  OpAssistor *assistor_;
  
  GpuTimer gpu_timer_;
  CpuTimer cpu_timer_;

  // occupancys for each kernel.
  // An element corresponds to a kernel.
  std::vector<double> gpu_kernel_occupancys_;
  std::vector<int> gpu_kernel_active_blocks_;
};
} // cux.

#endif //CUX_OPERATOR_H_
