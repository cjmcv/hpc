/*!
* \brief Operator.
*/

#ifndef CUX_OPERATOR_H_
#define CUX_OPERATOR_H_

#include <vector>
#include "util/util.h"
#include "util/timer.h"
#include "util/launch_config.h"
#include "array.h"

namespace cux {

// Used to check the correctness of the output of those functions.
class ResultChecker {
public:
  ResultChecker() :prev_data_(nullptr), len_(0) {}
  ~ResultChecker() {
    if (prev_data_ != nullptr) {
      delete[]prev_data_;
      len_ = 0;
    }
  }
  template <typename DType>
  bool CheckArray(DType *in, int len, int id);

private:
  // Set the benchmark data, which is correct by default.
  template <typename DType>
  void SetBenchmarkData(DType *in, int len);

private:
  float *prev_data_;
  int len_;
};

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

struct OpParams {
  LaunchConfig *launch_config;
};

class Operator {
public:
  Operator(Device *device)
    : device_(device), cublas_handle_(nullptr) {
    if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
      CUXLOG_ERR("Cannot create Cublas handle. Cublas won't be available.");
    }
  }
  //TODO: launch_config should be defined here. 
  inline void SetOpParams(const OpParams &params) {
    op_params_.launch_config = params.launch_config;
  }

  void QueryPotentialOccupancy(const void *kernel_address, int kernel_id, int threads_per_block, int shared_memory_size);

  void PrintRecordedInfo(const OpRunMode &mode, int kernel_id, const OpKernel *kernel_info);
  
  // Show relevant prompts.
  virtual void Help() const = 0;
  // Set the input and output data.
  virtual int SetIoData(const std::vector< Array4D* > &input,
                        const std::vector< Array4D* > &output) = 0;
  virtual void RunOnHost() = 0;
  virtual void RunOnDevice() = 0;

public: 
  Device *device_;
  OpParams op_params_;
  
  GpuTimer gpu_timer_;
  CpuTimer cpu_timer_;

  // Verify the correctness of the output.
  ResultChecker checker_;

  // occupancys for each kernel.
  // An element corresponds to a kernel.
  LaunchConfig *launch_config_;
  std::vector<double> gpu_kernel_occupancys_;
  std::vector<int> gpu_kernel_active_blocks_;

  cublasHandle_t cublas_handle_;
};
} // cux.

#endif //CUX_OPERATOR_H_
