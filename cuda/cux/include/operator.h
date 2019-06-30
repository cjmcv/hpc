/*!
* \brief Operator.
*/

#ifndef CUX_OPERATOR_HPP_
#define CUX_OPERATOR_HPP_

#include <vector>
#include "util/util.h"
#include "data.h"

namespace cux {

// Used to check the correctness of the output of those functions.
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
  // Set the benchmark data, which is correct by default.
  void SetBenchmarkData(const Dtype *in, const int len);

private:
  Dtype *prev_data_;
  int len_;
};

class PerformanceEvaluator {
public:
  // occupancy = (double)active_warps / max_warps;
  static void GetPotentialOccupancy(const void *kernel, const int block_size,
                                    const size_t dynamic_shared_mem, 
                                    int &active_block, double &occupancy);
  // It suggests a block size that achieves the best theoretical occupancy.
  // But the occupancy can not be translated directly to performance.
  static void GetSuggestedLayout(const void *kernel, const int count,
                                 const int dynamic_smem_usage,
                                 int &grid_size, int &block_size);
};

class Operator {
public:
  Operator(const int cpu_kernel_cnt, const int gpu_kernel_cnt)
    : loops_(1), 
    cpu_kernel_cnt_(cpu_kernel_cnt),
    gpu_kernel_cnt_(gpu_kernel_cnt) {}
  inline void SetLoops(const int loop) { loops_ = loop; }
  void PrintElapsedTime(const OpRunMode mode) const;
  
  // Show relevant prompts.
  virtual void Help() const {};
  // Set the input and output data.
  virtual int SetIoData(const std::vector< CuxData<float>* > &input,
                        const std::vector< CuxData<float>* > &output) { return -1; };
  virtual void RunOnHost() {};
  virtual void RunOnDevice() {};

public: 
  // How many times the Kernel will be executed.
  int loops_;
  
  GpuTimer gpu_timer_;
  std::vector<float> gpu_time_kernel_record_;
  float gpu_time_in_record_;
  float gpu_time_out_record_;
  float gpu_time_warnup_record_;

  CpuTimer cpu_timer_;
  std::vector<float> cpu_time_kernel_record_;
  
  // Verify the correctness of the output.
  ResultChecker<float> checker_;

  // The total number of Kenrels.
  int cpu_kernel_cnt_;
  int gpu_kernel_cnt_;
  
  // occupancys for each kernel.
  // An element corresponds to a kernel.
  std::vector<double> gpu_kernel_occupancys_;
  std::vector<int> gpu_kernel_active_blocks_;
};
} // cux.

#endif //CUX_OPERATOR_HPP_