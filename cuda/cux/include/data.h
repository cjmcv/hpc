/*!
* \brief CuxData.
*/

#ifndef CUX_CUXDATA_HPP_
#define CUX_CUXDATA_HPP_

#include "util.h"

namespace cux {

// TODO: 1. 用到Executor和Op上，作为数据操作的基本单元
//       2. 支持数据跨设备推送，cpu与gpu数据互通
template <typename Dtype>
class CuxData {
  enum MemHead {
    eUninitialized,
    eHeadAtCPU, 
    eHeadAtGPU, 
    eSynced 
  };

public:
  explicit CuxData(const int num, const int channels, const int height, const int width) {
    shape_.clear();
    shape_.push_back(num);
    shape_.push_back(channels);
    shape_.push_back(height);
    shape_.push_back(width);

    num_element_ = num * channels * height * width;

    cpu_data_ = nullptr;
    gpu_data_ = nullptr;
    mem_head_ = tind::eUninitialized;
  }

  ~CuxData() {
    if (cpu_data_ != nullptr) {
      delete[]cpu_data_;
      cpu_data_ = nullptr;
    }
    if (gpu_data_ != nullptr) {
      cudaFree(gpu_data_);
      gpu_data_ = nullptr;
    }
  }

  inline std::vector<int> &get_size() { return size_; }
  inline std::vector<int> &get_shape() { return shape_; }

  inline Dtype* cpu_data() { return cpu_data_; }
  inline Dtype* gpu_data() { return gpu_data_; }

  Dtype* get_cpu_data() {
    if (mem_head_ == tind::eUninitialized) {
      cpu_data_ = new Dtype[num_element_];
      mem_head_ = tind::eHeadAtCPU;

      return cpu_data_;
    }
    else if (mem_head_ == tind::eHeadAtCPU) {
      return cpu_data_;
    }
    else if (mem_head_ == tind::eHeadAtGPU) {

    }
  }

  Dtype* get_gpu_data() {
  
  }

private:
  Dtype *cpu_data_;
  Dtype *gpu_data_;

  // Take a tag to tell where the current memory has been saved (CPU or GPU).
  int mem_head_;
  // eNum, eChannels, eHeight, eWidth
  std::vector<int> shape_;
  int num_element_;
};

} // cux.
#endif //CUX_CUXDATA_HPP_