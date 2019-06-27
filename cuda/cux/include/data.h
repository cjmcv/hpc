/*!
* \brief CuxData.
*/

#ifndef CUX_CUXDATA_HPP_
#define CUX_CUXDATA_HPP_

#include "util/util.h"

namespace cux {

template <typename Dtype>
class CuxData {
public:
  enum MemoryHead {
    UNINITIALIZED,
    HEAD_AT_HOST,
    HEAD_AT_DEVICE
  };

public:
  explicit CuxData(CuxData &in) {
    std::vector<int> v = in.shape();
    shape_.assign(v.begin(), v.end());
    num_element_ = in.num_element();
    size_ = in.size();
    if (in.mem_head() == MemoryHead::UNINITIALIZED) {
      cpu_data_ = gpu_data_ = nullptr;
    }
    else if (in.mem_head() == MemoryHead::HEAD_AT_HOST) {
      memcpy(GetCpuData(), in.cpu_data(), size_);
      gpu_data_ = nullptr;
    }
    else {
      cpu_data_ = nullptr;
      cudaMemcpy(GetGpuData(), in.gpu_data(), size_);
    }
    mem_head_ = in.mem_head();
  }

  explicit CuxData(DataMode data_mode, const int num, const int channels, const int height, const int width) {
    data_mode_ = data_mode;

    shape_.clear();
    shape_.push_back(num);
    shape_.push_back(channels);
    shape_.push_back(height);
    shape_.push_back(width);

    num_element_ = num * channels * height * width;
    size_ = num_element_ * sizeof(Dtype);

    cpu_data_ = nullptr;
    gpu_data_ = nullptr;
    mem_head_ = MemoryHead::UNINITIALIZED;
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

  inline const std::vector<int> &shape() { return shape_; }
  inline int num_element() { return num_element_; }
  inline int size() { return size_; }
  inline int mem_head() { return mem_head_; }

  inline Dtype* cpu_data() { return cpu_data_; }
  inline Dtype* gpu_data() { return gpu_data_; }

  void SetCpuData(Dtype *data) {
    if (mem_head_ == MemoryHead::UNINITIALIZED) {
      cpu_data_ = new Dtype[num_element_];
      mem_head_ = MemoryHead::HEAD_AT_HOST;
    }
    memcpy(cpu_data_, data, size());
  }

  Dtype* GetCpuData() {
    if (mem_head_ == MemoryHead::UNINITIALIZED) {
      cpu_data_ = new Dtype[num_element_];
    }
    else if (mem_head_ == MemoryHead::HEAD_AT_DEVICE) {
      if (cpu_data_ == nullptr) {
        cpu_data_ = new Dtype[num_element_];
      }
      else if (data_mode_ == DataMode::INPUT) {
        // It is assumed that type A data will not be altered and will not need to be copied repeatedly
        return cpu_data_;
      }     
      // Type OUTPUT data is meaningless and does not need to be copied.
      if(data_mode_ != DataMode::OUTPUT)
        CUDA_CHECK(cudaMemcpy(cpu_data_, gpu_data_, size(), cudaMemcpyDeviceToHost));
    }
    mem_head_ = MemoryHead::HEAD_AT_HOST;
    return cpu_data_;
  }

  Dtype* GetGpuData() {
    if (mem_head_ == MemoryHead::UNINITIALIZED) {
      CUDA_CHECK(cudaMalloc(&gpu_data_, size()));
    }
    else if (mem_head_ == MemoryHead::HEAD_AT_HOST) {
      if (gpu_data_ == nullptr) {
        CUDA_CHECK(cudaMalloc(&gpu_data_, size()));
      }
      else if (data_mode_ == DataMode::INPUT) {
        return cpu_data_;
      }
      // Type OUTPUT data is meaningless and does not need to be copied.
      if (data_mode_ != DataMode::OUTPUT)
        CUDA_CHECK(cudaMemcpy(gpu_data_, cpu_data_, size(), cudaMemcpyHostToDevice));
    }
    mem_head_ = MemoryHead::HEAD_AT_DEVICE;
    return gpu_data_;
  }

private:
  DataMode data_mode_;

  Dtype *cpu_data_;
  Dtype *gpu_data_;

  // Take a tag to tell where the current memory has been saved (CPU or GPU).
  int mem_head_;
  // num, channels, height, width
  std::vector<int> shape_;
  int num_element_;
  int size_;
};

} // cux.
#endif //CUX_CUXDATA_HPP_