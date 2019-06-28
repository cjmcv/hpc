/*!
* \brief CuxData.
*/

#ifndef CUX_CUXDATA_HPP_
#define CUX_CUXDATA_HPP_

#include "util/util.h"

namespace cux {

template <typename Dtype>
class CuxDataBack {
public:
  CuxDataBack() : cpu_data_(nullptr), gpu_data_(nullptr) {}
  ~CuxDataBack() {
    if (cpu_data_ != nullptr) {
      delete[]cpu_data_;
      cpu_data_ = nullptr;
    }
    if (gpu_data_ != nullptr) {
      cudaFree(gpu_data_);
      gpu_data_ = nullptr;
    }
  }

  void SaveCpuData(const Dtype *cpu_data, const int num_element) {
    if (cpu_data_ == nullptr) {
      cpu_data_ = new Dtype[num_element];
    }
    memcpy(cpu_data_, cpu_data, sizeof(Dtype) * num_element);
  }
  void RestoreCpuData(Dtype *cpu_data, int &num_element) {
    if (cpu_data_ == nullptr) {
      return;
    }
    memcpy(cpu_data, cpu_data_, sizeof(Dtype) * num_element);
  }
  //
  void SaveGpuData(const Dtype *gpu_data, const int num_element) {
    if (gpu_data_ == nullptr) {
      CUDA_CHECK(cudaMalloc(&gpu_data_, sizeof(Dtype) * num_element));
    }
    CUDA_CHECK(cudaMemcpy(gpu_data_, gpu_data, sizeof(Dtype) * num_element, cudaMemcpyDeviceToDevice));
  }
  void RestoreGpuData(Dtype *gpu_data, int &num_element) {
    if (gpu_data_ == nullptr) {
      return;
    }
    CUDA_CHECK(cudaMemcpy(gpu_data, gpu_data_, sizeof(Dtype) * num_element, cudaMemcpyDeviceToDevice));
  }
public:
  Dtype *cpu_data_;
  Dtype *gpu_data_;
};

template <typename Dtype>
class CuxData {
public:
  explicit CuxData(const int num, const int channels, const int height, const int width) {
    shape_.clear();
    shape_.push_back(num);
    shape_.push_back(channels);
    shape_.push_back(height);
    shape_.push_back(width);

    num_element_ = num * channels * height * width;
    size_ = num_element_ * sizeof(Dtype);

    cpu_data_ = nullptr;
    gpu_data_ = nullptr;

    backup_ = nullptr;
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
    if (backup_ != nullptr) {
      delete backup_;
      backup_ = nullptr;
    }
  }

  inline const std::vector<int> &shape() { return shape_; }
  inline int num_element() { return num_element_; }
  inline int size() { return size_; }

  inline Dtype* cpu_data() { return cpu_data_; }
  inline Dtype* gpu_data() { return gpu_data_; }

  void Save(OpRunMode mode) {
    if (backup_ == nullptr)
      backup_ = new CuxDataBack<Dtype>();

    if (mode == ON_HOST)
      backup_->SaveCpuData(cpu_data_, num_element_);
    else
      backup_->SaveGpuData(gpu_data_, num_element_);
  }
  void Restore(OpRunMode mode) {
    if (backup_ == nullptr) {
      return;
    }
    if (mode == ON_HOST)
      backup_->RestoreCpuData(cpu_data_, num_element_);
    else
      backup_->RestoreGpuData(gpu_data_, num_element_);
  }

  Dtype* GetCpuData(DataFetchMode mode = NO_PUSH) {
    bool is_create = false;
    if (cpu_data_ == nullptr) {
      cpu_data_ = new Dtype[num_element_];
      is_create = true;
    }

    bool is_push = (mode == PUSH || (mode == PUSH_IF_EMPTY && is_create));
    if (is_push && gpu_data_ != nullptr) {
      CUDA_CHECK(cudaMemcpy(cpu_data_, gpu_data_, size(), cudaMemcpyDeviceToHost));
    }

    return cpu_data_;
  }

  Dtype* GetGpuData(DataFetchMode mode = NO_PUSH) {
    bool is_create = false;
    if (gpu_data_ == nullptr) {
      CUDA_CHECK(cudaMalloc(&gpu_data_, size()));
      is_create = true;
    }

    bool is_push = (mode == PUSH || (mode == PUSH_IF_EMPTY && is_create));
    if (is_push && cpu_data_ != nullptr) {
      CUDA_CHECK(cudaMemcpy(gpu_data_, cpu_data_, size(), cudaMemcpyHostToDevice));
    }

    return gpu_data_;
  }

private:
  Dtype *cpu_data_;
  Dtype *gpu_data_;

  // num, channels, height, width
  std::vector<int> shape_;
  int num_element_;
  // num_element_ * sizeof(Dtype)
  int size_;

  CuxDataBack<Dtype> *backup_;
};

} // cux.
#endif //CUX_CUXDATA_HPP_