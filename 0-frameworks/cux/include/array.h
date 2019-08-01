/*!
* \brief Array4D.
*/

#ifndef CUX_ARRAY4D_H_
#define CUX_ARRAY4D_H_

#include <vector>
#include "util/util.h"

namespace cux {

// Mainly used for backup and restore of Array4D.
class Array4DBackup {
public:
  Array4DBackup(int type_flag) : cpu_data_(nullptr),
    gpu_data_(nullptr),
    type_flag_(type_flag) {
    TYPE_SWITCH(type_flag, T, element_size_ = sizeof(T););
  }

  ~Array4DBackup() {
    if (cpu_data_ != nullptr) {
      delete[]cpu_data_;
      cpu_data_ = nullptr;
    }
    if (gpu_data_ != nullptr) {
      cudaFree(gpu_data_);
      gpu_data_ = nullptr;
    }
  }
  inline int type_flag() { return type_flag_; }

  void SaveCpuData(const void *cpu_data, const int num_element) {
    if (cpu_data_ == nullptr) {
      TYPE_SWITCH(type_flag_, T, cpu_data_ = new T[num_element];);
    }
    memcpy(cpu_data_, cpu_data, element_size_ * num_element);
  }
  void RestoreCpuData(void *cpu_data, int &num_element) {
    if (cpu_data_ == nullptr) {
      CUXLOG_ERR("RestoreCpuData -> The data has not been saved. Please check again.");
    }
    memcpy(cpu_data, cpu_data_, element_size_ * num_element);
  }
  //
  void SaveGpuData(const void *gpu_data, const int num_element) {
    if (gpu_data_ == nullptr) {
      CUDA_CHECK(cudaMalloc(&gpu_data_, element_size_ * num_element));
    }
    CUDA_CHECK(cudaMemcpy(gpu_data_, gpu_data, element_size_ * num_element, cudaMemcpyDeviceToDevice));
  }
  void RestoreGpuData(void *gpu_data, int &num_element) {
    if (gpu_data_ == nullptr) {
      CUXLOG_ERR("RestoreGpuData -> The data has not been saved. Please check again.");
    }
    CUDA_CHECK(cudaMemcpy(gpu_data, gpu_data_, element_size_ * num_element, cudaMemcpyDeviceToDevice));
  }
public:
  void *cpu_data_;
  void *gpu_data_;

  int type_flag_;
  int element_size_;
};

class Array4D {
public:
  explicit Array4D(int num, int channels, int height, int width) {
    shape_.clear();
    shape_.push_back(num);
    shape_.push_back(channels);
    shape_.push_back(height);
    shape_.push_back(width);
    num_element_ = num * channels * height * width;

    const int type_sum = DataTypeSum::kNum;
    cpu_data_.resize(type_sum);
    gpu_data_.resize(type_sum);
    backup_.resize(type_sum);

    for (int i = 0; i < type_sum; i++) {
      cpu_data_[i] = nullptr;
      gpu_data_[i] = nullptr;
      backup_[i] = nullptr;
    }
  }

  ~Array4D() {
    for (int i = 0; i < DataTypeSum::kNum; i++) {
      if (cpu_data_[i] != nullptr) {
        delete[]cpu_data_[i];
        cpu_data_[i] = nullptr;
      }
      if (gpu_data_[i] != nullptr) {
        cudaFree(gpu_data_[i]);
        gpu_data_[i] = nullptr;
      }
      if (backup_[i] != nullptr) {
        delete backup_[i];
        backup_[i] = nullptr;
      }
    }
  }

  inline const std::vector<int> shape() const { return shape_; }
  inline int num_element() { return num_element_; }

  inline void* cpu_data(int type_flag) { return cpu_data_[type_flag]; }
  inline void* gpu_data(int type_flag) { return gpu_data_[type_flag]; }

  // Save data to Array4DBackup.
  void Save(int type_flag, OpRunMode mode) {
    if (backup_[type_flag] == nullptr)
      backup_[type_flag] = new Array4DBackup(type_flag);

    if (mode == ON_HOST)
      backup_[type_flag]->SaveCpuData(cpu_data_[type_flag], num_element_);
    else // 0N_DEVICE
      backup_[type_flag]->SaveGpuData(gpu_data_[type_flag], num_element_);
  }
  // Restore data from Array4DBackup.
  void Restore(int type_flag, OpRunMode mode) {
    if (backup_[type_flag] == nullptr) {
      CUXLOG_ERR("Restore -> backup_[%d] does not exist.", type_flag);
    }
    if (mode == ON_HOST)
      backup_[type_flag]->RestoreCpuData(cpu_data_[type_flag], num_element_);
    else // 0N_DEVICE
      backup_[type_flag]->RestoreGpuData(gpu_data_[type_flag], num_element_);
  }

  template<typename DType>
  DType* GetCpuData(DataFetchMode mode = NO_PUSH) {
    int type_flag = cux::DataType<DType>::kFlag;

    bool is_create = false;
    if (cpu_data_[type_flag] == nullptr) {
      cpu_data_[type_flag] = new DType[num_element_];
      is_create = true;
    }
    bool is_push = (mode == PUSH || (mode == PUSH_IF_EMPTY && is_create));
    if (is_push && gpu_data_[type_flag] != nullptr) {
      CUDA_CHECK(cudaMemcpy(cpu_data_[type_flag], gpu_data_[type_flag], num_element_ * sizeof(DType), cudaMemcpyDeviceToHost));
    }
    return static_cast<DType*>(cpu_data_[type_flag]);
  }
  template<typename DType>
  DType* GetGpuData(DataFetchMode mode = NO_PUSH) {
    int type_flag = cux::DataType<DType>::kFlag;

    bool is_create = false;
    if (gpu_data_[type_flag] == nullptr) {
      CUDA_CHECK(cudaMalloc(&gpu_data_[type_flag], num_element_ * sizeof(DType)));
      is_create = true;
    }
    bool is_push = (mode == PUSH || (mode == PUSH_IF_EMPTY && is_create));
    if (is_push && cpu_data_[type_flag] != nullptr) {
      CUDA_CHECK(cudaMemcpy(gpu_data_[type_flag], cpu_data_[type_flag], num_element_ * sizeof(DType), cudaMemcpyHostToDevice));
    }
    return static_cast<DType*>(gpu_data_[type_flag]);
  }

  template<typename SrcType, typename DstType>
  void PrecsCpuCvt() {
    int src_data_type = DataType<SrcType>::kFlag;
    int dst_data_type = DataType<DstType>::kFlag;

    if (src_data_type == dst_data_type) {
      CUXLOG_ERR("PrecsCpuCvt -> src_data_type == dst_data_type.");
    }
    if (cpu_data_[src_data_type] == nullptr) {
      CUXLOG_ERR("PrecsCpuCvt -> cpu_data_[src_data_type] == nullptr.");
    }
    if (cpu_data_[dst_data_type] == nullptr) {
      TYPE_SWITCH(dst_data_type, T, cpu_data_[dst_data_type] = new T[num_element_];);
    }

    DstType *dst = (DstType *)cpu_data_[dst_data_type];
    SrcType *src = (SrcType *)cpu_data_[src_data_type];
    for (int i = 0; i < num_element_; i++) {
      dst[i] = src[i];
    }
  }

private:  
  // 4d: num, channels, height, width
  std::vector<int> shape_;
  int num_element_;

  std::vector<void*> cpu_data_;
  std::vector<void*> gpu_data_;

  // For data backup and restore.
  // Refer to the matrix C in gemm.
  std::vector<Array4DBackup*> backup_;
};

} // cux.
#endif //CUX_ARRAY4D_H_