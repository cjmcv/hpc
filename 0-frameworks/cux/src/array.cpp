/*!
* \brief Array.
*/

#include "array.h"

namespace cux {

Array4D::Array4D(int num, int channels, int height, int width) {
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

Array4D::~Array4D() {
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

// Save data to Array4DBackup.
void Array4D::Save(int type_flag, OpRunMode mode, bool is_save_if_empty) {
  if (backup_[type_flag] == nullptr)
    backup_[type_flag] = new Array4DBackup(type_flag);

  if (mode == ON_HOST) {
    if (is_save_if_empty && !(backup_[type_flag]->is_cpu_data_empty()))
      return;
    backup_[type_flag]->SaveCpuData(cpu_data_[type_flag], num_element_);
  }
  else { // 0N_DEVICE
    if (is_save_if_empty && !(backup_[type_flag]->is_gpu_data_empty()))
      return;
    backup_[type_flag]->SaveGpuData(gpu_data_[type_flag], num_element_);
  }
}
// Restore data from Array4DBackup.
void Array4D::Restore(int type_flag, OpRunMode mode) {
  if (backup_[type_flag] == nullptr) {
    CUXLOG_ERR("Restore -> backup_[%d] does not exist.", type_flag);
  }
  if (mode == ON_HOST)
    backup_[type_flag]->RestoreCpuData(cpu_data_[type_flag], num_element_);
  else // 0N_DEVICE
    backup_[type_flag]->RestoreGpuData(gpu_data_[type_flag], num_element_);
}

} // cux.
