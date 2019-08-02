/*!
* \brief Operator.
*/

#ifndef CUX_OP_ASSISTOR_H_
#define CUX_OP_ASSISTOR_H_

#include "util/util.h"

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
  bool CheckArray(DType *in, int len, int id) {
    if (id == 0) {
      SetBenchmarkData(in, len);
      return true;
    }
    float diff = 0.0;
    for (int i = 0; i < len; i++) {
      float t = prev_data_[i] - (float)in[i];
      diff += (t >= 0 ? t : -t);
    }
    if (diff < DBL_MIN) {
      CUXLOG_COUT("Pass: V0 vs V%d -> (diff: %f, first number: %f, %f)",
        id, diff, (float)prev_data_[0], (float)in[0]);
      return true;
    }
    else {
      CUXLOG_COUT("Fail: V0 vs V%d -> (diff: %f, first number: %f, %f)",
        id, diff, (float)prev_data_[0], (float)in[0]);
      return false;
    }
  }

private:
  // Set the benchmark data, which is correct by default.
  template <typename DType>
  void SetBenchmarkData(DType *in, int len) {
    if (prev_data_ == nullptr) {
      prev_data_ = new float[len];
      len_ = len;
    }
    else if (len_ != len) {
      delete[]prev_data_;
      prev_data_ = new float[len];
      len_ = len;
    }
    for (int i = 0; i < len; i++) {
      prev_data_[i] = in[i];
    }
  }

private:
  float *prev_data_;
  int len_;
};

class OpAssistor {
public:
  OpAssistor(Device *device)
    : device_(device), cublas_handle_(nullptr), launch_config_(nullptr){
    if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
      CUXLOG_ERR("Cannot create Cublas handle. Cublas won't be available.");
    }
    launch_config_ = new LaunchConfig(device);
    if (launch_config_ == nullptr) {
      CUXLOG_ERR("Cannot new a LaunchConfig.");
    }
    checker_ = new ResultChecker();
    if (checker_ == nullptr) {
      CUXLOG_ERR("Cannot new a checker_.");
    }
  }
  ~OpAssistor() {
    if (cublas_handle_ != nullptr) {
      cublasDestroy(cublas_handle_);
      cublas_handle_ = nullptr;
    }
    if (launch_config_ != nullptr) {
      delete launch_config_;
      launch_config_ = nullptr;
    }
    if (checker_ != nullptr) {
      delete checker_;
      checker_ = nullptr;
    }
  }
  inline Device *device() { return device_; }
  inline cublasHandle_t cublas_handle() { return cublas_handle_; }
  inline LaunchConfig *launch_config() { return launch_config_; }
  inline ResultChecker *checker() { return checker_; }

private:
  Device *device_;

  cublasHandle_t cublas_handle_;
  // Generate kernel launch config and Get occupancy.
  LaunchConfig *launch_config_;
  // Verify the correctness of the output.
  ResultChecker *checker_;
};

} // cux.

#endif //CUX_OP_ASSISTOR_H_
