#ifndef HCS_BLOB_H_
#define HCS_BLOB_H_

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "common.hpp"
#include "util/util.hpp"

namespace hcs {

class Blob {
public:
  Blob(std::string name) :is_created_(false),
    data_(nullptr), buffer_(nullptr),
    len_(0), size_(0),
    object_id_(-1), mode_(-1), type_(-1),
    name_(name), node_name_("noname"){
    shape_.clear();
  }
  ~Blob() { Release(); }

  inline void *data() { return data_; }
  inline int mode() const { return mode_; }
  inline int len() const { return len_; }
  inline const std::string &name() const { return name_; }
  inline const std::string &node_name() const { return node_name_; }
  inline std::vector<int> &shape() { return shape_; };
  inline void set_node_name(std::string name) { node_name_ = name; }

  bool Create(int num, int channel, int height, int width, int mode, int type);
  void Release();

  void ReleaseBuffer();
  // Get the pointer of buffer_;
  void *GetOtherSideBuffer();
  // Push data from buffer_ to data_.
  bool CheckPushBuffer(cudaStream_t stream = nullptr);
  bool CopyTo(Blob *to);
  bool CloneTo(Blob *to);
  bool SyncParams(int num, int channel, int height, int width, int mode, int type);

public:
  int object_id_;

private:  
  bool is_created_;
  
  void *data_;
  // buffer_ is the same size as data_, but at different devices.
  // For example, if data_ at host, buffer_ will be at device.
  void *buffer_;
  // The number of elements.
  int len_;
  // sizeof(type) * len.
  int size_;

  // host / device.
  int mode_;
  // int / float / ...
  int type_;
  // Mainly 4-dimension.
  std::vector<int> shape_;
  // blob name.
  std::string name_;
  // The node served by the blob.
  std::string node_name_;
};

bool Blob::Create(int num, int channel, int height, int width, int mode, int type) {
  object_id_ = -1;

  shape_.clear();
  shape_.push_back(num);
  shape_.push_back(channel);
  shape_.push_back(height);
  shape_.push_back(width);

  len_ = num * channel * height * width;
  if (len_ <= 0) { return false; }

  type_ = type;
  TYPE_SWITCH(type_, T, size_ = sizeof(T) * len_;);

  mode_ = mode;
  if (mode_ == ON_HOST) {
    TYPE_SWITCH(type_, T, data_ = new T[len_];);
  }
  else {
    TYPE_SWITCH(type_, T,
      CUDA_CHECK(cudaMalloc(&data_, size_));
    );
  }

  // Whenever data_ is created or recreated, the Buffer needs to be freed.
  // This can simplify the process of getting buffer_;
  ReleaseBuffer();

  is_created_ = true;
  return true;
}

void Blob::Release() {
  if (data_ != nullptr) {
    if (mode_ == ON_HOST)
      delete[]data_;
    else
      CUDA_CHECK(cudaFree(data_));

    data_ = nullptr;
  }
  ReleaseBuffer();

  is_created_ = false;
}

void Blob::ReleaseBuffer() {
  if (buffer_ != nullptr) {
    if (mode_ == ON_HOST) {
      CUDA_CHECK(cudaFree(buffer_));
    }
    else {
      delete[]buffer_;
    }
    buffer_ = nullptr;
  }
}

void *Blob::GetOtherSideBuffer() {
  if (!is_created_) {
    LOG(ERROR) << "GetOtherSideBuffer is not allowed when Blob is not created.";
  }
  // When data_ is on the host side, buffer_ is on the device side.
  if (buffer_ == nullptr) {
    if (mode_ == ON_HOST) {
      TYPE_SWITCH(type_, T,
        CUDA_CHECK(cudaMalloc(&buffer_, sizeof(T) * len_));
      );
    }
    else {
      TYPE_SWITCH(type_, T, buffer_ = new T[len_];);
    }
  }
  return buffer_;
}

bool Blob::CheckPushBuffer(cudaStream_t stream) {
  if (buffer_ == nullptr) {
    //LOG(ERROR) << "This buffer is empty.";
    return false;
  }
  
  if (mode_ == ON_HOST) {
    if (stream == nullptr) {
      CUDA_CHECK(cudaMemcpy(data_, buffer_, size_, cudaMemcpyDeviceToHost));
    }
    else {
      CUDA_CHECK(cudaMemcpyAsync(data_, buffer_, size_, cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
  }
  else {
    if (stream == nullptr) {
      CUDA_CHECK(cudaMemcpy(data_, buffer_, size_, cudaMemcpyHostToDevice));
    }
    else {
      CUDA_CHECK(cudaMemcpyAsync(data_, buffer_, size_, cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
  }
  return true;
}

bool Blob::SyncParams(int num, int channel, int height, int width, int mode, int type) {
  // Check dimension.
  if (is_created_ == false) {
    return Create(num, channel, height, width, mode, type);
  }

  int len = num * channel * height * width;
  if (mode_ != mode || type_ != type || len_ != len) {
    Release();
    return Create(num, channel, height, width, mode, type);
  }

  shape_.clear();
  shape_.push_back(num);
  shape_.push_back(channel);
  shape_.push_back(height);
  shape_.push_back(width);

  return true;
}

bool Blob::CloneTo(Blob *to) {
  if (!is_created_) {
    LOG(ERROR) << "CloneTo -> !is_created_.";
    return false;
  }
  
  to->SyncParams(shape_[0], shape_[1], shape_[2], shape_[3], mode_, type_);

  // Copy.
  if (mode_ == ON_HOST) {
    memcpy(to->data_, data_, size_);
  }
  else {
    CUDA_CHECK(cudaMemcpy(to->data_, data_, size_, cudaMemcpyDeviceToDevice));
  }

  // Pass object id.
  to->object_id_ = object_id_;
  return true;
}

}  // namespace hcs.

#endif // HCS_BLOB_H_