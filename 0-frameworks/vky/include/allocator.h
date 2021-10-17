#ifndef VKY_ALLOCATOR_H_
#define VKY_ALLOCATOR_H_

#include <iostream>
#include <map>

#include <vulkan/vulkan.hpp>

#include "data_type.h"
#include "command.h"

namespace vky {

class Allocator {
public:
  Allocator(const vk::Device& device, const vk::PhysicalDevice& physical_device, Command *command) :
    device_(device), physical_device_(physical_device), command_(command) {};
  
  ~Allocator() {
    ClearIOBufferMemory();
  }

  BufferMemory* Malloc(size_t size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties);

  void Free(BufferMemory *bm);

  BufferMemory *GetInBufferMemory(int size);
  BufferMemory *GetOutBufferMemory(int size);

  /// Copy device buffers using the transient command pool.
  /// Fully sync, no latency hiding.
  inline void CopyBuf(const vk::Buffer& src, vk::Buffer& dst, const uint32_t size) {
    command_->CopyBuffer(src, dst, size);
  }

private:
  void ClearIOBufferMemory();
  vk::Buffer CreateBuffer(uint32_t buffer_size,
    vk::BufferUsageFlags usage,
    vk::MemoryPropertyFlags properties) const;

  vk::DeviceMemory AllocateMemory(const vk::Buffer& buf, uint32_t memory_type_index);

  // TODO: test.
  vk::DeviceMemory AllocateDedicatedMemory(size_t size, uint32_t memory_type_index, vk::Buffer buffer);

  uint32_t SelectMemory(const vk::Buffer& buffer, vk::MemoryPropertyFlags properties) const;

private:
  vk::Device device_;
  vk::PhysicalDevice physical_device_;

  Command *command_;

  std::map<int, BufferMemory *> staging_in_datas_;
  std::map<int, BufferMemory *> staging_out_datas_;
}; // Allocator

// TODO: template <typename Dtype>
class VkyData {

enum MemoryHead {
  AT_HOST = 0,
  AT_DEVICE = 1
};

public:
  VkyData(Allocator *allocator, int channels, int height, int width, float *data = nullptr) {
    allocator_ = allocator;

    channels_ = channels;
    height_ = height;
    width_ = width;

    len_ = channels_ * height_ * width_;

    host_data_ = nullptr;
    host_data_ = new float[len_];

    if (data != nullptr) {
      memcpy(host_data_, data, sizeof(float) * height * width);
    }
    memory_head_ = MemoryHead::AT_HOST;

    device_data_ = nullptr;
  }

  VkyData(Allocator *allocator,int len, float *data = nullptr)
    : VkyData(allocator, 1, 1, len, data) {};

  ~VkyData() {
    if (host_data_ != nullptr) {
      delete[]host_data_;
      host_data_ = nullptr;
    }
    if (device_data_ != nullptr) {
      allocator_->Free(device_data_);
      device_data_ = nullptr;
    }
  }

  float *host_data() const { return host_data_; }
  BufferMemory *device_data() { return device_data_; }
  int channels() const { return channels_; }
  int height() const { return height_; }
  int width() const { return width_; }

  // Get data without check.
  float *get_host_data() { 
    if (memory_head_ == AT_HOST)
      return host_data_; 
    else {
      // Copy data from device to host.
      // 1. Copy data to staging buffer; 
      // 2. Copy the data from staging buffer to host.
      // staging_out_data_->buffer_ is created by eTransferDst can be used as the dst buffer of transfer command.
      allocator_->CopyBuf(device_data_->buffer_, staging_out_data_->buffer_, sizeof(float) * len_);
      // It shouldn't be multiplied by sizeof(float).
      std::copy(staging_out_data_->mapped_ptr_, staging_out_data_->mapped_ptr_ + len_, host_data_);

      memory_head_ = AT_HOST;
      return host_data_;
    }
  }

  BufferMemory *get_device_data() {
    if (memory_head_ == AT_DEVICE)
      return device_data_;
    else {
      CheckMalloc();
      // Copy data from host to device.
      // 1. Copy data to staging buffer; 
      // 2. Copy the data from staging buffer to normal buffer.
      std::copy(host_data_, host_data_ + len_, staging_in_data_->mapped_ptr_);
      // staging_in_data_->buffer_ is created by eTransferSrc can be used as the src buffer of transfer command.
      allocator_->CopyBuf(staging_in_data_->buffer_, device_data_->buffer_, sizeof(float) * len_);

      memory_head_ = AT_DEVICE;
      return device_data_;
    }
  }

private:
  void CheckMalloc() {
    if (device_data_ != nullptr)
      return;

    staging_in_data_ = allocator_->GetInBufferMemory(len_ * sizeof(float));
    staging_out_data_ = allocator_->GetOutBufferMemory(len_ * sizeof(float));

    device_data_ = allocator_->Malloc(len_ * sizeof(float),
                               vk::BufferUsageFlagBits::eStorageBuffer,
                               vk::MemoryPropertyFlagBits::eDeviceLocal);
    device_data_->height_ = height_;
    device_data_->width_ = width_;
    device_data_->channels_ = channels_;
  }
private:
  Allocator *allocator_;
  float *host_data_;

  MemoryHead memory_head_;

  // Not owned. Getting from allocator.
  BufferMemory *staging_in_data_;
  BufferMemory *staging_out_data_;
  // Owned.
  BufferMemory *device_data_;

  int height_;
  int width_;
  int channels_;

  int len_;
}; // VkyData

} // namespace vky

#endif  // VKY_ALLOCATOR_H_