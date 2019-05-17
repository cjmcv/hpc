#ifndef VKY_ALLOCATOR_H_
#define VKY_ALLOCATOR_H_

#include <iostream>
#include <fstream>
#include <type_traits>
#include <map>

#include <vulkan/vulkan.hpp>

#include <device.h>

namespace vky {

class BufferMemory {
public:
  vk::Buffer buffer_;
  vk::BufferUsageFlags usage_;
  int buffer_range_;

  vk::DeviceMemory memory_;
  vk::MemoryPropertyFlags properties_;
  float *mapped_ptr_;
};

class Allocator {
public:
  Allocator(const vk::Device& device, const vk::PhysicalDevice& physical_device, const int compute_queue_familly_id) :
    device_(device), physical_device_(physical_device) {

    // Check.
    compute_queue_familly_id_ = compute_queue_familly_id;
  };
  ~Allocator() {
    ClearIOBufferMemory();
  }

  BufferMemory* Malloc(size_t size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties) {
    BufferMemory *bm = new BufferMemory();

    bm->usage_ = usage;
    bm->properties_ = properties;

    bm->buffer_ = CreateBuffer(size, usage, properties);
    uint32_t memory_id = SelectMemory(bm->buffer_, properties);
    bm->memory_ = AllocateMemory(bm->buffer_, memory_id);

    device_.bindBufferMemory(bm->buffer_, bm->memory_, 0);

    bm->mapped_ptr_ = nullptr;
    if (properties == vk::MemoryPropertyFlagBits::eHostVisible)
      bm->mapped_ptr_ = (float *)device_.mapMemory(bm->memory_, 0, size);

    return bm;
  }

  void Free(BufferMemory *bm) {
    if (bm == nullptr)
      return;

    if(bm->mapped_ptr_ != nullptr)
      device_.unmapMemory(bm->memory_);
    device_.destroyBuffer(bm->buffer_, 0);
    device_.freeMemory(bm->memory_, 0);

    delete bm;
    bm = nullptr;
  }

  BufferMemory *GetInBufferMemory(int size) {
    std::map<int, BufferMemory*>::iterator it = staging_in_datas_.find(size);
    if (it == staging_in_datas_.end()) {
      BufferMemory *new_one = Malloc(size,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible);

      staging_in_datas_[size] = new_one;
      return new_one;
    }
    else {
      return it->second;
    }
  }
  BufferMemory *GetOutBufferMemory(int size) {
    std::map<int, BufferMemory*>::iterator it = staging_out_datas_.find(size);
    if (it == staging_out_datas_.end()) {
      BufferMemory *new_one = Malloc(size,
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible);

      staging_out_datas_[size] = new_one;
      return new_one;
    }
    else {
      return it->second;
    }
  }

  // TODO: Use some variable from executor£¿
  /// Copy device_ buffers using the transient command pool.
  /// Fully sync, no latency hiding whatsoever.
  void CopyBuf(const vk::Buffer& src, vk::Buffer& dst, const uint32_t size) {
    auto cmd_pool = device_.createCommandPool({ vk::CommandPoolCreateFlagBits::eTransient, compute_queue_familly_id_ });
    auto cmd_buf = device_.allocateCommandBuffers({ cmd_pool, vk::CommandBufferLevel::ePrimary, 1 })[0];
    cmd_buf.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
    auto region = vk::BufferCopy(0, 0, size);
    cmd_buf.copyBuffer(src, dst, 1, &region);
    cmd_buf.end();
    auto queue = device_.getQueue(compute_queue_familly_id_, 0);
    auto submit_info = vk::SubmitInfo(0, nullptr, nullptr, 1, &cmd_buf);
    queue.submit({ submit_info }, nullptr);
    queue.waitIdle();
    device_.freeCommandBuffers(cmd_pool, 1, &cmd_buf);
    device_.destroyCommandPool(cmd_pool);
  }

private:
  void ClearIOBufferMemory() {
    std::map<int, BufferMemory*>::iterator it;
    for (it = staging_in_datas_.begin(); it != staging_in_datas_.end(); ++it)
      Free(it->second);

    for (it = staging_out_datas_.begin(); it != staging_out_datas_.end(); ++it)
      Free(it->second);
  }
  vk::Buffer CreateBuffer(uint32_t buffer_size,
                          vk::BufferUsageFlags usage, 
                          vk::MemoryPropertyFlags properties) const {

    if (physical_device_.getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu
      && properties == vk::MemoryPropertyFlagBits::eDeviceLocal
      && usage == vk::BufferUsageFlagBits::eStorageBuffer) {
      usage |= vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
    }

    auto create_info = vk::BufferCreateInfo(vk::BufferCreateFlags(), buffer_size, usage);
    return device_.createBuffer(create_info);
  }

  vk::DeviceMemory AllocateMemory(const vk::Buffer& buf, uint32_t memory_type_index) {

    auto memory_reqs = device_.getBufferMemoryRequirements(buf);

    vk::MemoryAllocateInfo alloc_info;
    alloc_info.setAllocationSize(memory_reqs.size);
    alloc_info.setMemoryTypeIndex(memory_type_index);

    // TODO: move it?
    //flags_ = physical_device_.getMemoryProperties().memoryTypes[memory_type_index].propertyFlags;

    return device_.allocateMemory(alloc_info);
  }

  // TODO: test.
  vk::DeviceMemory AllocateDedicatedMemory(size_t size, uint32_t memory_type_index, vk::Buffer buffer) {
    vk::MemoryAllocateInfo alloc_info;
    alloc_info.setAllocationSize(size);
    alloc_info.setMemoryTypeIndex(memory_type_index);

    vk::MemoryDedicatedAllocateInfoKHR dedicated_alloc_info;
    dedicated_alloc_info.setPNext(0);
    dedicated_alloc_info.setBuffer(buffer);

    alloc_info.setPNext(&dedicated_alloc_info);

    return device_.allocateMemory(alloc_info);
  }

  ////////
  uint32_t SelectMemory(const vk::Buffer& buffer, vk::MemoryPropertyFlags properties) const {
    auto mem_properties = physical_device_.getMemoryProperties();
    auto memory_reqs = device_.getBufferMemoryRequirements(buffer);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; ++i) {
      if ((memory_reqs.memoryTypeBits & (1u << i))
        && ((properties & mem_properties.memoryTypes[i].propertyFlags) == properties)) {
        return i;
      }
    }
    throw std::runtime_error("failed to select memory with required properties");
  }

private:
  vk::Device device_;
  vk::PhysicalDevice physical_device_;

  //vk::MemoryPropertyFlags flags_;

  // Temp?
  uint32_t compute_queue_familly_id_; // Given by executor.

  std::map<int, BufferMemory *> staging_in_datas_;
  std::map<int, BufferMemory *> staging_out_datas_;
}; // Allocator

// TODO: template <typename Dtype>
class VkyData {
public:
  VkyData(Allocator *allocator, int channels, int height, int width, float *data = nullptr) {
    allocator_ = allocator;

    channels_ = channels;
    height_ = height;
    width_ = width;

    len_ = channels_ * height_ * width_;

    if (data == nullptr) {
      cpu_data_ = new float[len_];
      is_cpu_data_hold_ = true;
    }
    else {
      cpu_data_ = data;
      is_cpu_data_hold_ = false;
    }

    staging_in_data_ = allocator_->GetInBufferMemory(len_ * sizeof(float));
    staging_out_data_ = allocator_->GetOutBufferMemory(len_ * sizeof(float));

    data_ = allocator_->Malloc(len_ * sizeof(float),
                               vk::BufferUsageFlagBits::eStorageBuffer,
                               vk::MemoryPropertyFlagBits::eDeviceLocal);
  }

  VkyData(Allocator *allocator,int len, float *data = nullptr)
    : VkyData(allocator, 1, 1, len, data) {};

  ~VkyData() {
    if (is_cpu_data_hold_) {
      delete[]cpu_data_;
      cpu_data_ = nullptr;
    }
    allocator_->Free(data_);
  }

  void PushFromHost2Device() {
    std::copy(cpu_data_, cpu_data_ + sizeof(float) * len_, staging_in_data_->mapped_ptr_);
    allocator_->CopyBuf(staging_in_data_->buffer_, data_->buffer_, sizeof(float) * len_);
  }

  void PushFromDevice2Host() {
    allocator_->CopyBuf(data_->buffer_, staging_out_data_->buffer_, sizeof(float) * len_);
    // It shouldn't be multiplied by sizeof(float).
    std::copy(staging_out_data_->mapped_ptr_, staging_out_data_->mapped_ptr_ + len_, cpu_data_);
  }

  // Get data without check.
  float *cpu_data() const { return cpu_data_; }
  vk::Buffer &device_data() { return data_->buffer_; }

  int channels() const { return channels_; }
  int height() const { return height_; }
  int width() const { return width_; }

private:
  Allocator *allocator_;
  float *cpu_data_;
  bool is_cpu_data_hold_;

  // Not owned. Getting from allocator.
  BufferMemory *staging_in_data_;
  BufferMemory *staging_out_data_;
  // Owned.
  BufferMemory *data_;

  int channels_;
  int height_;
  int width_;

  int len_;
}; // VkyData

} // namespace vky

#endif  // VKY_ALLOCATOR_H_