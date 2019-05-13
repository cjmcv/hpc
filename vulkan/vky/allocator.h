#ifndef VKY_ALLOCATOR_H_
#define VKY_ALLOCATOR_H_

#include <iostream>
#include <fstream>
#include <type_traits>
#include <vulkan/vulkan.hpp>

#include <device.h>

namespace vky {

class Allocator {
public:
  Allocator(const vk::Device& device, const vk::PhysicalDevice& physical_device, const int compute_queue_familly_id) :
    device_(device), physical_device_(physical_device) {

    properties_ = vk::MemoryPropertyFlagBits::eHostVisible; //eDeviceLocal

    // Check.
    compute_queue_familly_id_ = compute_queue_familly_id;
  };
  virtual ~Allocator() {}

public:
  vk::Buffer CreateBuffer(uint32_t buffer_size,
                          vk::BufferUsageFlags usage) const {

    if (physical_device_.getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu
      && properties_ == vk::MemoryPropertyFlagBits::eDeviceLocal
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
    flags_ = physical_device_.getMemoryProperties().memoryTypes[memory_type_index].propertyFlags;

    return device_.allocateMemory(alloc_info);
  }

  // TODO: test.
  vk::DeviceMemory AllocateDedicatedMemory(size_t size, uint32_t memory_type_index, VkBuffer buffer) {
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
  vk::Device device() const { return device_; }

  uint32_t SelectMemory(const vk::Buffer& buf) const {
    auto mem_properties = physical_device_.getMemoryProperties();
    auto memory_reqs = device_.getBufferMemoryRequirements(buf);
    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; ++i) {
      if ((memory_reqs.memoryTypeBits & (1u << i))
        && ((properties_ & mem_properties.memoryTypes[i].propertyFlags) == properties_)) {
        return i;
      }
    }
    throw std::runtime_error("failed to select memory with required properties");
  }
  void BindBufferMemory(vk::Buffer &buf, vk::DeviceMemory &mem) {
    device_.bindBufferMemory(buf, mem, 0);
  }
  void *MapMemory(vk::DeviceMemory &mem, int size) {
    return device_.mapMemory(mem, 0, size);
  }
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
  vk::Device device_;
  vk::PhysicalDevice physical_device_;

  vk::MemoryPropertyFlags flags_;

  // Temp?
  vk::MemoryPropertyFlags properties_;
  uint32_t compute_queue_familly_id_; // Given by executor.
}; // Allocator

// TODO: The basic data unit.
// TODO: The allocator is given by executor?
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

    // Temp?
    vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer;
    buffer_ = allocator_->CreateBuffer(len_ * sizeof(float), usage);
    uint32_t memory_id = allocator_->SelectMemory(buffer_);

    mem_ = allocator_->AllocateMemory(buffer_, memory_id);

    allocator_->BindBufferMemory(buffer_, mem_);

    // TODO.
    map_data_ = (float *)allocator_->MapMemory(mem_, len_ * sizeof(float));
    
    printf("finish.\n");
  }

  VkyData(Allocator *allocator,int len, float *data = nullptr)
    : VkyData(allocator, 1, 1, len, data) {};

  void PushFromHost2Device() { 
    std::copy(cpu_data_, cpu_data_ + sizeof(float) * len_, map_data_);
  }
  void PushFromDevice2Host() {
    std::copy(map_data_, map_data_ + sizeof(float) * len_, cpu_data_);
  }

  // Get data without check.
  float *cpu_data() const { return cpu_data_; }
  vk::Buffer &device_data() { return buffer_; }

  int channels() const { return channels_; }
  int height() const { return height_; }
  int width() const { return width_; }

  ~VkyData() {
    if (is_cpu_data_hold_) {
      delete[]cpu_data_;
      cpu_data_ = nullptr;
    }
  }

private:
  Allocator *allocator_;
  float *cpu_data_;
  bool is_cpu_data_hold_;

  vk::Buffer buffer_;
  int buffer_range_;  
  float *map_data_;

  int channels_;
  int height_;
  int width_;
  int len_;

  // Temp?
  vk::DeviceMemory mem_;
}; // VkyData

} // namespace vky

#endif  // VKY_ALLOCATOR_H_