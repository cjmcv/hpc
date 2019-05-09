#ifndef VKY_ALLOCATOR_H_
#define VKY_ALLOCATOR_H_

#include <iostream>
#include <fstream>
#include <type_traits>
#include <vulkan/vulkan.hpp>

#include <device.h>

namespace vky {

/// Device buffer owning its chunk of memory.
template<class T>
class Allocator2 {
  // Helper class to access to (host-visible!!!) device memory from the host. 
  // Unmapping memory is not necessary.
  struct BufferHostView {
    using ptr_type = T*;

    const vk::Device device;
    const vk::DeviceMemory devMemory;
    const ptr_type data; ///< points to the first element
    const size_t size;   ///< number of elements

                         /// Constructor
    explicit BufferHostView(vk::Device device, vk::DeviceMemory devMem,
      size_t nelements)
      : device(device), devMemory(devMem)
      , data(ptr_type(device.mapMemory(devMem, 0, nelements * sizeof(T))))
      , size(nelements) {}

    ptr_type begin() { return data; }
    ptr_type end() { return data + size; }
  }; // BufferHostView

public:
  using value_type = T;

  Allocator2(Allocator2&&) = default;
  auto operator=(Allocator2&&)->Allocator2& = default;

  // Constructor
  // construction and binding.
  explicit Allocator2(const vk::Device& device, const vk::PhysicalDevice& phys_device,
    uint32_t n_elements,
    vk::MemoryPropertyFlags properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
    vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer)
    : dev_(&device) {

    physdev_ = phys_device;
    buf_ = CreateBuffer(device, n_elements * sizeof(T), update_usage(phys_device, properties, usage));
    uint32_t memory_id = SelectMemory(physdev_, *dev_, buf_, properties);
    mem_ = AllocMemory(*dev_, buf_, memory_id);
    flags_ = physdev_.getMemoryProperties().memoryTypes[memory_id].propertyFlags;
    size_ = n_elements;
    dev_->bindBufferMemory(buf_, mem_, 0);
  }

  /// Destructor
  ~Allocator2() noexcept {
    if (dev_) {
      dev_->freeMemory(mem_);
      dev_->destroyBuffer(buf_);
      dev_.release();
    }
  }

  size_t size() const { return size_; } /// @return number of items in the buffer
  operator vk::Buffer& () {
    return *reinterpret_cast<vk::Buffer*>(this + offsetof(Allocator2, buf_));
  }
  operator const vk::Buffer& () const {
    return *reinterpret_cast<const vk::Buffer*>(this + offsetof(Allocator2, buf_));
  }

  static Allocator2 fromHost(T *in_data, int len, const vk::Device& device, const vk::PhysicalDevice& physDev,
    vk::MemoryPropertyFlags properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
    vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer);

  void to_host(T *out_data, int len);

private:
  ///
  BufferHostView host_view() {
    return BufferHostView(*dev_, mem_, size());
  }

  /// crutch to modify buffer usage
  vk::BufferUsageFlags update_usage(const vk::PhysicalDevice& phys_device,
    vk::MemoryPropertyFlags properties,
    vk::BufferUsageFlags usage) const;

  /// Select memory with desired properties.
  /// @return id of the suitable memory, -1 if no suitable memory found.
  uint32_t SelectMemory(const vk::PhysicalDevice& phys_dev,
    const vk::Device& device,
    const vk::Buffer& buf,
    const vk::MemoryPropertyFlags properties) const;

  vk::DeviceMemory AllocMemory(const vk::Device& device,
    const vk::Buffer& buf,
    uint32_t memory_id) const;

  /// Copy device_ buffers using the transient command pool.
  /// Fully sync, no latency hiding whatsoever.
  static void CopyBuf(const vk::Buffer& src, vk::Buffer& dst, const uint32_t size,
    const vk::Device& device, const vk::PhysicalDevice& phys_dev) {

    const auto qf_id = GetComputeQueueFamilyId_ttt(phys_dev); // queue family id, TODO: use transfer queue
    auto cmd_pool = device.createCommandPool({ vk::CommandPoolCreateFlagBits::eTransient, qf_id });
    auto cmd_buf = device.allocateCommandBuffers({ cmd_pool, vk::CommandBufferLevel::ePrimary, 1 })[0];
    cmd_buf.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
    auto region = vk::BufferCopy(0, 0, size);
    cmd_buf.copyBuffer(src, dst, 1, &region);
    cmd_buf.end();
    auto queue = device.getQueue(qf_id, 0);
    auto submit_info = vk::SubmitInfo(0, nullptr, nullptr, 1, &cmd_buf);
    queue.submit({ submit_info }, nullptr);
    queue.waitIdle();
    device.freeCommandBuffers(cmd_pool, 1, &cmd_buf);
    device.destroyCommandPool(cmd_pool);
  }

  /// Create buffer on a device_. Does NOT allocate memory.
  vk::Buffer CreateBuffer(const vk::Device& device,
    uint32_t buf_size,
    vk::BufferUsageFlags usage) const;

  /// @return the index of a queue family that supports compute operations.
  /// Groups of queues that have the same capabilities (for instance_, they all supports graphics
  /// and computer operations), are grouped into queue families.
  /// When submitting a command buffer, you must specify to which queue in the family you are submitting to.
  static uint32_t GetComputeQueueFamilyId_ttt(const vk::PhysicalDevice& physicalDevice) {
#define ALL(x) begin(x), end(x)
    auto queueFamilies = physicalDevice.getQueueFamilyProperties();

    // prefer using compute-only queue. eTransfer / eCompute
    auto queue_it = std::find_if(ALL(queueFamilies), [](auto& f) {
      auto maskedFlags = ~vk::QueueFlagBits::eSparseBinding & f.queueFlags; // ignore sparse binding flag 
      return 0 < f.queueCount                                               // queue family does have some queues in it
        && (vk::QueueFlagBits::eCompute & maskedFlags)
        && !(vk::QueueFlagBits::eGraphics & maskedFlags);
    });
    if (queue_it != end(queueFamilies)) {
      return uint32_t(std::distance(begin(queueFamilies), queue_it));
    }

    // otherwise use any queue that has compute flag set
    queue_it = std::find_if(ALL(queueFamilies), [](auto& f) {
      auto maskedFlags = ~vk::QueueFlagBits::eSparseBinding & f.queueFlags;
      return 0 < f.queueCount && (vk::QueueFlagBits::eCompute & maskedFlags);
    });
    if (queue_it != end(queueFamilies)) {
      return uint32_t(std::distance(begin(queueFamilies), queue_it));
    }

    throw std::runtime_error("could not find a queue family that supports compute operations");
  }

private:
  vk::Buffer buf_;                        ///< device buffer
  vk::DeviceMemory mem_;                  ///< associated chunk of device memorys
  vk::PhysicalDevice physdev_;            ///< physical device owning the memory
  std::unique_ptr<const vk::Device> dev_; ///< pointer to logical device. no real ownership, just to provide value semantics to the class.
  vk::MemoryPropertyFlags flags_;         ///< Actual flags of allocated memory. Can be a superset of requested flags.
  size_t size_;                           ///< number of elements. actual allocated memory may be a bit bigger than necessary.

}; // Allocator2

class Allocator {
public:
  Allocator() {};
  virtual ~Allocator() {}

}; // Allocator

// TODO: The basic data unit.
// TODO: template <typename Dtype>
class VkyData {
public:
  VkyData() {};
  VkyData(int channels, int height, int width) {};

private:
  Allocator *allocator_;
  float *cpu_data_;

  vk::Buffer buffer_;
  int buffer_range_;
}; // Data

} // namespace vky

#endif  // VKY_ALLOCATOR_H_