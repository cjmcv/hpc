#include "allocator.h"

#include <iostream>

namespace vky {

/////////////
//  Public.
/////////////
template<class T>
Allocator2<T> Allocator2<T>::fromHost(T *in_data, int len,
  const vk::Device& device, const vk::PhysicalDevice& physDev,
  vk::MemoryPropertyFlags properties, vk::BufferUsageFlags usage) {

  auto r = Allocator2<T>(device, physDev, len, properties, usage);
  if (r.flags_ & vk::MemoryPropertyFlagBits::eHostVisible) { // memory is host-visible
    std::copy(in_data, in_data + sizeof(T) * len, r.host_view().data);
  }
  else { // memory is not host visible, use staging buffer
    auto stage_buf = fromHost(in_data, len, device, physDev,
      vk::MemoryPropertyFlagBits::eHostVisible,
      vk::BufferUsageFlagBits::eTransferSrc);
    CopyBuf(stage_buf, r, stage_buf.size() * sizeof(T), device, physDev);
  }
  return r;
}

template<class T>
void Allocator2<T>::to_host(T *out_data, int len) {
  if (flags_ & vk::MemoryPropertyFlagBits::eHostVisible) { // memory IS host visible
    auto hv = host_view();
    std::copy(std::begin(hv), std::end(hv), out_data);
  }
  else {
    // memory is not host visible, use staging buffer
    // copy device memory to staging buffer
    auto stage_buf = Allocator2(*dev_, physdev_, size(),
      vk::MemoryPropertyFlagBits::eHostVisible,
      vk::BufferUsageFlagBits::eTransferDst);
    CopyBuf(buf_, stage_buf, size() * sizeof(T), *dev_, physdev_);
    stage_buf.to_host(out_data, len); // copy from staging buffer to host
  }
}

/////////////
//  Private.
/////////////
/// crutch to modify buffer usage
template<class T>
vk::BufferUsageFlags Allocator2<T>::update_usage(const vk::PhysicalDevice& phys_device,
                                                vk::MemoryPropertyFlags properties,
                                                vk::BufferUsageFlags usage) const {
  if (phys_device.getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu
    && properties == vk::MemoryPropertyFlagBits::eDeviceLocal
    && usage == vk::BufferUsageFlagBits::eStorageBuffer) {
    usage |= vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
  }
  return usage;
}

/// Select memory with desired properties.
/// @return id of the suitable memory, -1 if no suitable memory found.
template<class T>
uint32_t Allocator2<T>::SelectMemory(const vk::PhysicalDevice& phys_dev,
                                    const vk::Device& device,
                                    const vk::Buffer& buf,
                                    const vk::MemoryPropertyFlags properties) const {
  auto mem_properties = phys_dev.getMemoryProperties();
  auto memory_reqs = device.getBufferMemoryRequirements(buf);
  for (uint32_t i = 0; i < mem_properties.memoryTypeCount; ++i) {
    if ((memory_reqs.memoryTypeBits & (1u << i))
      && ((properties & mem_properties.memoryTypes[i].propertyFlags) == properties)) {
      return i;
    }
  }
  throw std::runtime_error("failed to select memory with required properties");
}

template<class T>
vk::DeviceMemory Allocator2<T>::AllocMemory(const vk::Device& device,
                                           const vk::Buffer& buf,
                                           uint32_t memory_id) const {
  auto memoryReqs = device.getBufferMemoryRequirements(buf);
  auto allocInfo = vk::MemoryAllocateInfo(memoryReqs.size, memory_id);
  return device.allocateMemory(allocInfo);
}

/// Create buffer on a device_. Does NOT allocate memory.
template<class T>
vk::Buffer Allocator2<T>::CreateBuffer(const vk::Device& device,
                                      uint32_t buffer_size, 
                                      vk::BufferUsageFlags usage) const {
  auto create_info = vk::BufferCreateInfo(vk::BufferCreateFlags(), buffer_size, usage);
  return device.createBuffer(create_info);
}

template class Allocator2<int>;
template class Allocator2<float>;
template class Allocator2<double>;

} // namespace vky
