#include "allocator.h"

#include <iostream>

namespace vky {

/////////////
//  Public.
/////////////
BufferMemory* Allocator::Malloc(size_t size, 
  vk::BufferUsageFlags usage,
  vk::MemoryPropertyFlags properties) {

  BufferMemory *bm = new BufferMemory();
  bm->usage_ = usage;
  bm->properties_ = properties;

  bm->buffer_range_ = size;
  bm->buffer_ = CreateBuffer(size, usage, properties);
  uint32_t memory_id = SelectMemory(bm->buffer_, properties);
  bm->memory_ = AllocateMemory(bm->buffer_, memory_id);

  device_.bindBufferMemory(bm->buffer_, bm->memory_, 0);

  bm->mapped_ptr_ = nullptr;
  if (properties == vk::MemoryPropertyFlagBits::eHostVisible)
    bm->mapped_ptr_ = (float *)device_.mapMemory(bm->memory_, 0, size);

  return bm;
}

void Allocator::Free(BufferMemory *bm) {
  if (bm->mapped_ptr_ != nullptr)
    device_.unmapMemory(bm->memory_);
  device_.destroyBuffer(bm->buffer_, 0);
  device_.freeMemory(bm->memory_, 0);

  delete bm;
}

BufferMemory *Allocator::GetInBufferMemory(int size) {
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
BufferMemory *Allocator::GetOutBufferMemory(int size) {
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


/////////////
//  Private.
void Allocator::ClearIOBufferMemory() {
  std::map<int, BufferMemory*>::iterator it;
  for (it = staging_in_datas_.begin(); it != staging_in_datas_.end(); ++it)
    Free(it->second);

  for (it = staging_out_datas_.begin(); it != staging_out_datas_.end(); ++it)
    Free(it->second);
}

vk::Buffer Allocator::CreateBuffer(uint32_t buffer_size,
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

vk::DeviceMemory Allocator::AllocateMemory(const vk::Buffer& buf, uint32_t memory_type_index) {

  auto memory_reqs = device_.getBufferMemoryRequirements(buf);

  vk::MemoryAllocateInfo alloc_info;
  alloc_info.setAllocationSize(memory_reqs.size);
  alloc_info.setMemoryTypeIndex(memory_type_index);

  return device_.allocateMemory(alloc_info);
}

// TODO: test.
vk::DeviceMemory Allocator::AllocateDedicatedMemory(size_t size, uint32_t memory_type_index, vk::Buffer buffer) {
  vk::MemoryAllocateInfo alloc_info;
  alloc_info.setAllocationSize(size);
  alloc_info.setMemoryTypeIndex(memory_type_index);

  vk::MemoryDedicatedAllocateInfoKHR dedicated_alloc_info;
  dedicated_alloc_info.setPNext(0);
  dedicated_alloc_info.setBuffer(buffer);

  alloc_info.setPNext(&dedicated_alloc_info);

  return device_.allocateMemory(alloc_info);
}

uint32_t Allocator::SelectMemory(const vk::Buffer& buffer, vk::MemoryPropertyFlags properties) const {
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

} // namespace vky
