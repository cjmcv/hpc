#ifndef VKY_DATA_TYPE_H_
#define VKY_DATA_TYPE_H_

#include <vulkan/vulkan.hpp>

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

} // namespace vky

#endif  // VKY_DATA_TYPE_H_