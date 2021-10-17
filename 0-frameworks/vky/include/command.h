#ifndef VKY_COMMAND_H_
#define VKY_COMMAND_H_

#include <iostream>
#include <vulkan/vulkan.hpp>

#include "device.h"
#include "pipeline.h"

namespace vky {

class Command {

public:
  Command() {}
  ~Command() {}

  int Initialize(const vk::Device device, const uint32_t compute_queue_familly_id);
  void UnInitialize();

  void Begin();
  void End();

  void ComputeShader(Pipeline *pipeline, const uint32_t *group_count_xyz, const void *params, const int params_size);
  void CopyBuffer(const vk::Buffer& src, vk::Buffer& dst, const uint32_t size);

private:
  vk::Device device_;

  vk::CommandPool cmd_pool_;             // used to allocate command buffers
  vk::CommandBuffer cmd_buffer_;

  vk::Queue queue_;
  vk::Fence fence_;
}; // class Command

} // namespace vky

#endif  // VKY_COMMAND_H_