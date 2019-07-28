#include "command.h"

namespace vky {

///////////////
// <Public

int Command::Initialize(const vk::Device device, const uint32_t compute_queue_familly_id) {
  device_ = device;

  auto command_pool_create_info = vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlags(), compute_queue_familly_id);
  cmd_pool_ = device_.createCommandPool(command_pool_create_info);

  auto alloc_info = vk::CommandBufferAllocateInfo(cmd_pool_, vk::CommandBufferLevel::ePrimary, 1);
  cmd_buffer_ = device_.allocateCommandBuffers(alloc_info)[0];

  // 0 is the queue index in the family, by default just the first one is used
  queue_ = device_.getQueue(compute_queue_familly_id, 0);
  // fence makes sure the control is not returned to CPU till command buffer is depleted
  // create fence
  fence_ = device_.createFence(vk::FenceCreateInfo());

  return 0;
}

void Command::UnInitialize() { 
  device_.destroyFence(fence_);
  device_.freeCommandBuffers(cmd_pool_, 1, &cmd_buffer_);
  device_.destroyCommandPool(cmd_pool_);
}

void Command::Begin(CommandType type) {
  // CommandPool must be reset before CommandBuffer starts.
  device_.resetCommandPool(cmd_pool_, vk::CommandPoolResetFlags());
  if (type == COMPUTE) {
    // Fences must be reset before being submitted
    device_.resetFences(fence_);
  }
  // Start recording commands into the newly allocated command buffer.
  // buffer is only submitted and used once
  // auto begin_info = vk::CommandBufferBeginInfo();
  cmd_buffer_.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
}
void Command::End(CommandType type) {
  cmd_buffer_.end(); // end recording commands
                     // submit the command buffer to the queue and set up a fence.  
  auto submit_info = vk::SubmitInfo(0, nullptr, nullptr, 1, &cmd_buffer_); // submit a single command buffer
  if (type == COMPUTE) {
    queue_.submit({ submit_info }, fence_);
    device_.waitForFences({ fence_ }, true, uint64_t(-1));      // wait for the fence indefinitely
  }
  else {
    queue_.submit({ submit_info }, nullptr);
    queue_.waitIdle();
  }
}

void Command::Submit(Pipeline *pipeline, const uint32_t *group_count_xyz, const void *params, const int params_size) {
  Begin(COMPUTE);
  ComputeShader(pipeline, group_count_xyz, params, params_size);
  End(COMPUTE);
}
void Command::Submit(const vk::Buffer& src, vk::Buffer& dst, const uint32_t size) {
  Begin(COPY);
  CopyBuffer(src, dst, size);
  End(COPY);
}

///////////////
// <Private

void Command::ComputeShader(Pipeline *pipeline, const uint32_t *group_count_xyz, const void *params, const int params_size) {
  // Before dispatch bind a pipeline, AND a descriptor set.
  // The validation layer will NOT give warnings if you forget those.
  cmd_buffer_.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->pipeline());
  cmd_buffer_.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline->pipeline_layout(),
    0, { pipeline->descriptor_set() }, {});

  cmd_buffer_.pushConstants(pipeline->pipeline_layout(), vk::ShaderStageFlagBits::eCompute, 0, params_size, params);
  // Start the compute pipeline, and execute the compute shader.
  // The number of workgroups is specified in the arguments.
  cmd_buffer_.dispatch(group_count_xyz[0], group_count_xyz[1], group_count_xyz[2]);
}

void Command::CopyBuffer(const vk::Buffer& src, vk::Buffer& dst, const uint32_t size) {
  auto region = vk::BufferCopy(0, 0, size);
  cmd_buffer_.copyBuffer(src, dst, 1, &region);
}
} // namespace vky