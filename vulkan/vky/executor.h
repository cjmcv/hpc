#ifndef VKY_EXECUTOR_H_
#define VKY_EXECUTOR_H_

#include <iostream>
#include <fstream>
#include <map>

#include <vulkan/vulkan.hpp>

#include "device.h"
#include "allocator.h"

namespace vky {

inline uint32_t div_up(uint32_t x, uint32_t y) { return (x + y - 1u) / y; }

// TODO: Handle exeception.
// TODO: Release resource.
// TODO: Specify a transfer pipeline for buffer copy.

class Pipeline {

public:
  Pipeline() {
    num_descriptors_ = 0;
    push_constant_count_ = 0;
    local_shader_module_ = nullptr;
    descriptor_set_layout_ = nullptr;
    pipeline_layout_ = nullptr;
    pipeline_ = nullptr;
  }

  vk::PipelineLayout pipeline_layout() const{ return pipeline_layout_; }
  vk::Pipeline pipeline() const { return pipeline_; }

  vk::DescriptorSet descriptor_set() const { return descriptor_set_; }

  int Initialize(const vk::Device device, const vk::ShaderModule shader,
    const int buffer_count, const int push_constant_count) {

    device_ = device;
    num_descriptors_ = buffer_count;
    push_constant_count_ = push_constant_count;

    CreateDescriptorsetLayout();  // num_descriptors_
    CreatePipelineLayout();       // push_constant_count_
    CreatePipeline(shader);
    return 0;
  }

  int CreateDescriptorsetLayout() {
    std::vector<vk::DescriptorSetLayoutBinding> layout_binding(num_descriptors_);
    for (int i = 0; i < num_descriptors_; i++) {
      layout_binding[i].setBinding(i);
      layout_binding[i].setDescriptorType(vk::DescriptorType::eStorageBuffer);
      layout_binding[i].setDescriptorCount(1);
      layout_binding[i].setStageFlags(vk::ShaderStageFlagBits::eCompute);
      layout_binding[i].setPImmutableSamplers(0);
    }
      
    vk::DescriptorSetLayoutCreateInfo create_info = vk::DescriptorSetLayoutCreateInfo(
      vk::DescriptorSetLayoutCreateFlags(), num_descriptors_, layout_binding.data());
      
    descriptor_set_layout_ = device_.createDescriptorSetLayout(create_info);

    return 0;
  }

  int CreatePipelineLayout() {

    auto push_const_range = vk::PushConstantRange(vk::ShaderStageFlagBits::eCompute,
      0, sizeof(int) * push_constant_count_);
    auto pipe_layout_create_info = vk::PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(),
      1, &descriptor_set_layout_, 1, &push_const_range);
    pipeline_layout_ = device_.createPipelineLayout(pipe_layout_create_info);

    return 0;
  }

  // TODO: Use it in funciton CreatePipeline.
  void SetOptimalLocalSizeXYZ(DeviceInfo &info, int width, int height, int channel) {
    uint32_t local_size_x = 0;
    uint32_t local_size_y = 0;
    uint32_t local_size_z = 0;

    if (channel > 0) {
      local_size_z = info.max_workgroup_size_[2];
      while ((uint32_t)channel < local_size_z) {
        local_size_z /= 2;
      }
    }
    else {
      local_size_z = std::min((uint32_t)128, info.max_workgroup_size_[2]);
    }

    uint32_t max_local_size_xy = info.max_workgroup_invocations_ / local_size_z;

    if (height == width || (height < 0 && width < 0)) {
      uint32_t local_size_xy = std::sqrt(max_local_size_xy);
      uint32_t local_size_xy_prefer = 128;
      while (local_size_xy < local_size_xy_prefer) {
        local_size_xy_prefer /= 2;
      }
      local_size_x = local_size_xy_prefer;
      local_size_y = local_size_xy_prefer;
    }

    if (height > 0 && width > 0) {
      if (height > width) {
        float ps = height / (float)width;
        float local_size_xy = sqrt(max_local_size_xy / ps);
        local_size_y = local_size_xy * ps;
        local_size_x = std::max((uint32_t)local_size_xy, (uint32_t)1);
      }
      else {
        float ps = width / (float)height;
        float local_size_xy = sqrt(max_local_size_xy / ps);
        local_size_y = std::max((uint32_t)local_size_xy, (uint32_t)1);
        local_size_x = local_size_xy * ps;
      }

      uint32_t local_size_y_prefer = std::min((uint32_t)128, info.max_workgroup_size_[1]);
      while (local_size_y < local_size_y_prefer) {
        local_size_y_prefer /= 2;
      }

      uint32_t local_size_x_prefer = std::min((uint32_t)128, info.max_workgroup_size_[0]);
      while (local_size_x < local_size_x_prefer)
      {
        local_size_x_prefer /= 2;
      }

      local_size_y = local_size_y_prefer;
      local_size_x = local_size_x_prefer;
    }
    else if (height > 0) {
      local_size_y = std::min(max_local_size_xy, info.max_workgroup_size_[1]);
      while ((uint32_t)height < local_size_y) {
        local_size_y /= 2;
      }

      uint32_t max_local_size_x = max_local_size_xy / local_size_y;
      local_size_x = std::min(max_local_size_x, info.max_workgroup_size_[0]);
    }
    else if (width > 0) {
      local_size_x = std::min(max_local_size_xy, info.max_workgroup_size_[0]);
      while ((uint32_t)width < local_size_x) {
        local_size_x /= 2;
      }

      uint32_t max_local_size_y = max_local_size_xy / local_size_x;
      local_size_y = std::min(max_local_size_y, info.max_workgroup_size_[1]);
    }
  }

  // Create compute pipeline consisting of a single stage with compute shader.
  // Specialization constants specialized here.
  int CreatePipeline(const vk::ShaderModule shader) {
    pipe_cache_ = device_.createPipelineCache(vk::PipelineCacheCreateInfo());

    // specialize constants of the shader
    // {constantID, offset, size}
    // 3: local_size_x, local_size_y, local_size_z
    auto spec_entries = std::array<vk::SpecializationMapEntry, 3>{
      { { 0, 0, sizeof(int) }, 
        { 1, 1 * sizeof(int), sizeof(int) },
        { 2, 2 * sizeof(int), sizeof(int) } }
    };
    // TODO: Replace it by SetOptimalLocalSizeXYZ.
    int WORKGROUP_SIZE = 16;
    auto spec_values = std::array<int, 3>{WORKGROUP_SIZE, WORKGROUP_SIZE, 1};
    auto spec_info = vk::SpecializationInfo(spec_entries.size(), spec_entries.data(),
      spec_values.size() * sizeof(int), spec_values.data());   // TODO: Change sizeof(int) to a manual type?

    // Specify the compute shader stage, and it's entry point (main), and specializations
    auto stage_create_info = vk::PipelineShaderStageCreateInfo(
      vk::PipelineShaderStageCreateFlags(),
      vk::ShaderStageFlagBits::eCompute, 
      shader, 
      "main",
      &spec_info);

    auto pipeline_create_info = vk::ComputePipelineCreateInfo(
      vk::PipelineCreateFlags(),
      stage_create_info, 
      pipeline_layout_);

    pipeline_ = device_.createComputePipeline(pipe_cache_, pipeline_create_info, nullptr);

    return 0;
  }

  int CreateDescriptorPool() {
    vk::DescriptorPoolSize descriptor_pool_size =
      vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, num_descriptors_);
    vk::DescriptorPoolCreateInfo descriptor_pool_create_info =
      vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(), 1, 1, &descriptor_pool_size);
    descriptor_pool_ = device_.createDescriptorPool(descriptor_pool_create_info);

    return 0;
  }

  int ReleaseDescriptorPool() {
    device_.destroyDescriptorPool(descriptor_pool_);
    return 0;
  }

  int AllocateDescriptorSet() {
    ///Create descriptor set.Actually associate buffers to binding points in bindLayout.
    /// Buffer sizes are specified here as well.
    vk::DescriptorSetAllocateInfo allocate_info = 
      vk::DescriptorSetAllocateInfo(descriptor_pool_, 1, &descriptor_set_layout_);

    descriptor_set_ = device_.allocateDescriptorSets(allocate_info)[0];
    return 0;
  }

  // TODO: Now, the size of the buffers has to be the same .
  int UpdateDescriptorSet(const std::vector<vk::Buffer> &buffers, const int size) {
    if (buffers.size() != num_descriptors_) {
      throw std::runtime_error("UpdateDescriptorSet -> buffers.size() != num_descriptors_");
    }

    std::vector<vk::DescriptorBufferInfo> buffers_info(num_descriptors_);
    for (int i = 0; i < num_descriptors_; i++) {
      buffers_info[i].setBuffer(buffers[i]);
      buffers_info[i].setOffset(0);
      buffers_info[i].setRange(size);
    }

    std::vector<vk::WriteDescriptorSet> write_descriptor_sets(num_descriptors_);
    for (int i = 0; i < num_descriptors_; i++) {
      write_descriptor_sets[i].setDstSet(descriptor_set_);
      write_descriptor_sets[i].setDstBinding(i);
      write_descriptor_sets[i].setDstArrayElement(0);
      write_descriptor_sets[i].setDescriptorCount(1);
      write_descriptor_sets[i].setDescriptorType(vk::DescriptorType::eStorageBuffer);
      write_descriptor_sets[i].setPImageInfo(nullptr);

      write_descriptor_sets[i].setPBufferInfo(&(buffers_info[i]));
    }

    device_.updateDescriptorSets(write_descriptor_sets, {});
    return 0;
  }

private:
  vk::Device device_;

  vk::ShaderModule local_shader_module_; 
  vk::DescriptorSetLayout descriptor_set_layout_; // channel++ definition of the shader binding interface
  vk::DescriptorPool descriptor_pool_;  // descriptors pool
  vk::DescriptorSet descriptor_set_; 

  vk::PipelineLayout pipeline_layout_;
  vk::PipelineCache pipe_cache_;
  vk::Pipeline pipeline_;

  int num_descriptors_;
  int push_constant_count_;
}; // class Pipeline

class Command {
public:

  Command() {}
  ~Command() {}

  int Initialize(const vk::Device device, const uint32_t compute_queue_familly_id) {
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

  int Reset() {
    // CommandPool must be reset before CommandBuffer starts.
    device_.resetCommandPool(cmd_pool_, vk::CommandPoolResetFlags());
    // Fences must be reset before being submitted
    device_.resetFences(fence_);
    return 0;
  }

  void Begin() {
    // Start recording commands into the newly allocated command buffer.
    // buffer is only submitted and used once
    auto begin_info = vk::CommandBufferBeginInfo();
    cmd_buffer_.begin(begin_info);
  }

  void End() {
    cmd_buffer_.end(); // end recording commands
    assert(cmd_buffer_ != vk::CommandBuffer{}); // TODO: this should be a check for a valid command buffer
  }

  void Bind(Pipeline *pipeline) {
    // Before dispatch bind a pipeline, AND a descriptor set.
    // The validation layer will NOT give warnings if you forget those.
    cmd_buffer_.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->pipeline());
    cmd_buffer_.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline->pipeline_layout(),
      0, { pipeline->descriptor_set() }, {});
  }

  void PushAndDispatch(Pipeline *pipeline, const int *group_count_xyz, const void *params, const int params_size) {
    cmd_buffer_.pushConstants(pipeline->pipeline_layout(), vk::ShaderStageFlagBits::eCompute, 0, params_size, params);
    // Start the compute pipeline, and execute the compute shader.
    // The number of workgroups is specified in the arguments.
    cmd_buffer_.dispatch(group_count_xyz[0], group_count_xyz[1], group_count_xyz[2]);
  }

  void Fences() {
    // submit the command buffer to the queue and set up a fence.  
    auto submit_info = vk::SubmitInfo(0, nullptr, nullptr, 1, &cmd_buffer_); // submit a single command buffer
    queue_.submit({ submit_info }, fence_);
    device_.waitForFences({ fence_ }, true, uint64_t(-1));      // wait for the fence indefinitely
  }

private:
  vk::Device device_;

  vk::CommandPool cmd_pool_;             // used to allocate command buffers
  vk::CommandBuffer cmd_buffer_;

  vk::Queue queue_;
  vk::Fence fence_;
}; // class Command

// TODO: Operator -> GeneralOp => One Pipeline, manual setup IO
//                -> CustomizedOp -> COp1 => 1.Pipeline and IO are fixed.
//                                -> COp2... 2.Can be designed externally.
class OperatorA {
public:
  int Initialize(const vk::Device device, const vk::ShaderModule shader) {
    pipes_ = new Pipeline();
    pipes_->Initialize(device, shader, 2, 3);
    return 0;
  }

  int Run(Command *command,
    const std::vector<vk::Buffer> &buffers,
    const int buffer_range, 
    const int *group_count_xyz,
    const void *params,
    const int params_size) {
    
    pipes_->CreateDescriptorPool();
    pipes_->AllocateDescriptorSet();
    pipes_->UpdateDescriptorSet(buffers, buffer_range);
    ////

    command->Reset();

    command->Begin();
    command->Bind(pipes_);
    command->PushAndDispatch(pipes_, group_count_xyz, params, params_size);
    command->End();

    command->Fences();

    // UnbindParameters
    pipes_->ReleaseDescriptorPool();

    return 0;
  }

private:
  // TODO: std::vector<Pipeline *> pipes_;
  Pipeline *pipes_;

}; // class Operator

class Executor {

public:

  Executor() {}
  ~Executor() {}

  const vk::Device &device() const { return device_; }
  Allocator *allocator() const { return allocator_; }

  int Initialize(const DeviceInfo &device_info, const std::string &shaders_dir_path) {
    // Init Device.
    device_ = CreateDevice(device_info.physical_device_, device_info.compute_queue_familly_id_);
    RegisterShaders(shaders_dir_path);

    // Init command.
    comd_ = new Command();
    comd_->Initialize(device_, device_info.compute_queue_familly_id_);

    // Init Operators.
    op_ = new OperatorA();
    op_->Initialize(device_, shader("saxpy"));

    allocator_ = new Allocator(device_, device_info.physical_device_, device_info.compute_queue_familly_id_);
    return 0;
  }

  int UnInitialize() {
    if (allocator_ != nullptr) {
      delete allocator_;
      allocator_ = nullptr;
    }

    return 0;
  }
  // TODO: Data, Bind the buffers and buffer_range together. 
  //       Create a new class for Buffer.
  int Run(const std::vector<vk::Buffer> &buffers,
    const int buffer_range,
    const int *group_count_xyz, 
    const void *params, 
    const int params_size) const {

    op_->Run(comd_, buffers, buffer_range, group_count_xyz, params, params_size);

    return 0;
  }

private:

  void RegisterShaders(const std::string &shaders_dir_path) {
    shaders_map_["add"] = CreateShaderModule(shaders_dir_path + "shaders/add.spv");
    shaders_map_["saxpy"] = CreateShaderModule(shaders_dir_path + "shaders/saxpy.spv");
  }

  vk::ShaderModule shader(const std::string &str) {
    std::map<std::string, vk::ShaderModule>::iterator it = shaders_map_.find(str);
    if (it == shaders_map_.end()) {
      throw std::runtime_error(std::string("could not find shader: ") + it->first);
    }
    return it->second;
  }

  vk::ShaderModule CreateShaderModule(const std::string &filename) {
    // Read binary shader file into array of uint32_t. little endian assumed.
    auto fin = std::ifstream(filename.c_str(), std::ios::binary);
    if (!fin.is_open()) {
      throw std::runtime_error(std::string("could not open file ") + filename.c_str());
    }
    auto code = std::vector<char>(std::istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>());
    // Padded by 0s to a boundary of 4.
    code.resize(4 * div_up(code.size(), size_t(4)));

    vk::ShaderModuleCreateFlags flags = vk::ShaderModuleCreateFlags();
    auto shader_module_create_info = vk::ShaderModuleCreateInfo(flags, code.size(),
      reinterpret_cast<uint32_t*>(code.data()));

    return device_.createShaderModule(shader_module_create_info);
  }

  vk::Device CreateDevice(const vk::PhysicalDevice &physical_device, const uint32_t compute_queue_familly_id) {
    // create logical device to interact with the physical one
    // When creating the device specify what queues it has
    // TODO: when physical device is a discrete gpu, transfer queue needs to be included
    float p = float(1.0); // queue priority
    vk::DeviceQueueCreateInfo queue_create_info =
      vk::DeviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), compute_queue_familly_id, 1, &p);
    vk::DeviceCreateInfo device_create_info = 
      vk::DeviceCreateInfo(vk::DeviceCreateFlags(), 1, &queue_create_info, 0, nullptr);//layers_.size(), layers_.data()

    return physical_device.createDevice(device_create_info, nullptr);
  }

private:  
  vk::Device device_;    // logical device providing access to a physical one 
  std::map<std::string, vk::ShaderModule> shaders_map_;

  Command *comd_;
  // TODO: GeneralOp and CustomizedOp.
  OperatorA *op_;

  Allocator *allocator_;
}; // class Executor

} // namespace vky

#endif  // VKY_EXECUTOR_H_