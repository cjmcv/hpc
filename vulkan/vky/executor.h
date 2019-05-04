#ifndef VKY_EXECUTOR_H_
#define VKY_EXECUTOR_H_

#include <iostream>
#include <fstream>
#include <map>

#include <vulkan/vulkan.hpp>

namespace vky {

#define ARR_VIEW(x) uint32_t(x.size()), x.data()
#define ST_VIEW(s)  uint32_t(sizeof(s)), &s
#define ALL(x) begin(x), end(x)

inline auto div_up(uint32_t x, uint32_t y) { return (x + y - 1u) / y; }

// TODO: Change to a variate. 
static constexpr auto NumDescriptors = uint32_t(2);
constexpr uint32_t WORKGROUP_SIZE = 16; ///< compute shader workgroup dimension is WORKGROUP_SIZE x WORKGROUP_SIZE

class DeviceManager {
public:
  DeviceManager() {}
  ~DeviceManager() {}

  int device_count() const { return device_count_; }

  vk::Device device() const { return device_; }
  vk::PhysicalDevice phys_device() const { return phys_devices_[0]; } // TODO

  // TODO: Use string to get.
  vk::ShaderModule shader(std::string str = "saxpy") const { return shaders_map_.find(str)->second; }
  uint32_t compute_queue_familly_id() const { return compute_queue_familly_id_; }

  int Initialize(bool is_enable_validation);

  int CreateInstance(std::vector<const char*> &layers,
    std::vector<const char*> &extensions);

  int SearchPhysicalDevices();

  int CreateDevice(int device_id = 0);

private:

  vk::ShaderModule CreateShaderModule(const char* filename);

  /// filter list of desired extensions to include only those supported by current Vulkan instance.
  std::vector<const char*> EnabledExtensions(const std::vector<const char*>& extensions) const {
    auto ret = std::vector<const char*>{};
    auto instanceExtensions = vk::enumerateInstanceExtensionProperties();
    for (auto e : extensions) {
      auto it = std::find_if(ALL(instanceExtensions)
        , [=](auto& p) { return strcmp(p.extensionName, e); });
      if (it != end(instanceExtensions)) {
        ret.push_back(e);
      }
      else {
        std::cerr << "[WARNING]: extension " << e << " is not found" "\n";
      }
    }
    return ret;
  }

  /// filter list of desired extensions to include only those supported by current Vulkan instance_
  std::vector<const char*> EnabledLayers(const std::vector<const char*>& layers) const {
    auto ret = std::vector<const char*>{};
    auto instanceLayers = vk::enumerateInstanceLayerProperties();
    for (auto l : layers) {
      auto it = std::find_if(ALL(instanceLayers)
        , [=](auto& p) { return strcmp(p.layerName, l); });
      if (it != end(instanceLayers)) {
        ret.push_back(l);
      }
      else {
        std::cerr << "[WARNING] layer " << l << " is not found" "\n";
      }
    }
    return ret;
  }

  /// @return the index of a queue family that supports compute operations.
  /// Groups of queues that have the same capabilities (for instance, they all supports graphics
  /// and computer operations), are grouped into queue families.
  /// When submitting a command buffer, you must specify to which queue in the family you are submitting to.
  uint32_t GetComputeQueueFamilyId(const vk::PhysicalDevice& physical_device) const {
    auto queue_families = physical_device.getQueueFamilyProperties();

    // prefer using compute-only queue
    auto queue_it = std::find_if(ALL(queue_families), [](auto& f) {
      auto maskedFlags = ~vk::QueueFlagBits::eSparseBinding & f.queueFlags; // ignore sparse binding flag 
      return 0 < f.queueCount                                               // queue family does have some queues in it
        && (vk::QueueFlagBits::eCompute & maskedFlags)
        && !(vk::QueueFlagBits::eGraphics & maskedFlags);
    });
    if (queue_it != end(queue_families)) {
      return uint32_t(std::distance(begin(queue_families), queue_it));
    }

    // otherwise use any queue that has compute flag set
    queue_it = std::find_if(ALL(queue_families), [](auto& f) {
      auto maskedFlags = ~vk::QueueFlagBits::eSparseBinding & f.queueFlags;
      return 0 < f.queueCount && (vk::QueueFlagBits::eCompute & maskedFlags);
    });
    if (queue_it != end(queue_families)) {
      return uint32_t(std::distance(begin(queue_families), queue_it));
    }

    throw std::runtime_error("could not find a queue family that supports compute operations");
  }


private:
  uint32_t device_id_;
  uint32_t compute_queue_familly_id_;
  vk::Instance instance_;

  std::vector<vk::PhysicalDevice> phys_devices_;
  uint32_t device_count_;
  // TODO: list for multi-devices.
  vk::Device device_;               // logical device providing access to a physical one     
  // TODO: list for multi-shaders.
  // Use string to search it.
  //vk::ShaderModule shader_;
  std::map<std::string, vk::ShaderModule> shaders_map_;

}; // class Device

// TODO: Remove it.
// C++ mirror of the shader push constants interface
struct PushParams {
  uint32_t width;  //< frame width
  uint32_t height; //< frame height
  float a;         //< saxpy (\$ y = y + ax \$) scaling factor
};

class Pipeline {

public:
  Pipeline(const DeviceManager* device_manager) {  
    devm_ = device_manager;

    local_shader_module_ = nullptr;
    descriptor_set_layout_ = nullptr;
    pipeline_layout_ = nullptr;
    pipeline_ = nullptr;
  }

  vk::PipelineLayout pipeline_layout() const{ return pipeline_layout_; }
  vk::Pipeline pipeline() const { return pipeline_; }

  vk::DescriptorSet descriptor_set() const { return descriptor_set_; }
  vk::DescriptorPool descriptor_pool() const { return descriptor_pool_; }

  int Initialize() {
    CreateDescriptorsetLayout();
    CreatePipelineLayout();
    CreatePipeline();
    return 0;
  }

  int CreateDescriptorsetLayout() {
    auto bind_layout = std::array<vk::DescriptorSetLayoutBinding, NumDescriptors>{ {
      { 0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute }, 
      { 1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute }
      }};
    auto create_info = vk::DescriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags()
      , ARR_VIEW(bind_layout));
    descriptor_set_layout_ = devm_->device().createDescriptorSetLayout(create_info);

    return 0;
  }

  int CreatePipelineLayout() {
    auto push_const_range = vk::PushConstantRange(vk::ShaderStageFlagBits::eCompute,
      0, sizeof(PushParams));
    auto pipe_layout_create_info = vk::PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(),
      1, &descriptor_set_layout_, 1, &push_const_range);
    pipeline_layout_ = devm_->device().createPipelineLayout(pipe_layout_create_info);

    return 0;
  }

  // TODO: std::string shader_name 
  // Create compute pipeline consisting of a single stage with compute shader.
  // Specialization constants specialized here.
  int CreatePipeline() {
    pipe_cache_ = devm_->device().createPipelineCache(vk::PipelineCacheCreateInfo());

    // specialize constants of the shader
    auto specEntries = std::array<vk::SpecializationMapEntry, 2>{
      { {0, 0, sizeof(int)}, { 1, 1 * sizeof(int), sizeof(int) }}
    };
    auto spec_values = std::array<int, 2>{WORKGROUP_SIZE, WORKGROUP_SIZE};
    auto spec_info = vk::SpecializationInfo(ARR_VIEW(specEntries),
      spec_values.size() * sizeof(int), spec_values.data());

    // Specify the compute shader stage, and it's entry point (main), and specializations
    auto stage_create_info = vk::PipelineShaderStageCreateInfo(
      vk::PipelineShaderStageCreateFlags(),
      vk::ShaderStageFlagBits::eCompute,
      devm_->shader(), "main", &spec_info);
    auto pipeline_create_info = vk::ComputePipelineCreateInfo(
      vk::PipelineCreateFlags(),
      stage_create_info, pipeline_layout_);

    pipeline_ = devm_->device().createComputePipeline(pipe_cache_, pipeline_create_info, nullptr);

    return 0;
  }

  int CreateDescriptorPool() {
    vk::DescriptorPoolSize descriptor_pool_size =
      vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, NumDescriptors);
    vk::DescriptorPoolCreateInfo descriptor_pool_create_info =
      vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(), 1, 1, &descriptor_pool_size);
    descriptor_pool_ = devm_->device().createDescriptorPool(descriptor_pool_create_info);

    return 0;
  }
  int AllocateDescriptorSet() {
    ///Create descriptor set.Actually associate buffers to binding points in bindLayout.
    /// Buffer sizes are specified here as well.
    auto descriptorSetAI = vk::DescriptorSetAllocateInfo(descriptor_pool_, 1, &descriptor_set_layout_);
    descriptor_set_ = devm_->device().allocateDescriptorSets(descriptorSetAI)[0];

    return 0;
  }

private:
  const DeviceManager* devm_;

  vk::ShaderModule local_shader_module_; 
  vk::DescriptorSetLayout descriptor_set_layout_; // c++ definition of the shader binding interface
  vk::DescriptorPool descriptor_pool_;  // descriptors pool
  vk::DescriptorSet descriptor_set_; 

  vk::PipelineLayout pipeline_layout_;
  vk::PipelineCache pipe_cache_;
  vk::Pipeline pipeline_;

}; // class Pipeline

class Command {
public:

  //, uint32_t queue_index
  Command() {}

  ~Command() {}

  vk::CommandPool cmd_pool() const { return cmd_pool_; }

  int Initialize(const DeviceManager* device_manager) {
    devm_ = device_manager;

    auto command_pool_create_info = vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlags(), devm_->compute_queue_familly_id());
    cmd_pool_ = devm_->device().createCommandPool(command_pool_create_info);

    auto alloc_info = vk::CommandBufferAllocateInfo(cmd_pool_, vk::CommandBufferLevel::ePrimary, 1);
    cmd_buffer_ = devm_->device().allocateCommandBuffers(alloc_info)[0];

    return 0;
  }

  vk::SubmitInfo Begin(Pipeline *pipeline, const PushParams& p) {
    // Start recording commands into the newly allocated command buffer.
    // buffer is only submitted and used once
    auto begin_info = vk::CommandBufferBeginInfo();
    cmd_buffer_.begin(begin_info);

    // Before dispatch bind a pipeline, AND a descriptor set.
    // The validation layer will NOT give warnings if you forget those.
    cmd_buffer_.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->pipeline());
    cmd_buffer_.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline->pipeline_layout(),
      0, { pipeline->descriptor_set() }, {});

    cmd_buffer_.pushConstants(pipeline->pipeline_layout(), vk::ShaderStageFlagBits::eCompute, 0, ST_VIEW(p));

    // Start the compute pipeline, and execute the compute shader.
    // The number of workgroups is specified in the arguments.
    cmd_buffer_.dispatch(div_up(p.width, WORKGROUP_SIZE), div_up(p.height, WORKGROUP_SIZE), 1);
    cmd_buffer_.end(); // end recording commands

    assert(cmd_buffer_ != vk::CommandBuffer{}); // TODO: this should be a check for a valid command buffer
    auto submit_info = vk::SubmitInfo(0, nullptr, nullptr, 1, &cmd_buffer_); // submit a single command buffer

    return submit_info;
  }

private:
  const DeviceManager* devm_;

  vk::CommandPool cmd_pool_;             // used to allocate command buffers
  vk::CommandBuffer cmd_buffer_;

}; // class Command

// TODO: Operator -> OperatorA
//                -> OperatorB
class OperatorA {
public:
  int Initialize(const DeviceManager* device_manager, Command *comd) {
    devm_ = device_manager;
    comd_ = comd;

    pipes_ = new Pipeline(device_manager);
    pipes_->Initialize();
    return 0;
  }

  int Run(const vk::Buffer& in, const PushParams& p, vk::Buffer& out) {
    
    pipes_->CreateDescriptorPool();

    pipes_->AllocateDescriptorSet();
    // TODO: Move ?
    uint32_t size = p.width*p.height;
    auto out_info = vk::DescriptorBufferInfo(out, 0, sizeof(float)*size);
    auto in_info = vk::DescriptorBufferInfo(in, 0, sizeof(float)*size);

    auto writeDsSets = std::array<vk::WriteDescriptorSet, NumDescriptors>{ {
      { pipes_->descriptor_set(), 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &out_info },
      { pipes_->descriptor_set(), 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &in_info }
      } };
    devm_->device().updateDescriptorSets(writeDsSets, {});

    ////

    vk::SubmitInfo submit_info = comd_->Begin(pipes_, p);
                                                                             // submit the command buffer to the queue and set up a fence.
    auto queue = devm_->device().getQueue(devm_->compute_queue_familly_id(), 0); // 0 is the queue index in the family, by default just the first one is used
    auto fence = devm_->device().createFence(vk::FenceCreateInfo()); // fence makes sure the control is not returned to CPU till command buffer is depleted
    queue.submit({ submit_info }, fence);
    devm_->device().waitForFences({ fence }, true, uint64_t(-1));      // wait for the fence indefinitely
    devm_->device().destroyFence(fence);

    // UnbindParameters
    devm_->device().destroyDescriptorPool(pipes_->descriptor_pool());
    devm_->device().resetCommandPool(comd_->cmd_pool(), vk::CommandPoolResetFlags());

    return 0;
  }

private:
  const DeviceManager* devm_;
  Command* comd_;
  // TODO: std::vector<Pipeline *> pipes_;
  Pipeline *pipes_;

}; // class Operator

class Executor {

public:

  Executor() {}
  ~Executor() {}

  // TODO. move ?
  vk::Device device() const { return devm_->device(); }
  vk::PhysicalDevice phys_device() const { return devm_->phys_device(); } // TODO

  int Initialize() {
    // Init Device.
    devm_ = new DeviceManager();
    devm_->Initialize(true);
    // Init command.
    comd_ = new Command();
    comd_->Initialize(devm_);
    // Init Operators.
    ops_ = new OperatorA();
    ops_->Initialize(devm_, comd_);

    return 0;
  }

  int Run(const vk::Buffer& in, const PushParams& p, vk::Buffer& out) const {
    ops_->Run(in, p, out);

    return 0;
  }

private: 
  DeviceManager *devm_;
  Command *comd_;

  // TODO: std::vector<OperatorA *> ops_;
  OperatorA *ops_;

}; // class Executor

} // namespace vky

#endif  // VKY_EXECUTOR_H_