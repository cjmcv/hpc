#ifndef VKY_EXECUTOR_H_
#define VKY_EXECUTOR_H_

#include <iostream>
#include <fstream>

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
  // TODO: Use string to get.
  vk::ShaderModule shader() const { return shader_; }

  int Initialize(bool is_enable_validation) {
    std::vector<const char*> layers = std::vector<const char*>{};
    std::vector<const char*> extensions = std::vector<const char*>{};
    if (is_enable_validation) {
      // "=" in vector is deep copy.
      layers = EnabledLayers({ "VK_LAYER_LUNARG_standard_validation" });
      extensions = EnabledExtensions({ VK_EXT_DEBUG_REPORT_EXTENSION_NAME });
    }

    CreateInstance(layers, extensions);
    SearchPhysicalDevices();
    CreateDevice();

    // TODO: move to other place.
    CreateShaderModule("src/shaders/saxpy.spv");

    return 0;
  }

  int CreateInstance(std::vector<const char*> &layers,
                     std::vector<const char*> &extensions) {

    auto app_info = vk::ApplicationInfo("Example Filter", 0, "no_engine",
      0, VK_API_VERSION_1_0); // The only important field here is apiVersion
    auto create_info = vk::InstanceCreateInfo(vk::InstanceCreateFlags(), &app_info,
      ARR_VIEW(layers), ARR_VIEW(extensions));

    instance_ = vk::createInstance(create_info);
    return 0;
  }

  int SearchPhysicalDevices() {
    vkEnumeratePhysicalDevices(instance_, &device_count_, NULL);
    if (device_count_ == 0) {
      throw std::runtime_error("could not find a device with vulkan support");
    }
    phys_devices_.resize(device_count_);
    instance_.enumeratePhysicalDevices(&device_count_, phys_devices_.data());

    return 0;
  }

  int CreateDevice(int device_id = 0) {
    compute_queue_familly_id_ = GetComputeQueueFamilyId(phys_devices_[device_id]);

    // create logical device_ to interact with the physical one
    // When creating the device specify what queues it has
    // TODO: when physical device is a discrete gpu, transfer queue needs to be included
    auto p = float(1.0); // queue priority
    auto queue_create_info = vk::DeviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), compute_queue_familly_id_, 1, &p);
    auto device_create_info = vk::DeviceCreateInfo(vk::DeviceCreateFlags(), 1, &queue_create_info, 0, nullptr);//ARR_VIEW(layers_)
    device_ = phys_devices_[device_id].createDevice(device_create_info, nullptr);
  }

  int CreateShaderModule(const char* filename) {
    // Read binary shader_ file into array of uint32_t. little endian assumed.
    auto fin = std::ifstream(filename, std::ios::binary);
    if (!fin.is_open()) {
      throw std::runtime_error(std::string("could not open file ") + filename);
    }
    auto code = std::vector<char>(std::istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>());
    // Padded by 0s to a boundary of 4.
    code.resize(4 * div_up(code.size(), size_t(4)));

    vk::ShaderModuleCreateFlags flags = vk::ShaderModuleCreateFlags();
    auto shader_module_create_info = vk::ShaderModuleCreateInfo(flags, code.size(),
      reinterpret_cast<uint32_t*>(code.data()));

    shader_ = device_.createShaderModule(shader_module_create_info);

    return 0;
  }

public: //private
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
  /// Groups of queues that have the same capabilities (for instance_, they all supports graphics
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


public:  //private
  uint32_t device_id_;
  uint32_t compute_queue_familly_id_;
  vk::Instance instance_;

  std::vector<vk::PhysicalDevice> phys_devices_;
  uint32_t device_count_;
  // TODO: list for multi-devices.
  vk::Device device_;               // logical device providing access to a physical one     
  // TODO: list for multi-shaders.
  vk::ShaderModule shader_;

}; // class Device

class Command {
public:

  //, uint32_t queue_index
  Command() {

  }

  ~Command() {

  }

  int Initialize(const DeviceManager* device_manager) {
    vk::DescriptorPoolSize descriptor_pool_size =
      vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, NumDescriptors);
    vk::DescriptorPoolCreateInfo descriptor_pool_create_info =
      vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(), 1, 1, &descriptor_pool_size);
    dsc_pool_ = device_manager->device().createDescriptorPool(descriptor_pool_create_info);

    auto command_pool = vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlags(), device_manager->compute_queue_familly_id_);
    cmd_pool_ = device_manager->device().createCommandPool(command_pool);

  }

private:
  mutable vk::DescriptorPool dsc_pool_;  // descriptors pool
  vk::CommandPool cmd_pool_;             // used to allocate command buffers


}; // class Command

// C++ mirror of the shader push constants interface
struct PushParams {
  uint32_t width;  //< frame width
  uint32_t height; //< frame height
  float a;         //< saxpy (\$ y = y + ax \$) scaling factor
};

class Pipeline {

public:


  Pipeline(const DeviceManager* device_manager) {
    local_shader_module_ = nullptr;
    descriptorset_layout_ = nullptr;
    pipeline_layout_ = nullptr;
    pipeline_ = nullptr;

    devm_ = device_manager;
  }

  int CreateDescriptorsetLayout() {
    auto bind_layout = std::array<vk::DescriptorSetLayoutBinding, NumDescriptors>{ {
      {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}
      , { 1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute }
      }};
    auto create_info = vk::DescriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags()
      , ARR_VIEW(bind_layout));
    descriptorset_layout_ = devm_->device().createDescriptorSetLayout(create_info);

    return 0;
  }

  int CreatePipelineLayout() {
    auto push_const_range = vk::PushConstantRange(vk::ShaderStageFlagBits::eCompute,
      0, sizeof(PushParams));
    auto pipeline_layoutCI = vk::PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(),
      1, &descriptorset_layout_, 1, &push_const_range);
    pipeline_layout_ = devm_->device().createPipelineLayout(pipeline_layoutCI);

    return 0;
  }

  // TODO: std::string shader_name 
  // Create compute pipeline consisting of a single stage with compute shader_.
  // Specialization constants specialized here.
  int CreatePipeline() {
    pipe_cache_ = devm_->device().createPipelineCache(vk::PipelineCacheCreateInfo());

    // specialize constants of the shader_
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

private:
  const DeviceManager* devm_;

  vk::ShaderModule local_shader_module_; 
  vk::DescriptorSetLayout descriptorset_layout_; // c++ definition of the shader binding interface
  vk::PipelineLayout pipeline_layout_;

  vk::PipelineCache pipe_cache_;
  vk::Pipeline pipeline_;

}; // class Pipeline

class Operator {
public:
  int Initialize() {

  }

private:
  std::vector<Pipeline *> pipes_;

}; // class Operator

class Executor {

  Executor() {

  }
  ~Executor() {

  }

  int Initialize() {
    // Init Device.
    devm_ = new DeviceManager();
    devm_->Initialize(true);
    // Init command.
    comd_ = new Command();
    comd_->Initialize(devm_);

    // Init Operators.

  }

  int Run(vk::Buffer& out, const vk::Buffer& in, const PushParams& p) {
    // BindParameters
  }

private:
  Command *comd_;
  std::vector<Operator *> ops_;

  DeviceManager *devm_;

}; // class Executor

} // namespace vky

#endif  // VKY_EXECUTOR_H_