#include "executor.hpp"

#include "allocator.hpp"

#include <vulkan/vulkan.hpp>

#define ARR_VIEW(x) uint32_t(x.size()), x.data()
#define ST_VIEW(s)  uint32_t(sizeof(s)), &s
#define ALL(x) begin(x), end(x)

VKAPI_ATTR VkBool32 VKAPI_CALL DebugReporter(
      VkDebugReportFlagsEXT, VkDebugReportObjectTypeEXT,
      uint64_t, size_t, int32_t,
      const char*    layer_prefix,
      const char*    message,
      void*          /*pUserData*/){
   std::cerr << "[WARNING]: Vulkan says: " << layer_prefix << ": " << message << "\n";
   return VK_FALSE;
}

using namespace vuh;
namespace {
	constexpr uint32_t WORKGROUP_SIZE = 16; ///< compute shader workgroup dimension is WORKGROUP_SIZE x WORKGROUP_SIZE

#ifdef NDEBUG
	constexpr bool isEnableValidation = false;
#else
	constexpr bool isEnableValidation = true;
#endif
} // namespace

inline auto div_up(uint32_t x, uint32_t y) { return (x + y - 1u) / y; }

/// Constructor
Executor::Executor(const std::string& shader_path){

	instance_ = CreateInstance();

	DebugReportCallback_ = isEnableValidation ?
                        RegisterValidationReporter(instance_, DebugReporter) : nullptr;

  vkEnumeratePhysicalDevices(instance_, &device_count_, NULL);
  if (device_count_ == 0) {
    throw std::runtime_error("could not find a device with vulkan support");
  }
  phys_devices_.resize(device_count_);
  instance_.enumeratePhysicalDevices(&device_count_, phys_devices_.data());

	compute_queue_familly_id_ = GetComputeQueueFamilyId(phys_devices_[0]);

  /// create logical device_ to interact with the physical one

  // When creating the device specify what queues it has
  // TODO: when physical device is a discrete gpu, transfer queue needs to be included
  auto p = float(1.0); // queue priority
  auto queue_create_info = vk::DeviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), compute_queue_familly_id_, 1, &p);
  auto device_create_info = vk::DeviceCreateInfo(vk::DeviceCreateFlags(), 1, &queue_create_info, ARR_VIEW(layers_));
  device_ = phys_devices_[0].createDevice(device_create_info, nullptr);

	auto command_pool = vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlags(), compute_queue_familly_id_);
	cmd_pool_ = device_.createCommandPool(command_pool);
	dsc_pool_ = AllocDescriptorPool(device_);

 	cmd_buffer_ = vk::CommandBuffer{};

  dsc_layout_ = CreateDescriptorSetLayout(device_);
	pipe_layout_ = CreatePipelineLayout(device_, dsc_layout_);
  pipe_cache_ = device_.createPipelineCache(vk::PipelineCacheCreateInfo());

  shader_ = LoadShader(device_, shader_path.c_str());
	pipe_ = CreateComputePipeline(device_, shader_, pipe_layout_, pipe_cache_);
}

/// Destructor
Executor::~Executor() noexcept {
	device_.destroyPipeline(pipe_);
	device_.destroyPipelineLayout(pipe_layout_);
	device_.destroyPipelineCache(pipe_cache_);
	device_.destroyCommandPool(cmd_pool_);
	device_.destroyDescriptorPool(dsc_pool_);
	device_.destroyDescriptorSetLayout(dsc_layout_);
	device_.destroyShaderModule(shader_);
	device_.destroy();

	if(DebugReportCallback_){
		// unregister callback.
		auto destroyFn = PFN_vkDestroyDebugReportCallbackEXT(
					vkGetInstanceProcAddr(instance_, "vkDestroyDebugReportCallbackEXT"));
		if(destroyFn){
			destroyFn(instance_, DebugReportCallback_, nullptr);
		} else {
			std::cerr << "Could not load vkDestroyDebugReportCallbackEXT\n";
		}
	}

	instance_.destroy();
}

/// filter list of desired extensions to include only those supported by current Vulkan instance.
std::vector<const char*> Executor::EnabledExtensions(const std::vector<const char*>& extensions) const {
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
std::vector<const char*> Executor::EnabledLayers(const std::vector<const char*>& layers) const {
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

/// Register a callback function for the extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME,
/// so that warnings emitted from the validation layer are actually printed.
VkDebugReportCallbackEXT Executor::RegisterValidationReporter(const vk::Instance& instance_,
  PFN_vkDebugReportCallbackEXT reporter) const {
  auto ret = VkDebugReportCallbackEXT(nullptr);
  auto createInfo = VkDebugReportCallbackCreateInfoEXT{};
  createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
  createInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT
    | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
  createInfo.pfnCallback = reporter;

  // We have to explicitly load this function
  auto DebugReportFunc = PFN_vkCreateDebugReportCallbackEXT(
    instance_.getProcAddr("vkCreateDebugReportCallbackEXT"));
  if (DebugReportFunc) {
    DebugReportFunc(instance_, &createInfo, nullptr, &ret);
  }
  else {
    std::cerr << "Could not load vkCreateDebugReportCallbackEXT\n";
  }
  return ret;
}


/// Create vulkan Instance with app specific parameters.
vk::Instance Executor::CreateInstance() {

  layers_ = isEnableValidation ? EnabledLayers({ "VK_LAYER_LUNARG_standard_validation" })
    : std::vector<const char*>{};
  auto extensions = isEnableValidation ? EnabledExtensions({ VK_EXT_DEBUG_REPORT_EXTENSION_NAME })
    : std::vector<const char*>{};

  auto app_info = vk::ApplicationInfo("Example Filter", 0, "no_engine", 
                                      0, VK_API_VERSION_1_0); // The only important field here is apiVersion
  auto create_info = vk::InstanceCreateInfo(vk::InstanceCreateFlags(), &app_info,
                                            ARR_VIEW(layers_), ARR_VIEW(extensions));

  return vk::createInstance(create_info);
}

/// @return the index of a queue family that supports compute operations.
/// Groups of queues that have the same capabilities (for instance_, they all supports graphics
/// and computer operations), are grouped into queue families.
/// When submitting a command buffer, you must specify to which queue in the family you are submitting to.
uint32_t Executor::GetComputeQueueFamilyId(const vk::PhysicalDevice& physical_device) const {
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

/// create shader_ module, reading spir-v from a file
vk::ShaderModule Executor::LoadShader(const vk::Device& device, 
                                            const char* filename,
                                            vk::ShaderModuleCreateFlags flags) const {
  /// Read binary shader_ file into array of uint32_t. little endian assumed.
  auto fin = std::ifstream(filename, std::ios::binary);
  if (!fin.is_open()) {
    throw std::runtime_error(std::string("could not open file ") + filename);
  }
  auto code = std::vector<char>(std::istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>());
  /// Padded by 0s to a boundary of 4.
  code.resize(4 * div_up(code.size(), size_t(4)));

  auto shaderCI = vk::ShaderModuleCreateInfo(flags, code.size(),
    reinterpret_cast<uint32_t*>(code.data()));
  return device.createShaderModule(shaderCI);
}

///
void Executor::BindParameters(vk::Buffer& out, const vk::Buffer& in, 
                                   const Executor::PushParams& p) const {
  vk::DescriptorSet dsc_set = CreateDescriptorSet(device_, dsc_pool_, dsc_layout_, out, in, p.width*p.height);
	cmd_buffer_ = CreateCommandBuffer(device_, cmd_pool_, pipe_, pipe_layout_, dsc_set, p);
}

///
void Executor::UnbindParameters() const {
	device_.destroyDescriptorPool(dsc_pool_);
	device_.resetCommandPool(cmd_pool_, vk::CommandPoolResetFlags());
	dsc_pool_ = AllocDescriptorPool(device_);
}

/// Run (sync) the filter on previously bound parameters
void Executor::Run() const {
	assert(cmd_buffer_ != vk::CommandBuffer{}); // TODO: this should be a check for a valid command buffer
	auto submit_info = vk::SubmitInfo(0, nullptr, nullptr, 1, &cmd_buffer_); // submit a single command buffer

	// submit the command buffer to the queue and set up a fence.
	auto queue = device_.getQueue(compute_queue_familly_id_, 0); // 0 is the queue index in the family, by default just the first one is used
	auto fence = device_.createFence(vk::FenceCreateInfo()); // fence makes sure the control is not returned to CPU till command buffer is depleted
	queue.submit({ submit_info }, fence);
	device_.waitForFences({fence}, true, uint64_t(-1));      // wait for the fence indefinitely
	device_.destroyFence(fence);
}

/// Run (sync) the filter
void Executor::execute(vk::Buffer& out, const vk::Buffer& in,
                             const Executor::PushParams& p) const {
	BindParameters(out, in, p);
	Run();
	UnbindParameters();
}

/// Specify a descriptor set layout (number and types of descriptors).
vk::DescriptorSetLayout Executor::CreateDescriptorSetLayout(const vk::Device& device) {
	auto bind_layout = std::array<vk::DescriptorSetLayoutBinding, NumDescriptors>{{
	            {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}
	           ,{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}
		                                                                          }};
	auto layoutCI = vk::DescriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags()
	                                                  , ARR_VIEW(bind_layout));
	return device.createDescriptorSetLayout(layoutCI);
}

/// Allocate descriptor pool for a descriptors to all storage buffer in use
vk::DescriptorPool Executor::AllocDescriptorPool(const vk::Device& device) {
  vk::DescriptorPoolSize descriptor_pool_size = 
    vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, NumDescriptors);
  vk::DescriptorPoolCreateInfo descriptor_poolCI = 
    vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(), 1, 1, &descriptor_pool_size);
	return device.createDescriptorPool(descriptor_poolCI);
}

/// Pipeline layout defines shader_ interface as a set of layout bindings and push constants.
vk::PipelineLayout Executor::CreatePipelineLayout(const vk::Device& device, 
                                         const vk::DescriptorSetLayout& dsc_layout) {
	auto push_const_range = vk::PushConstantRange(vk::ShaderStageFlagBits::eCompute, 
	                                                0, sizeof(PushParams));
	auto pipeline_layoutCI = vk::PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), 
	                                                     1, &dsc_layout, 1, &push_const_range);
	return device.createPipelineLayout(pipeline_layoutCI);
}

/// Create compute pipeline consisting of a single stage with compute shader_.
/// Specialization constants specialized here.
vk::Pipeline Executor::CreateComputePipeline(const vk::Device& device, 
                                         const vk::ShaderModule& shader, 
                                         const vk::PipelineLayout& pipe_layout, 
                                         const vk::PipelineCache& cache) {
	// specialize constants of the shader_
	auto specEntries = std::array<vk::SpecializationMapEntry, 2>{
		{{0, 0, sizeof(int)}, {1, 1*sizeof(int), sizeof(int)}}
	};
	auto spec_values = std::array<int, 2>{WORKGROUP_SIZE, WORKGROUP_SIZE};
	auto spec_info = vk::SpecializationInfo(ARR_VIEW(specEntries), 
	                                        spec_values.size()*sizeof(int), spec_values.data());

	// Specify the compute shader_ stage, and it's entry point (main), and specializations
	auto stageCI = vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), 
	                                                 vk::ShaderStageFlagBits::eCompute, 
	                                                 shader, "main", &spec_info);
	auto pipelineCI = vk::ComputePipelineCreateInfo(vk::PipelineCreateFlags(), 
	                                                stageCI, pipe_layout);
	return device.createComputePipeline(cache, pipelineCI, nullptr);
}

/// Create descriptor set. Actually associate buffers to binding points in bindLayout.
/// Buffer sizes are specified here as well.
vk::DescriptorSet Executor::CreateDescriptorSet(const vk::Device& device, 
                                       const vk::DescriptorPool& pool, 
                                       const vk::DescriptorSetLayout& layout, 
                                       vk::Buffer& out, 
                                       const vk::Buffer& in,
                                       uint32_t size) {
	auto descriptorSetAI = vk::DescriptorSetAllocateInfo(pool, 1, &layout);
	auto descriptorSet = device.allocateDescriptorSets(descriptorSetAI)[0];

	auto out_info = vk::DescriptorBufferInfo(out, 0, sizeof(float)*size);
	auto in_info = vk::DescriptorBufferInfo(in, 0, sizeof(float)*size);

	auto writeDsSets = std::array<vk::WriteDescriptorSet, NumDescriptors>{{
	           {descriptorSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &out_info},
	           {descriptorSet, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &in_info}
		                                                                   }};
	device.updateDescriptorSets(writeDsSets, {});
	return descriptorSet;
}

/// Create command buffer, push the push constants, bind descriptors and define the work batch size.
/// All command buffers allocated from given command pool must be submitted to queues of corresponding
/// family ONLY.
vk::CommandBuffer Executor::CreateCommandBuffer(const vk::Device& device, 
                                       const vk::CommandPool& cmd_pool, 
                                       const vk::Pipeline& pipeline, 
                                       const vk::PipelineLayout& pipe_layout, 
                                       const vk::DescriptorSet& dsc_set, 
                                       const Executor::PushParams& p) {
	// allocate a command buffer from the command pool.
	auto alloc_info = vk::CommandBufferAllocateInfo(cmd_pool, vk::CommandBufferLevel::ePrimary, 1);
	auto command_buffer = device.allocateCommandBuffers(alloc_info)[0];

	// Start recording commands into the newly allocated command buffer.
//	auto beginInfo = vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit); // buffer is only submitted and used once
	auto begin_info = vk::CommandBufferBeginInfo();
  command_buffer.begin(begin_info);

	// Before dispatch bind a pipeline, AND a descriptor set.
	// The validation layer will NOT give warnings if you forget those.
  command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
  command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipe_layout,
	                                  0, { dsc_set }, {});

  command_buffer.pushConstants(pipe_layout, vk::ShaderStageFlagBits::eCompute, 0, ST_VIEW(p));

	// Start the compute pipeline, and execute the compute shader_.
	// The number of workgroups is specified in the arguments.
  command_buffer.dispatch(div_up(p.width, WORKGROUP_SIZE), div_up(p.height, WORKGROUP_SIZE), 1);
  command_buffer.end(); // end recording commands
	return command_buffer;
}
