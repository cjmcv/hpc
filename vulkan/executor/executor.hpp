#pragma once

#include <vulkan/vulkan.hpp>
#include <iostream>

/// doc me
struct Executor {
	static constexpr auto NumDescriptors = uint32_t(2); ///< number of binding descriptors (array input-output parameters)
	
	/// C++ mirror of the shader push constants interface
	struct PushParams {
		uint32_t width;  ///< frame width
		uint32_t height; ///< frame height
		float a;         ///< saxpy (\$ y = y + ax \$) scaling factor
	};
	
public: // data
  std::vector<const char*> layers_; 
	vk::Instance instance_;                        // Vulkan instance
	VkDebugReportCallbackEXT DebugReportCallback_; //
  uint32_t device_count_;
  std::vector<vk::PhysicalDevice> phys_devices_; // physical device
  
	vk::Device device_;                  ///< logical device providing access to a physical one
	vk::ShaderModule shader_;            ///< compute shader
	vk::DescriptorSetLayout dsc_layout_;  ///< c++ definition of the shader binding interface
	mutable vk::DescriptorPool dsc_pool_; ///< descriptors pool
	vk::CommandPool cmd_pool_;            ///< used to allocate command buffers
	vk::PipelineCache pipe_cache_;        ///< pipeline cache
	vk::PipelineLayout pipe_layout_;      ///< defines shader interface as a set of layout bindings and push constants
	
	vk::Pipeline pipe_;                   ///< pipeline to submit compute commands
	mutable vk::CommandBuffer cmd_buffer_; ///< commands recorded here, once command buffer is submitted to a queue those commands get executed
	
	uint32_t compute_queue_familly_id_;   ///< index of the queue family supporting compute loads

public:
	explicit Executor(const std::string& shaderPath);
	~Executor() noexcept;
	

  void execute(vk::Buffer& out, const vk::Buffer& in, const PushParams& p ) const;

private: // helpers	
  
  vk::Instance CreateInstance();

  std::vector<const char*> EnabledExtensions(const std::vector<const char*>& extensions) const;

  std::vector<const char*> EnabledLayers(const std::vector<const char*>& layers) const;

  VkDebugReportCallbackEXT RegisterValidationReporter(
    const vk::Instance& instance, PFN_vkDebugReportCallbackEXT reporter) const;

  uint32_t GetComputeQueueFamilyId(const vk::PhysicalDevice& physical_device) const;

  vk::ShaderModule LoadShader(const vk::Device& device,
                              const char* filename,
                              vk::ShaderModuleCreateFlags flags
                            = vk::ShaderModuleCreateFlags()) const;

  void BindParameters(vk::Buffer& out, const vk::Buffer& in, const PushParams& p) const;
  void UnbindParameters() const;
  void Run() const;

	
	static vk::DescriptorSetLayout CreateDescriptorSetLayout(const vk::Device& device);
	static vk::DescriptorPool AllocDescriptorPool(const vk::Device& device);
	
	static vk::PipelineLayout CreatePipelineLayout(const vk::Device& device, 
	                                 const vk::DescriptorSetLayout& dsc_layout);
	
	static vk::Pipeline CreateComputePipeline(const vk::Device& device, const vk::ShaderModule& shader, 
	                                  const vk::PipelineLayout& pipe_layout, 
	                                  const vk::PipelineCache& cache);
	
	static vk::DescriptorSet CreateDescriptorSet(const vk::Device& device, const vk::DescriptorPool& pool, 
	                                const vk::DescriptorSetLayout& layout, 
	                                vk::Buffer& out, 
	                                const vk::Buffer& in, 
	                                uint32_t size);
	
	static vk::CommandBuffer CreateCommandBuffer(const vk::Device& device, const vk::CommandPool& cmd_pool, 
	                                const vk::Pipeline& pipeline, const vk::PipelineLayout& pipe_layout, 
	                                const vk::DescriptorSet& dscSet, 
	                                const PushParams& p);
}; // struct MixpixFilter
