#include "command.h"

namespace vky {

///////////////
// <Public
int Pipeline::Initialize(const vk::Device device, 
                         const vk::ShaderModule shader, 
                         const uint32_t buffer_count,
                         const uint32_t push_constant_count,
                         const uint32_t *local_size) {

  device_ = device;

  num_descriptors_ = buffer_count;
  push_constant_count_ = push_constant_count;

  CreateDescriptorsetLayout();  // num_descriptors_
  CreatePipelineLayout();       // push_constant_count_
  CreatePipeline(shader, local_size);       // pipe_cache_ and pipeline_

  CreateDescriptorPool();
  AllocateDescriptorSet();
  return 0;
}
void Pipeline::UnInitialize() {
  // TODO: Recheck.
  //FreeDescriptorSet();
  DestroyDescriptorPool();

  DestroyPipeline();
  DestroyPipelineLayout();
  DestroyDescriptorsetLayout();
}


int Pipeline::CreateDescriptorPool() {
  vk::DescriptorPoolSize descriptor_pool_size =
    vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, num_descriptors_);
  vk::DescriptorPoolCreateInfo descriptor_pool_create_info =
    vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(), 1, 1, &descriptor_pool_size);
  descriptor_pool_ = device_.createDescriptorPool(descriptor_pool_create_info);

  return 0;
}

int Pipeline::DestroyDescriptorPool() {
  device_.destroyDescriptorPool(descriptor_pool_);
  return 0;
}

int Pipeline::AllocateDescriptorSet() {
  ///Create descriptor set.Actually associate buffers to binding points in bindLayout.
  /// Buffer sizes are specified here as well.
  vk::DescriptorSetAllocateInfo allocate_info =
    vk::DescriptorSetAllocateInfo(descriptor_pool_, 1, &descriptor_set_layout_);

  descriptor_set_ = device_.allocateDescriptorSets(allocate_info)[0];
  return 0;
}
void Pipeline::FreeDescriptorSet() {
  device_.freeDescriptorSets(descriptor_pool_, 1, &descriptor_set_);
}

int Pipeline::UpdateDescriptorSet(const std::vector<vky::BufferMemory *> &buffer_memorys) {
  if (buffer_memorys.size() != num_descriptors_) {
    throw std::runtime_error("UpdateDescriptorSet -> buffers.size() != num_descriptors_");
  }

  std::vector<vk::DescriptorBufferInfo> buffers_info(num_descriptors_);
  for (int i = 0; i < num_descriptors_; i++) {
    buffers_info[i].setBuffer(buffer_memorys[i]->buffer_);
    buffers_info[i].setOffset(0);
    buffers_info[i].setRange(buffer_memorys[i]->buffer_range_);
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
///////////////
// <Private
int Pipeline::CreateDescriptorsetLayout() {
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
void Pipeline::DestroyDescriptorsetLayout() {
  device_.destroyDescriptorSetLayout(descriptor_set_layout_);
}

int Pipeline::CreatePipelineLayout() {

  auto push_const_range = vk::PushConstantRange(vk::ShaderStageFlagBits::eCompute,
    0, sizeof(int) * push_constant_count_);
  auto pipe_layout_create_info = vk::PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(),
    1, &descriptor_set_layout_, 1, &push_const_range);
  pipeline_layout_ = device_.createPipelineLayout(pipe_layout_create_info);

  return 0;
}
void Pipeline::DestroyPipelineLayout() {
  device_.destroyPipelineLayout(pipeline_layout_);
}

// Create compute pipeline consisting of a single stage with compute shader.
// Specialization constants specialized here.
int Pipeline::CreatePipeline(const vk::ShaderModule shader, const uint32_t *local_size) {
  pipe_cache_ = device_.createPipelineCache(vk::PipelineCacheCreateInfo());

  // specialize constants of the shader
  // {constantID, offset, size}
  // 3: local_size_x, local_size_y, local_size_z
  auto spec_entries = std::array<vk::SpecializationMapEntry, 3>{
    { { 0, 0, sizeof(uint32_t) },
      { 1, 1 * sizeof(uint32_t), sizeof(uint32_t) },
      { 2, 2 * sizeof(uint32_t), sizeof(uint32_t) } }
  };
  auto spec_values = std::array<uint32_t, 3>{local_size[0], local_size[1], local_size[2]};
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

void Pipeline::DestroyPipeline() {
  device_.destroyPipeline(pipeline_);
  device_.destroyPipelineCache(pipe_cache_);
}

} // namespace vky