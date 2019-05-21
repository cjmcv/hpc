#include "command.h"

namespace vky {

///////////////
// <Public
int Pipeline::Initialize(const vk::Device device, 
  const uint32_t *max_workgroup_size,
  const uint32_t max_workgroup_invocations,
  const vk::ShaderModule shader, 
  const uint32_t buffer_count,
  const uint32_t push_constant_count) {

  device_ = device;
  max_workgroup_size_[0] = max_workgroup_size[0];
  max_workgroup_size_[1] = max_workgroup_size[1];
  max_workgroup_size_[2] = max_workgroup_size[2];
  max_workgroup_invocations_ = max_workgroup_invocations;

  num_descriptors_ = buffer_count;
  push_constant_count_ = push_constant_count;

  CreateDescriptorsetLayout();  // num_descriptors_
  CreatePipelineLayout();       // push_constant_count_
  CreatePipeline(shader);       // pipe_cache_ and pipeline_

  CreateDescriptorPool();
  AllocateDescriptorSet();
  return 0;
}
void Pipeline::UnInitialize() {
  FreeDescriptorSet();
  DestroyDescriptorPool();

  DestroyPipeline();
  DestroyPipelineLayout();
  DestroyDescriptorsetLayout();
}

void Pipeline::SetOptimalLocalSizeXYZ(const int height, const int width, const int channels) {
  if (channels > 0) {
    local_size_z_ = max_workgroup_size_[2];
    while ((uint32_t)channels < local_size_z_) {
      local_size_z_ /= 2;
    }
  }
  else {
    local_size_z_ = std::min((uint32_t)128, max_workgroup_size_[2]);
  }

  uint32_t max_local_size_xy = max_workgroup_invocations_ / local_size_z_;

  if (height == width || (height < 0 && width < 0)) {
    uint32_t local_size_xy = std::sqrt(max_local_size_xy);
    uint32_t local_size_xy_prefer = 128;
    while (local_size_xy < local_size_xy_prefer) {
      local_size_xy_prefer /= 2;
    }
    local_size_x_ = local_size_xy_prefer;
    local_size_y_ = local_size_xy_prefer;
  }

  if (height > 0 && width > 0) {
    if (height > width) {
      float ps = height / (float)width;
      float local_size_xy = sqrt(max_local_size_xy / ps);
      local_size_y_ = local_size_xy * ps;
      local_size_x_ = std::max((uint32_t)local_size_xy, (uint32_t)1);
    }
    else {
      float ps = width / (float)height;
      float local_size_xy = sqrt(max_local_size_xy / ps);
      local_size_y_ = std::max((uint32_t)local_size_xy, (uint32_t)1);
      local_size_x_ = local_size_xy * ps;
    }

    uint32_t local_size_y_prefer = std::min((uint32_t)128, max_workgroup_size_[1]);
    while (local_size_y_ < local_size_y_prefer) {
      local_size_y_prefer /= 2;
    }

    uint32_t local_size_x_prefer = std::min((uint32_t)128, max_workgroup_size_[0]);
    while (local_size_x_ < local_size_x_prefer)
    {
      local_size_x_prefer /= 2;
    }

    local_size_y_ = local_size_y_prefer;
    local_size_x_ = local_size_x_prefer;
  }
  else if (height > 0) {
    local_size_y_ = std::min(max_local_size_xy, (uint32_t)max_workgroup_size_[1]);
    while ((uint32_t)height < local_size_y_) {
      local_size_y_ /= 2;
    }

    uint32_t max_local_size_x = max_local_size_xy / local_size_y_;
    local_size_x_ = std::min(max_local_size_x, (uint32_t)max_workgroup_size_[0]);
  }
  else if (width > 0) {
    local_size_x_ = std::min(max_local_size_xy, (uint32_t)max_workgroup_size_[0]);
    while ((uint32_t)width < local_size_x_) {
      local_size_x_ /= 2;
    }

    uint32_t max_local_size_y = max_local_size_xy / local_size_x_;
    local_size_y_ = std::min(max_local_size_y, (uint32_t)max_workgroup_size_[1]);
  }
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
int Pipeline::CreatePipeline(const vk::ShaderModule shader) {
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

void Pipeline::DestroyPipeline() {
  device_.destroyPipeline(pipeline_);
  device_.destroyPipelineCache(pipe_cache_);
}

} // namespace vky