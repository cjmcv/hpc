#ifndef VKY_OPERATOR_H_
#define VKY_OPERATOR_H_

#include <iostream>
#include <fstream>
#include <map>

#include <vulkan/vulkan.hpp>

#include "data_type.h"

#include "device.h"
#include "allocator.h"
#include "command.h"
#include "pipeline.h"

namespace vky {

//////////////////////////////////////
// OpParams: object accessible.
class OpParams {
public:
  OpParams(
    std::string name,
    std::string shader_file,
    uint32_t buffer_count,
    uint32_t push_constant_count,
    uint32_t local_width,
    uint32_t local_height,
    uint32_t local_channels,
    uint32_t group_depends_id) :
    name_(name),
    shader_file_(shader_file),
    buffer_count_(buffer_count),
    push_constant_count_(push_constant_count),
    local_width_(local_width),
    local_height_(local_height),
    local_channels_(local_channels),
    group_depends_id_(group_depends_id) {
  };

  std::string name_;
  std::string shader_file_;

  uint32_t buffer_count_;
  uint32_t push_constant_count_;

  uint32_t local_width_;
  uint32_t local_height_;
  uint32_t local_channels_;
  uint32_t group_depends_id_;
};

class Operator {
public:
  int Initialize(const vk::Device device, 
    const uint32_t *max_workgroup_size,
    const uint32_t max_workgroup_invocations,
    const OpParams *op_params,
    const vk::ShaderModule shader) {

    op_params_ = op_params; // Only one for Op.

    pipe_ = new Pipeline();
    uint32_t buffer_count = op_params_->buffer_count_;
    uint32_t push_constant_count = op_params_->push_constant_count_;
    GetOptimalLocalSizeXYZ(max_workgroup_size, 
                           max_workgroup_invocations, 
                           op_params_->local_width_, 
                           op_params_->local_height_, 
                           op_params_->local_channels_);
    pipe_->Initialize(device, shader, buffer_count, push_constant_count, local_size_xyz_);

    return 0;
  }

  void UnInitialize() {
    if (pipe_ != nullptr) {
      pipe_->UnInitialize();
      pipe_ = nullptr;
    }
  }

  int Run(Command *command,
    const std::vector<vky::BufferMemory *> &buffer_memorys,
    const void *push_params,
    const int push_params_size) {

    pipe_->UpdateDescriptorSet(buffer_memorys);

    int index = op_params_->group_depends_id_;
    GetGroupCount(buffer_memorys[index]->width_, buffer_memorys[index]->height_, buffer_memorys[index]->channels_);

    command->ComputeShader(pipe_, group_count_xyz_, push_params, push_params_size);
    return 0;
  }

private:
  void GetOptimalLocalSizeXYZ(const uint32_t *max_workgroup_size,
    const uint32_t max_workgroup_invocations,
    const uint32_t width = 32,
    const uint32_t height = 32,
    const uint32_t channels = 32);

  void GetGroupCount(const uint32_t width,
    const uint32_t height,
    const uint32_t channels);

private:
  DeviceInfo *device_info_;
  uint32_t local_size_xyz_[3];
  uint32_t group_count_xyz_[3];

  Pipeline *pipe_;
  const OpParams *op_params_;
}; // class NormalOp

} // namespace vky

#endif  // VKY_OPERATOR_H_