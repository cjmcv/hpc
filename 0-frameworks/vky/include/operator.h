#ifndef VKY_OPERATOR_H_
#define VKY_OPERATOR_H_

#include <iostream>
#include <fstream>
#include <map>

#include <vulkan/vulkan.hpp>

#include "data_type.h"
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
    const vk::ShaderModule shader);

  void UnInitialize();

  int Run(Command *command,
    const std::vector<vky::BufferMemory *> &buffer_memorys,
    const void *push_params,
    const int push_params_size);

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