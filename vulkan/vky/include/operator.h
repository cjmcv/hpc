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
// TODO: Operator -> GeneralOp => One Pipeline, manual setup IO
//                -> CustomizedOp -> COp1 => 1.Pipeline and IO are fixed.
//                                -> COp2... 2.Can be designed externally.
class OpParams {};

class Operator {
public:
  virtual int Initialize(const vk::Device device,
                         const uint32_t *max_workgroup_size,
                         const uint32_t max_workgroup_invocations,
                         const vk::ShaderModule shader) {
    return -1;
  }
  virtual void UnInitialize() { return; }
  virtual int Run(Command *command,
                  const std::vector<vky::BufferMemory *> &buffer_memorys,
                  const void *params,
                  const int params_size) {
    return -1;
  }

protected:
  void GetOptimalLocalSizeXYZ(const uint32_t *max_workgroup_size,
                              const uint32_t max_workgroup_invocations,
                              const uint32_t width = 32,
                              const uint32_t height = 32,
                              const uint32_t channels = 32);

  void GetGroupCount(const uint32_t width, 
                     const uint32_t height, 
                     const uint32_t channels);

protected:
  DeviceInfo *device_info_;
  uint32_t local_size_xyz_[3];
  uint32_t group_count_xyz_[3];
};

////////////////////////////
// NormalOp
class NormalOpParams :OpParams {
public:
  std::string shader_;

  uint32_t buffer_count_;
  uint32_t push_constant_count_;
 
  uint32_t local_width_;
  uint32_t local_height_;
  uint32_t local_channels_;
  uint32_t group_depends_id_;
};

/*
uint32_t buffer_count = 2;
uint32_t push_constant_count = 3;
GetOptimalLocalSizeXYZ(max_workgroup_size, max_workgroup_invocations, 32, 32, 1);
group_depends_id_ = 0;
*/
class NormalOp :Operator {
public:
  int Initialize(const vk::Device device, 
    const uint32_t *max_workgroup_size,
    const uint32_t max_workgroup_invocations,
    const vk::ShaderModule shader) {

    ///
    op_params_ = new NormalOpParams();
    op_params_->buffer_count_ = 2;
    op_params_->push_constant_count_ = 3;
    op_params_->local_width_ = 32;
    op_params_->local_height_ = 32;
    op_params_->local_channels_ = 1;
    op_params_->group_depends_id_ = 0;
    ///
    pipes_ = new Pipeline();
    uint32_t buffer_count = op_params_->buffer_count_;
    uint32_t push_constant_count = op_params_->push_constant_count_;
    GetOptimalLocalSizeXYZ(max_workgroup_size, 
                           max_workgroup_invocations, 
                           op_params_->local_width_, 
                           op_params_->local_height_, 
                           op_params_->local_channels_);
    pipes_->Initialize(device, shader, buffer_count, push_constant_count, local_size_xyz_);

    return 0;
  }
  void UnInitialize() {
    if (pipes_ != nullptr) {
      pipes_->UnInitialize();
      pipes_ = nullptr;
    }
  }

  int Run(Command *command,
    const std::vector<vky::BufferMemory *> &buffer_memorys,
    const void *push_params,
    const int push_params_size) {

    pipes_->UpdateDescriptorSet(buffer_memorys);

    GetGroupCount(buffer_memorys[0]->width_, buffer_memorys[0]->height_, buffer_memorys[0]->channels_);
    command->Submit(pipes_, group_count_xyz_, push_params, push_params_size);
    return 0;
  }

private:
  // TODO: std::vector<Pipeline *> pipes_;
  Pipeline *pipes_;

  NormalOpParams *op_params_;
}; // class Operator

} // namespace vky

#endif  // VKY_OPERATOR_H_