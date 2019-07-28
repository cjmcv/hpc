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

#include "op_hub.h"

namespace vky {
// TODO: Operator -> GeneralOp => One Pipeline, manual setup IO
//                -> CustomizedOp -> COp1 => 1.Pipeline and IO are fixed.
//                                -> COp2... 2.Can be designed externally.
class Operator {
public:
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

class NormalOp :public Operator {
public:
  int Initialize(const vk::Device device, 
    const uint32_t *max_workgroup_size,
    const uint32_t max_workgroup_invocations,
    const NormalOpParams *op_params,
    const vk::ShaderModule shader) {

    op_params_ = op_params; // Only one for NormalOp.

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

    command->Submit(pipe_, group_count_xyz_, push_params, push_params_size);
    return 0;
  }

private:
  Pipeline *pipe_;
  const NormalOpParams *op_params_;
}; // class NormalOp

// TODO: A combination of multiple operators.
class HybridOp :public Operator {
public:
  int Initialize(const vk::Device device,
    const uint32_t *max_workgroup_size,
    const uint32_t max_workgroup_invocations,
    const std::vector<NormalOpParams *> &op_params,
    const std::vector<vk::ShaderModule> &shader) {
    
    num_sub_ops_ = op_params.size();
    for (int i = 0; i < num_sub_ops_; i++) {
      ops_.push_back(new NormalOp());
      ops_[i]->Initialize(device, max_workgroup_size, max_workgroup_invocations, op_params[i], shader[i]);
    }
    return 0;
  }
  void UnInitialize() {
    for (int i = 0; i < num_sub_ops_; i++) {
      ops_[i]->UnInitialize();
    }
  }

  int Run(Command *command,
    const std::vector<vky::BufferMemory *> &buffer_memorys,
    const void *push_params,
    const int push_params_size) {

    //for (int i = 0; i < num_sub_ops_; i++) {
    //  ops_[i]->Run(command, buffer_memorys, push_params, push_params_size);
    //}
    return 0;
  }

  std::vector<NormalOp *> ops_;
  int num_sub_ops_;
}; // class NormalOp

} // namespace vky

#endif  // VKY_OPERATOR_H_