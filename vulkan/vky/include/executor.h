#ifndef VKY_EXECUTOR_H_
#define VKY_EXECUTOR_H_

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

inline uint32_t div_up(uint32_t x, uint32_t y) { return (x + y - 1u) / y; }

// TODO: Handle exeception.
// TODO: Release resource.
// TODO: Specify a transfer pipeline for buffer copy.

// TODO: Operator -> GeneralOp => One Pipeline, manual setup IO
//                -> CustomizedOp -> COp1 => 1.Pipeline and IO are fixed.
//                                -> COp2... 2.Can be designed externally.
class OperatorA {
public:
  int Initialize(const vk::Device device, 
    const uint32_t *max_workgroup_size,
    const uint32_t max_workgroup_invocations,
    const vk::ShaderModule shader) {

    pipes_ = new Pipeline();
    buffer_count_ = 2;
    push_constant_count_ = 3;
    GetOptimalLocalSizeXYZ(max_workgroup_size, max_workgroup_invocations, 32, 32, 1);
    pipes_->Initialize(device, shader, buffer_count_, push_constant_count_, local_size_xyz_);

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
    const void *params,
    const int params_size) {

    pipes_->UpdateDescriptorSet(buffer_memorys);

    GetGroupCount(buffer_memorys[0]->width_, buffer_memorys[0]->height_, buffer_memorys[0]->channels_);
    command->Submit(pipes_, group_count_xyz_, params, params_size);
    return 0;
  }

private:
  void GetOptimalLocalSizeXYZ(const uint32_t *max_workgroup_size,
                              const uint32_t max_workgroup_invocations,
                              const uint32_t width = 32,
                              const uint32_t height = 32,
                              const uint32_t channels = 32) {
    if (channels > 0) {
      local_size_xyz_[2] = max_workgroup_size[2];
      while (channels < local_size_xyz_[2]) {
        local_size_xyz_[2] /= 2;
      }
    }
    else {
      local_size_xyz_[2] = std::min((uint32_t)128, max_workgroup_size[2]);
    }

    uint32_t max_local_size_xy = max_workgroup_invocations / local_size_xyz_[2];

    if (height == width || (height < 0 && width < 0)) {
      uint32_t local_size_xy = std::sqrt(max_local_size_xy);
      uint32_t local_size_xy_prefer = 128;
      while (local_size_xy < local_size_xy_prefer) {
        local_size_xy_prefer /= 2;
      }
      local_size_xyz_[0] = local_size_xy_prefer;
      local_size_xyz_[1] = local_size_xy_prefer;
    }

    if (height > 0 && width > 0) {
      if (height > width) {
        float ps = height / (float)width;
        float local_size_xy = sqrt(max_local_size_xy / ps);
        local_size_xyz_[1] = local_size_xy * ps;
        local_size_xyz_[0] = std::max((uint32_t)local_size_xy, (uint32_t)1);
      }
      else {
        float ps = width / (float)height;
        float local_size_xy = sqrt(max_local_size_xy / ps);
        local_size_xyz_[1] = std::max((uint32_t)local_size_xy, (uint32_t)1);
        local_size_xyz_[0] = local_size_xy * ps;
      }

      uint32_t local_size_y_prefer = std::min((uint32_t)128, max_workgroup_size[1]);
      while (local_size_xyz_[1] < local_size_y_prefer) {
        local_size_y_prefer /= 2;
      }

      uint32_t local_size_x_prefer = std::min((uint32_t)128, max_workgroup_size[0]);
      while (local_size_xyz_[0] < local_size_x_prefer) {
        local_size_x_prefer /= 2;
      }

      local_size_xyz_[1] = local_size_y_prefer;
      local_size_xyz_[0] = local_size_x_prefer;
    }
    else if (height > 0) {
      local_size_xyz_[1] = std::min(max_local_size_xy, (uint32_t)max_workgroup_size[1]);
      while ((uint32_t)height < local_size_xyz_[1]) {
        local_size_xyz_[1] /= 2;
      }

      uint32_t max_local_size_x = max_local_size_xy / local_size_xyz_[1];
      local_size_xyz_[0] = std::min(max_local_size_x, (uint32_t)max_workgroup_size[0]);
    }
    else if (width > 0) {
      local_size_xyz_[0] = std::min(max_local_size_xy, (uint32_t)max_workgroup_size[0]);
      while ((uint32_t)width < local_size_xyz_[0]) {
        local_size_xyz_[0] /= 2;
      }

      uint32_t max_local_size_y = max_local_size_xy / local_size_xyz_[0];
      local_size_xyz_[1] = std::min(max_local_size_y, (uint32_t)max_workgroup_size[1]);
    }
  }

  void GetGroupCount(const uint32_t width, const uint32_t height, const uint32_t channels) {
    group_count_xyz_[0] = (width + local_size_xyz_[0] - 1) / local_size_xyz_[0];
    group_count_xyz_[1] = (height + local_size_xyz_[1] - 1) / local_size_xyz_[1];
    group_count_xyz_[2] = (channels + local_size_xyz_[2] - 1) / local_size_xyz_[2];
  }

private:
  DeviceInfo *device_info_;
  // TODO: std::vector<Pipeline *> pipes_;
  Pipeline *pipes_;

  uint32_t buffer_count_;
  uint32_t push_constant_count_;

  uint32_t local_size_xyz_[3];
  uint32_t group_count_xyz_[3];
}; // class Operator

class Executor {

public:

  Executor() {}
  ~Executor() {}

  const vk::Device &device() const { return device_; }
  Allocator *allocator() const { return allocator_; }

  int Initialize(const DeviceInfo *device_info, const std::string &shaders_dir_path) {
    // Init Device.
    device_ = CreateDevice(device_info->physical_device_, device_info->compute_queue_familly_id_);
    RegisterShaders(shaders_dir_path);

    // Init command.
    command_ = new Command();
    command_->Initialize(device_, device_info->compute_queue_familly_id_);

    // Init Operators.
    op_ = new OperatorA();
    op_->Initialize(device_, 
                    device_info->max_workgroup_size_, 
                    device_info->max_workgroup_invocations_, 
                    shader("saxpy"));

    allocator_ = new Allocator(device_, device_info->physical_device_, command_);
    return 0;
  }

  int UnInitialize() {
    if (allocator_ != nullptr) {
      delete allocator_;
      allocator_ = nullptr;
    }
    if (op_ != nullptr) {
      op_->UnInitialize();
      delete op_;
      op_ = nullptr;
    }
    if (command_ != nullptr) {
      command_->UnInitialize();
      delete command_;
      command_ = nullptr;
    }
    return 0;
  }
  // TODO: Data, Bind the buffers and buffer_range together. 
  //       Create a new class for Buffer.
  int Run(const std::vector<vky::BufferMemory *> &buffer_memorys,
    const void *params, 
    const int params_size) const {

    op_->Run(command_, buffer_memorys, params, params_size);

    return 0;
  }

private:

  void RegisterShaders(const std::string &shaders_dir_path) {
    shaders_map_["add"] = CreateShaderModule(shaders_dir_path + "shaders/add.spv");
    shaders_map_["saxpy"] = CreateShaderModule(shaders_dir_path + "shaders/saxpy.spv");
  }

  vk::ShaderModule shader(const std::string &str) {
    std::map<std::string, vk::ShaderModule>::iterator it = shaders_map_.find(str);
    if (it == shaders_map_.end()) {
      throw std::runtime_error(std::string("could not find shader: ") + it->first);
    }
    return it->second;
  }

  vk::ShaderModule CreateShaderModule(const std::string &filename) {
    // Read binary shader file into array of uint32_t. little endian assumed.
    auto fin = std::ifstream(filename.c_str(), std::ios::binary);
    if (!fin.is_open()) {
      throw std::runtime_error(std::string("could not open file ") + filename.c_str());
    }
    auto code = std::vector<char>(std::istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>());
    // Padded by 0s to a boundary of 4.
    code.resize(4 * div_up(code.size(), size_t(4)));

    vk::ShaderModuleCreateFlags flags = vk::ShaderModuleCreateFlags();
    auto shader_module_create_info = vk::ShaderModuleCreateInfo(flags, code.size(),
      reinterpret_cast<uint32_t*>(code.data()));

    return device_.createShaderModule(shader_module_create_info);
  }

  vk::Device CreateDevice(const vk::PhysicalDevice &physical_device, const uint32_t compute_queue_familly_id) {
    // create logical device to interact with the physical one
    // When creating the device specify what queues it has
    // TODO: when physical device is a discrete gpu, transfer queue needs to be included
    float p = float(1.0); // queue priority
    vk::DeviceQueueCreateInfo queue_create_info =
      vk::DeviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), compute_queue_familly_id, 1, &p);
    vk::DeviceCreateInfo device_create_info = 
      vk::DeviceCreateInfo(vk::DeviceCreateFlags(), 1, &queue_create_info, 0, nullptr);//layers_.size(), layers_.data()

    return physical_device.createDevice(device_create_info, nullptr);
  }

private:  
  vk::Device device_;    // logical device providing access to a physical one 
  std::map<std::string, vk::ShaderModule> shaders_map_;

  Command *command_;
  // TODO: GeneralOp and CustomizedOp.
  OperatorA *op_;

  Allocator *allocator_;
}; // class Executor

} // namespace vky

#endif  // VKY_EXECUTOR_H_