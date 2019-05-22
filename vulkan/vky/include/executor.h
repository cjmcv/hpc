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
#include "operator.h"

namespace vky {

inline uint32_t div_up(uint32_t x, uint32_t y) { return (x + y - 1u) / y; }

// TODO: Handle exeception.
// TODO: Release resource.
// TODO: Specify a transfer pipeline for buffer copy.

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

    // TODO: 1.op factory; 2.And get Op in running.
    //       std::map<std::string, Operator *> ops;
    //       One Execitor can have multiple ops.
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

    // TODO: Get op from map, if it is not exist, then create one by factory.
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