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
//#include "operator.h"
#include "op_factory.h"

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
    //RegisterShaderNamePath(shaders_dir_path);

    // Init command.
    command_ = new Command();
    command_->Initialize(device_, device_info->compute_queue_familly_id_);

    // One factory for one executor.
    op_factory_ = new OpFactory(device_, 
                                device_info->max_workgroup_size_,
                                device_info->max_workgroup_invocations_, 
                                shaders_dir_path);

    allocator_ = new Allocator(device_, device_info->physical_device_, command_);
    return 0;
  }

  int UnInitialize() {
    if (allocator_ != nullptr) {
      delete allocator_;
      allocator_ = nullptr;
    }
    if (op_factory_ != nullptr) {
      delete op_factory_;
      op_factory_ = nullptr;
    }
    if (command_ != nullptr) {
      command_->UnInitialize();
      delete command_;
      command_ = nullptr;
    }
    return 0;
  }

  int Run(const std::string op_name, 
          const std::vector<vky::BufferMemory *> &buffer_memorys,
          const void *push_params, 
          const int push_params_size) {

    // Get op from the map of factory first.
    // If it is not exist, then create one by factory.
    op_ = op_factory_->GetOpByName(op_name);

    op_->Run(command_, buffer_memorys, push_params, push_params_size);

    return 0;
  }

private:

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
  std::map<std::string, vk::ShaderModule> shaders_name_obj_;
  std::map<std::string, std::string> shaders_name_path_;

  Command *command_;
  Operator *op_;
  OpFactory *op_factory_;

  Allocator *allocator_;
}; // class Executor

} // namespace vky

#endif  // VKY_EXECUTOR_H_