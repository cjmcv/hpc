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
#include "op_factory.h"

namespace vky {

class Executor {

public:

  Executor() {}
  ~Executor() {}

  const vk::Device &device() const { return device_; }
  Allocator *allocator() const { return allocator_; }

  int Initialize(const int physical_device_id, const std::string &shaders_dir_path);

  int UnInitialize();

  int Run(const std::string op_name, 
          const std::vector<vky::BufferMemory *> &buffer_memorys,
          const void *push_params, 
          const int push_params_size);

private:
  // Filter list of desired extensions to include only those supported by current Vulkan instance.
  std::vector<const char*> EnabledExtensions(const std::vector<const char*>& extensions) const;

  // Filter list of desired layers to include only those supported by current Vulkan instance
  std::vector<const char*> EnabledLayers(const std::vector<const char*>& layers) const;

  // Create & Destory vk::Instance.
  vk::Instance &CreateInstance(bool is_enable_validation);
  void DestroyInstance();

private:
  DeviceManager dm_;
  vk::Instance instance_;

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