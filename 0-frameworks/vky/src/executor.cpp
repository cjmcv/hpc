#include "executor.h"

namespace vky {

///////////////
// <Public
int Executor::Initialize(const int physical_device_id, const std::string &shaders_dir_path) {
  instance_ = CreateInstance(true);

  dm_.QueryDeviceInfo(instance_);
  dm_.PrintDevicesInfo();

  // Create logical device by physical_device.
  device_ = dm_.CreateLogicalDevice(physical_device_id);

  DeviceInfo device_info = dm_.device_info(physical_device_id);

  // Init command.
  command_ = new Command();
  command_->Initialize(device_, device_info.compute_queue_familly_id_);

  // One factory for one executor.
  op_factory_ = new OpFactory(device_,
    device_info.max_workgroup_size_,
    device_info.max_workgroup_invocations_,
    shaders_dir_path);

  allocator_ = new Allocator(device_, device_info.physical_device_, command_);
  return 0;
}

int Executor::UnInitialize() {
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

  device_.destroy();

  DestroyInstance();
  return 0;
}

int Executor::Run(const std::string op_name,
  const std::vector<vky::BufferMemory *> &buffer_memorys,
  const void *push_params,
  const int push_params_size) {

  // Get op from the map of factory.
  // If it doesn't exist, then use factory to create one.
  op_ = op_factory_->GetOpByName(op_name);
  op_->Run(command_, buffer_memorys, push_params, push_params_size);

  return 0;
}

///////////////
// <Private.
std::vector<const char*> Executor::EnabledExtensions(const std::vector<const char*>& extensions) const {
  auto ret = std::vector<const char*>{};
  auto instance_extensions = vk::enumerateInstanceExtensionProperties();

  for (auto e : extensions) {
    bool is_exist = false;
    for (auto ie : instance_extensions) {
      if (!strcmp(ie.extensionName, e)) {
        ret.push_back(e);
        is_exist = true;
        break;
      }
    }

    if (!is_exist) {
      std::cerr << "[WARNING] extension " << e << " can not be found. \n";
    }
  }

  // Not all extension are supported.
  if (ret.size() != extensions.size()) {
    std::cout << "Supported extensions: " << std::endl;
    for (auto ie : instance_extensions) {
      std::cout << ie.extensionName << std::endl;
    }
  }
  return ret;
}

std::vector<const char*> Executor::EnabledLayers(const std::vector<const char*>& layers) const {
  auto ret = std::vector<const char*>{};
  auto instance_layers = vk::enumerateInstanceLayerProperties();
  for (auto l : layers) {
    bool is_exist = false;
    for (auto il : instance_layers) {
      if (!strcmp(il.layerName, l)) {
        ret.push_back(l);
        is_exist = true;
        break;
      }
    }

    if (!is_exist) {
      std::cerr << "[WARNING] layer " << l << " can not be found. \n";
    }
  }

  // Not all layers are supported.
  if (ret.size() != layers.size()) {
    std::cout << "Supported layers: " << std::endl;
    for (auto il : instance_layers) {
      std::cout << il.layerName << std::endl;
    }
  }

  return ret;
}

vk::Instance &Executor::CreateInstance(bool is_enable_validation) {

  std::vector<const char*> layers = std::vector<const char*>{};
  std::vector<const char*> extensions = std::vector<const char*>{};
  if (is_enable_validation) {
    // "=" in vector is deep copy. 
    // Note: VK_LAYER_LUNARG_standard_validation is deprecated.
    layers = EnabledLayers({ "VK_LAYER_KHRONOS_validation" });
    // The same as VK_EXT_DEBUG_REPORT_EXTENSION_NAME
    extensions = EnabledExtensions({ "VK_EXT_debug_report" });
  }

  auto app_info = vk::ApplicationInfo("Vulkan Compute Example", 0, "no_engine",
    0, VK_API_VERSION_1_0); // The only important field here is apiVersion
  auto create_info = vk::InstanceCreateInfo(vk::InstanceCreateFlags(), &app_info,
    layers.size(), layers.data(), 
    extensions.size(), extensions.data());

  return vk::createInstance(create_info);
}

void Executor::DestroyInstance() {
  instance_.destroy();
}
} // namespace vky