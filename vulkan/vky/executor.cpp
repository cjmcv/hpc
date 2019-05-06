#include "executor.h"

namespace vky {

//  DeviceManager
// <Public
int DeviceManager::Initialize(bool is_enable_validation) {
  std::vector<const char*> layers = std::vector<const char*>{};
  std::vector<const char*> extensions = std::vector<const char*>{};
  if (is_enable_validation) {
    // "=" in vector is deep copy.
    layers = EnabledLayers({ "VK_LAYER_LUNARG_standard_validation" });
    extensions = EnabledExtensions({ VK_EXT_DEBUG_REPORT_EXTENSION_NAME });
  }

  CreateInstance(layers, extensions);
  SearchPhysicalDevices();

  return 0;
}

int DeviceManager::CreateInstance(std::vector<const char*> &layers,
  std::vector<const char*> &extensions) {

  auto app_info = vk::ApplicationInfo("Example Filter", 0, "no_engine",
    0, VK_API_VERSION_1_0); // The only important field here is apiVersion
  auto create_info = vk::InstanceCreateInfo(vk::InstanceCreateFlags(), &app_info,
    layers.size(), layers.data(), 
    extensions.size(), extensions.data());

  instance_ = vk::createInstance(create_info);
  return 0;
}

int DeviceManager::SearchPhysicalDevices() {
  vkEnumeratePhysicalDevices(instance_, &device_count_, NULL);
  if (device_count_ == 0) {
    throw std::runtime_error("could not find a device with vulkan support");
  }
  physical_devices_.resize(device_count_);
  instance_.enumeratePhysicalDevices(&device_count_, physical_devices_.data());

  return 0;
}

} // namespace vky