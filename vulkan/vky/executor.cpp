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
  CreateDevice();

  // TODO: move to other place.
  shaders_map_["saxpy"] = CreateShaderModule("src/shaders/saxpy.spv");
  //shader_ = CreateShaderModule("src/shaders/saxpy.spv");

  return 0;
}

int DeviceManager::CreateInstance(std::vector<const char*> &layers,
  std::vector<const char*> &extensions) {

  auto app_info = vk::ApplicationInfo("Example Filter", 0, "no_engine",
    0, VK_API_VERSION_1_0); // The only important field here is apiVersion
  auto create_info = vk::InstanceCreateInfo(vk::InstanceCreateFlags(), &app_info,
    ARR_VIEW(layers), ARR_VIEW(extensions));

  instance_ = vk::createInstance(create_info);
  return 0;
}

int DeviceManager::SearchPhysicalDevices() {
  vkEnumeratePhysicalDevices(instance_, &device_count_, NULL);
  if (device_count_ == 0) {
    throw std::runtime_error("could not find a device with vulkan support");
  }
  phys_devices_.resize(device_count_);
  instance_.enumeratePhysicalDevices(&device_count_, phys_devices_.data());

  return 0;
}

int DeviceManager::CreateDevice(int device_id) {
  compute_queue_familly_id_ = GetComputeQueueFamilyId(phys_devices_[device_id]);

  // create logical device_ to interact with the physical one
  // When creating the device specify what queues it has
  // TODO: when physical device is a discrete gpu, transfer queue needs to be included
  auto p = float(1.0); // queue priority
  auto queue_create_info = vk::DeviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), compute_queue_familly_id_, 1, &p);
  auto device_create_info = vk::DeviceCreateInfo(vk::DeviceCreateFlags(), 1, &queue_create_info, 0, nullptr);//ARR_VIEW(layers_)
  device_ = phys_devices_[device_id].createDevice(device_create_info, nullptr);

  return 0;
}

// <Private
vk::ShaderModule DeviceManager::CreateShaderModule(const char* filename) {
  // Read binary shader file into array of uint32_t. little endian assumed.
  auto fin = std::ifstream(filename, std::ios::binary);
  if (!fin.is_open()) {
    throw std::runtime_error(std::string("could not open file ") + filename);
  }
  auto code = std::vector<char>(std::istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>());
  // Padded by 0s to a boundary of 4.
  code.resize(4 * div_up(code.size(), size_t(4)));

  vk::ShaderModuleCreateFlags flags = vk::ShaderModuleCreateFlags();
  auto shader_module_create_info = vk::ShaderModuleCreateInfo(flags, code.size(),
    reinterpret_cast<uint32_t*>(code.data()));

  return device_.createShaderModule(shader_module_create_info);
}

} // namespace vky