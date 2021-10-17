#include "device.h"

namespace vky {

///////////////
// <Public
int DeviceManager::Initialize(bool is_enable_validation) {
  devices_info_ = nullptr;

  std::vector<const char*> layers = std::vector<const char*>{};
  std::vector<const char*> extensions = std::vector<const char*>{};
  if (is_enable_validation) {
    // "=" in vector is deep copy. 
    // Note: VK_LAYER_LUNARG_standard_validation is deprecated.
    layers = EnabledLayers({ "VK_LAYER_KHRONOS_validation" });
    // The same as VK_EXT_DEBUG_REPORT_EXTENSION_NAME
    extensions = EnabledExtensions({ "VK_EXT_debug_report" }); 
  }

  CreateInstance(layers, extensions);
  QueryDeviceInfo();

  return 0;
}

int DeviceManager::UnInitialize() {
  if (devices_info_ != nullptr) {
    delete[]devices_info_;
    devices_info_ = nullptr;
  }
  DestroyInstance();
  return 0;
}

void DeviceManager::PrintDevicesInfo() const {
  std::cout << "Number of devices: " << devices_count_ << std::endl;

  for (int i = 0; i < devices_count_; i++) {
    DeviceInfo &info = devices_info_[i];

    ///////////////////////////////////
    //  Information.
    std::cout << std::endl;
    std::cout << "////////////////device:" << i << "//////////////////" << std::endl;
    std::cout << "device name: " << info.device_name_ << std::endl;
    std::cout << "api version: " << info.api_version_ << std::endl;
    std::cout << "driver version: " << info.driver_version_ << std::endl;
    std::cout << "vendor id: " << info.vendor_id_ << std::endl;
    std::cout << "device id: " << info.device_id_ << std::endl;

    std::string type;
    if (info.type_ == vk::PhysicalDeviceType::eOther)
      type = "Other";
    else if(info.type_ == vk::PhysicalDeviceType::eIntegratedGpu)
      type = "IntegratedGpu";
    else if (info.type_ == vk::PhysicalDeviceType::eDiscreteGpu)
      type = "DiscreteGpu";
    else if (info.type_ == vk::PhysicalDeviceType::eVirtualGpu)
      type = "VirtualGpu";
    else if (info.type_ == vk::PhysicalDeviceType::eCpu)
      type = "Cpu";
    std::cout << "device type: " << type << std::endl;

    ///////////////////////////////////
    // Capability
    std::cout << "max compute shared memory size: " << info.max_shared_memory_size_ << std::endl;
    std::cout << "max workgroup count: [" << info.max_workgroup_count_[0]
                                  << ", " << info.max_workgroup_count_[1] 
                                  << ", " << info.max_workgroup_count_[2] << "]" << std::endl;
    std::cout << "max workgroup invocations: " << info.max_workgroup_invocations_ << std::endl;
    std::cout << "max workgroup size: [" << info.max_workgroup_size_[0]
                                 << ", " << info.max_workgroup_size_[1]
                                 << ", " << info.max_workgroup_size_[2] << "]" << std::endl;

    std::cout << "memory map alignment: " << info.memory_map_alignment_ << std::endl;
    std::cout << "buffer offset alignment: " << info.buffer_offset_alignment_ << std::endl;
    std::cout << "compute queue familly id: " << info.compute_queue_familly_id_ << std::endl;
  }

  std::cout << std::endl << "//////////////////////////////////////////" << std::endl;
}

///////////////
// <Private.
std::vector<const char*> DeviceManager::EnabledExtensions(const std::vector<const char*>& extensions) const {
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

std::vector<const char*> DeviceManager::EnabledLayers(const std::vector<const char*>& layers) const {
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

int DeviceManager::CreateInstance(std::vector<const char*> &layers,
  std::vector<const char*> &extensions) {

  auto app_info = vk::ApplicationInfo("Vulkan Compute Example", 0, "no_engine",
    0, VK_API_VERSION_1_0); // The only important field here is apiVersion
  auto create_info = vk::InstanceCreateInfo(vk::InstanceCreateFlags(), &app_info,
    layers.size(), layers.data(), 
    extensions.size(), extensions.data());

  instance_ = vk::createInstance(create_info);
  return 0;
}

int DeviceManager::DestroyInstance() {
  instance_.destroy();
  return 0;
}

uint32_t DeviceManager::GetComputeQueueFamilyId(const vk::PhysicalDevice& physical_device) const {
  auto queue_families = physical_device.getQueueFamilyProperties();

  // prefer using compute-only queue
  auto queue_it = std::find_if(queue_families.begin(), queue_families.end(), [](auto& f) {
    auto masked_flags = ~vk::QueueFlagBits::eSparseBinding & f.queueFlags; // ignore sparse binding flag 
    return 0 < f.queueCount                                               // queue family does have some queues in it
      && (vk::QueueFlagBits::eCompute & masked_flags)
      && !(vk::QueueFlagBits::eGraphics & masked_flags);
  });
  if (queue_it != end(queue_families)) {
    return uint32_t(std::distance(begin(queue_families), queue_it));
  }

  // otherwise use any queue that has compute flag set
  queue_it = std::find_if(queue_families.begin(), queue_families.end(), [](auto& f) {
    auto masked_flags = ~vk::QueueFlagBits::eSparseBinding & f.queueFlags;
    return 0 < f.queueCount && (vk::QueueFlagBits::eCompute & masked_flags);
  });
  if (queue_it != end(queue_families)) {
    return uint32_t(std::distance(begin(queue_families), queue_it));
  }

  throw std::runtime_error("could not find a queue family that supports compute operations");
}

int DeviceManager::QueryDeviceInfo() {

  instance_.enumeratePhysicalDevices(&devices_count_, nullptr);
  if (devices_count_ == 0) {
    throw std::runtime_error("could not find a device with vulkan support");
  }

  std::vector<vk::PhysicalDevice> physical_devices;
  physical_devices.resize(devices_count_);
  instance_.enumeratePhysicalDevices(&devices_count_, physical_devices.data());

  if (devices_info_ != nullptr) {
    delete[]devices_info_;
    devices_info_ = nullptr;
  }

  devices_info_ = new DeviceInfo[devices_count_];
  for (uint32_t i = 0; i < devices_count_; i++) {
    const vk::PhysicalDevice& physical_device = physical_devices[i];
    DeviceInfo &info = devices_info_[i];

    vk::PhysicalDeviceProperties device_properties;
    physical_device.getProperties(&device_properties);

    info.physical_device_ = physical_device;

    // info
    memcpy(info.device_name_, device_properties.deviceName, VK_MAX_PHYSICAL_DEVICE_NAME_SIZE * sizeof(char));
    info.api_version_ = device_properties.apiVersion;
    info.driver_version_ = device_properties.driverVersion;
    info.vendor_id_ = device_properties.vendorID;
    info.device_id_ = device_properties.deviceID;
    info.type_ = device_properties.deviceType;

    // capability
    info.max_shared_memory_size_ = device_properties.limits.maxComputeSharedMemorySize;

    info.max_workgroup_count_[0] = device_properties.limits.maxComputeWorkGroupCount[0];
    info.max_workgroup_count_[1] = device_properties.limits.maxComputeWorkGroupCount[1];
    info.max_workgroup_count_[2] = device_properties.limits.maxComputeWorkGroupCount[2];

    info.max_workgroup_invocations_ = device_properties.limits.maxComputeWorkGroupInvocations;

    info.max_workgroup_size_[0] = device_properties.limits.maxComputeWorkGroupSize[0];
    info.max_workgroup_size_[1] = device_properties.limits.maxComputeWorkGroupSize[1];
    info.max_workgroup_size_[2] = device_properties.limits.maxComputeWorkGroupSize[2];

    info.memory_map_alignment_ = device_properties.limits.minMemoryMapAlignment;
    info.buffer_offset_alignment_ = device_properties.limits.minStorageBufferOffsetAlignment;

    info.compute_queue_familly_id_ = GetComputeQueueFamilyId(physical_device);
  }
  return 0;
}

} // namespace vky