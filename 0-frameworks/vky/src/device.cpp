#include "device.h"

namespace vky {

///////////////
// <Public

int DeviceManager::QueryDeviceInfo(vk::Instance &instance) {

  uint32_t device_count;
  instance.enumeratePhysicalDevices(&device_count, nullptr);
  if (device_count == 0) {
    throw std::runtime_error("could not find a device with vulkan support");
  }

  std::vector<vk::PhysicalDevice> physical_devices;
  physical_devices.resize(device_count);
  instance.enumeratePhysicalDevices(&device_count, physical_devices.data());

  devices_info_.resize(device_count);
  for (uint32_t i = 0; i < device_count; i++) {
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

void DeviceManager::PrintDevicesInfo() const {
  std::cout << "Number of devices: " << devices_info_.size() << std::endl;

  for (int i = 0; i < devices_info_.size(); i++) {
    const DeviceInfo &info = devices_info_[i];

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

vk::Device DeviceManager::CreateLogicalDevice(int physical_device_id) {
  uint32_t compute_queue_familly_id = devices_info_[physical_device_id].compute_queue_familly_id_;
  vk::PhysicalDevice &physical_device = devices_info_[physical_device_id].physical_device_;

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

///////////////
// <Private.
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

} // namespace vky