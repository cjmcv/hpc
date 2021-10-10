#ifndef VKY_DEVICE_H_
#define VKY_DEVICE_H_

#include <iostream>
#include <vulkan/vulkan.hpp>

namespace vky {

class DeviceInfo {
public:
  vk::PhysicalDevice physical_device_;

  // info
  char device_name_[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
  uint32_t api_version_;
  uint32_t driver_version_;
  uint32_t vendor_id_;
  uint32_t device_id_;

  // eOther = VK_PHYSICAL_DEVICE_TYPE_OTHER,
  // eIntegratedGpu = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
  // eDiscreteGpu = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
  // eVirtualGpu = VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU,
  // eCpu = VK_PHYSICAL_DEVICE_TYPE_CPU
  vk::PhysicalDeviceType type_;

  // hardware capability
  uint32_t max_shared_memory_size_;
  uint32_t max_workgroup_count_[3];
  uint32_t max_workgroup_invocations_;
  uint32_t max_workgroup_size_[3];

  uint32_t memory_map_alignment_;
  uint32_t buffer_offset_alignment_;

  // runtime
  uint32_t compute_queue_familly_id_;
};

// TODO: 部分内容与device无关，应跟环境有关
class DeviceManager {
public:
  DeviceManager():devices_count_(0) {}
  ~DeviceManager() {}

  int device_count() const { return devices_count_; }
  vk::PhysicalDevice physical_device(int id) const { return devices_info_[id].physical_device_; }
  DeviceInfo *device_info(int id) const { return &devices_info_[id]; }

  int Initialize(bool is_enable_validation);
  int UnInitialize();

  void PrintDevicesInfo() const;

private: 
  // filter list of desired extensions to include only those supported by current Vulkan instance.
  std::vector<const char*> EnabledExtensions(const std::vector<const char*>& extensions) const;

  // filter list of desired layers to include only those supported by current Vulkan instance
  std::vector<const char*> EnabledLayers(const std::vector<const char*>& layers) const;

  // 
  int CreateInstance(std::vector<const char*> &layers, std::vector<const char*> &extensions);

  // @return the index of a queue family that supports compute operations.
  // Groups of queues that have the same capabilities (for instance, they all supports graphics
  // and computer operations), are grouped into queue families.
  // When submitting a command buffer, you must specify to which queue in the family you are submitting to.
  uint32_t GetComputeQueueFamilyId(const vk::PhysicalDevice& physical_device) const;

  int QueryPhysicalDevices();

private:
  vk::Instance instance_;

  uint32_t devices_count_;
  DeviceInfo *devices_info_;
}; // class DeviceManager

} // namespace vky

#endif  // VKY_DEVICE_H_