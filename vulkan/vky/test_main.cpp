#include "executor.h"
#include "allocator.h"

#include <time.h>

// TODO: Remove it.
// C++ mirror of the shader push constants interface
struct PushParams {
  uint32_t width;  //< frame width
  uint32_t height; //< frame height
  float a;         //< saxpy (\$ y = y + ax \$) scaling factor
};

int TestExecutor() {
  const int width = 30;
  const int height = 30;
  const float a = 2.0f; // saxpy scaling factor

  float *x = new float[width*height];
  float *y = new float[width*height];
  for (int i = 0; i < width*height; i++) {
    x[i] = 2.5f;
    y[i] = 1.3f;
  }
  vky::DeviceManager *devm = new vky::DeviceManager();
  devm->Initialize(true);
  devm->PrintDevicesInfo();

  int physical_device_id = 0;
  std::string shaders_dir_path = "D:/projects/github/hpc/vulkan/vky/";

  vky::DeviceInfo selected_device_info = devm->device_info(physical_device_id);

  vky::Executor *executor = new vky::Executor();
  executor->Initialize(selected_device_info, shaders_dir_path);

  vky::Allocator *allocator = executor->allocator();
  vky::VkyData vdata_x(allocator, width*height, x);
  vky::VkyData vdata_y(allocator, width*height, y);

  clock_t time = clock();

  // The order must be the same as defined in comp.
  int group_count_xyz[3];
  //group_count_xyz[0] = (m.w + pipeline->local_size_x - 1) / pipeline->local_size_x;
  //group_count_xyz[1] = (m.h + pipeline->local_size_y - 1) / pipeline->local_size_y;
  //group_count_xyz[2] = (m.c + pipeline->local_size_z - 1) / pipeline->local_size_z;
  group_count_xyz[0] = vky::div_up(width, 16); // TODO: WORKGROUP_SIZE = 16, has been defined in executor.h.
  group_count_xyz[1] = vky::div_up(height, 16);
  group_count_xyz[2] = 1;

  std::vector<vk::Buffer> buffers;
  buffers.push_back(vdata_y.get_device_data()->buffer_);
  buffers.push_back(vdata_x.get_device_data()->buffer_);

  PushParams params;
  params.width = width;
  params.height = height;
  params.a = a;

  int buffer_range = params.width * params.height * sizeof(float);
  for (int i = 0; i < 10; i++) {
    executor->Run(buffers, buffer_range, group_count_xyz, &params, sizeof(params));
  }

  printf("%f seconds\n", (double)(clock() - time) / CLOCKS_PER_SEC);


  float *vdata_cpu = vdata_y.get_host_data();

  for (int i = 0; i < width * height; i++) {
    std::cout << vdata_cpu[i] << ", ";
  }

  // TODO: UnInitialize.

  delete[]x;
  delete[]y;

  // TODO: Check Release. 
  //       The reason may be related to the life cycle of vky::Allocator2.
  //delete devm;
  //delete executor;

  return 0;
}

void TestVkyData() {
  int len = 100;
  float *x = new float[len];
  for (int i = 0; i < len; i++) {
    x[i] = i;
  }

  vky::DeviceManager *devm = new vky::DeviceManager();
  devm->Initialize(true);
  devm->PrintDevicesInfo();

  int physical_device_id = 0;
  std::string shaders_dir_path = "D:/projects/github/hpc/vulkan/vky/";

  vky::DeviceInfo selected_device_info = devm->device_info(physical_device_id);

  vky::Executor *executor = new vky::Executor();
  executor->Initialize(selected_device_info, shaders_dir_path);

  // TODO.
  vky::Allocator *allocator = executor->allocator();
  vky::VkyData vdata(allocator, len, x);

  float *vdata_host = vdata.host_data();
  vky::BufferMemory *data_device = vdata.get_device_data();

  memset(vdata_host, 0, sizeof(float) * len);
  for (int i = 0; i < len; i++) {
    std::cout << vdata_host[i] << ", ";
  }
  std::cout << std::endl << "cleaned." << std::endl;

  vdata_host = vdata.get_host_data();
  for (int i = 0; i < len; i++) {
    std::cout << vdata_host[i] << ", ";
  }
}

int main(int argc, char* argv[]) {

  //TestExecutor();

  TestVkyData();

  return 0;
}
