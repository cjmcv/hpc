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

int main(int argc, char* argv[]) {
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
  
  vky::Allocator2<float> d_x = vky::Allocator2<float>::fromHost(x, width * height, executor->device(), selected_device_info.physical_device_);
  vky::Allocator2<float> d_y = vky::Allocator2<float>::fromHost(y, width * height, executor->device(), selected_device_info.physical_device_);

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
  buffers.push_back(d_y);
  buffers.push_back(d_x);

  PushParams params;
  params.width = width;
  params.height = height;
  params.a = a;

  int buffer_range = params.width * params.height * sizeof(float);
  for (int i = 0; i < 10; i++) {
    executor->Run(buffers, buffer_range, group_count_xyz, &params, sizeof(params));
  }

  printf("%f seconds\n", (double)(clock() - time) / CLOCKS_PER_SEC);

  float *out = new float[width*height];
  d_y.to_host(out, width*height);

  for (int i = 0; i < width * height; i++) {
    std::cout << out[i] << ", ";
  }

  // TODO: UnInitialize.

  delete[]x;
  delete[]y;
  delete[]out;

  // TODO: Check Release. 
  //       The reason may be related to the life cycle of vky::Allocator2.
  //delete devm;
  //delete executor;

  return 0;
}
