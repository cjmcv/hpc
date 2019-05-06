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

  int physical_device_id = 0;
  std::string shaders_dir_path = "D:/projects/github/hpc/vulkan/vky/";

  vky::Executor *executor = new vky::Executor();
  executor->Initialize(devm->physical_device(physical_device_id), shaders_dir_path);
  
  vky::Allocator<float> d_x = vky::Allocator<float>::fromHost(x, width * height, executor->device(), executor->physical_device());
  vky::Allocator<float> d_y = vky::Allocator<float>::fromHost(y, width * height, executor->device(), executor->physical_device());

  clock_t time = clock();
 
  // The order must be the same as defined in comp.
  int group_xyz[3];
  group_xyz[0] = vky::div_up(width, 16); // TODO: WORKGROUP_SIZE = 16, has been defined in executor.h.
  group_xyz[1] = vky::div_up(height, 16);
  group_xyz[2] = 1;

  std::vector<vk::Buffer> buffers;
  buffers.push_back(d_y);
  buffers.push_back(d_x);

  PushParams params;
  params.width = width;
  params.height = height;
  params.a = a;

  int buffer_range = params.width * params.height * sizeof(float);
  for (int i = 0; i < 10; i++) {
    executor->Run(buffers, buffer_range, group_xyz, &params, sizeof(params));
  }

  printf("%f seconds\n", (double)(clock() - time) / CLOCKS_PER_SEC);

  float *out = new float[width*height];
  d_y.to_host(out, width*height);

  for (int i = 0; i < width * height; i++) {
    std::cout << out[i] << ", ";
  }

  delete[]x;
  delete[]y;
  delete[]out;

   return 0;
}
