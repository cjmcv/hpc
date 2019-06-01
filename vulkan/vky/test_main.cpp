#include "executor.h"
#include "allocator.h"

#include <time.h>

// TODO: Remove it.
// C++ mirror of the shader push constants interface
struct PushParamsSaxpy {
  uint32_t width;  //< frame width
  uint32_t height; //< frame height
  float a;         //< saxpy (\$ y = y + ax \$) scaling factor
};
struct PushParamsAdd {
  uint32_t width;  //< frame width
  uint32_t height; //< frame height
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
  std::string shaders_dir_path = "D:/projects/github/hpc/vulkan/vky/shaders/";

  vky::DeviceInfo *selected_device_info = devm->device_info(physical_device_id);

  vky::Executor *executor = new vky::Executor();
  executor->Initialize(selected_device_info, shaders_dir_path);

  vky::Allocator *allocator = executor->allocator();
  vky::VkyData *vdata_x = new vky::VkyData(allocator, width*height, x);
  vky::VkyData *vdata_y = new vky::VkyData(allocator, width*height, y);

  std::vector<vky::BufferMemory *> buffer_mems;
  buffer_mems.push_back(vdata_y->get_device_data());
  buffer_mems.push_back(vdata_x->get_device_data());

  // Saxpy;
  //{
  //  PushParamsSaxpy params;
  //  params.width = width;
  //  params.height = height;
  //  params.a = a;

  //  // Warm up.
  //  executor->Run("saxpy", buffer_mems, &params, sizeof(params)); 
  //
  //  clock_t time = clock();
  //  for (int i = 0; i < 2; i++) {
  //    executor->Run("saxpy", buffer_mems, &params, sizeof(params));
  //  }
  //  printf("%f seconds\n", (double)(clock() - time) / CLOCKS_PER_SEC);
  //}
  // Add.
  {
    PushParamsAdd params;
    params.width = width;
    params.height = height;

    // Warm up.
    executor->Run("add", buffer_mems, &params, sizeof(params));

    //clock_t time = clock();
    //for (int i = 0; i < 2; i++) {
    //  executor->Run("add", buffer_mems, &params, sizeof(params));
    //}
    //printf("%f seconds\n", (double)(clock() - time) / CLOCKS_PER_SEC);
  }

  float *vdata_cpu = vdata_y->get_host_data();

  for (int i = 0; i < width * height; i++) {
    std::cout << vdata_cpu[i] << ", ";
  }

  // Release.
  delete[]x;
  delete[]y;

  delete vdata_x;
  delete vdata_y;

  executor->UnInitialize();
  delete executor;

  devm->UnInitialize();
  delete devm;

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

  vky::DeviceInfo *selected_device_info = devm->device_info(physical_device_id);

  vky::Executor *executor = new vky::Executor();
  executor->Initialize(selected_device_info, shaders_dir_path);

  vky::Allocator *allocator = executor->allocator();
  vky::VkyData *vdata = new vky::VkyData(allocator, len, x);

  float *vdata_host = vdata->host_data();
  vky::BufferMemory *data_device = vdata->get_device_data();

  memset(vdata_host, 0, sizeof(float) * len);
  for (int i = 0; i < len; i++) {
    std::cout << vdata_host[i] << ", ";
  }
  std::cout << std::endl << "cleaned." << std::endl;

  vdata_host = vdata->get_host_data();
  for (int i = 0; i < len; i++) {
    std::cout << vdata_host[i] << ", ";
  }

  // Release
  delete[]x;
  delete vdata;

  executor->UnInitialize();
  delete executor;

  devm->UnInitialize();
  delete devm;
}

int main(int argc, char* argv[]) {

  TestExecutor();

  //TestVkyData();

  return 0;
}
