#include "executor.h"
#include "allocator.h"

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

  vky::Executor *executor = new vky::Executor();
  executor->Initialize();
  
  auto d_x = vky::Allocator<float>::fromHost(x, width * height, executor->device(), executor->phys_device());
  auto d_y = vky::Allocator<float>::fromHost(y, width * height, executor->device(), executor->phys_device());

  executor->Run(d_x, { width, height, a }, d_y);

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
