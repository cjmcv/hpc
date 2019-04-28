#include "executor.hpp"
#include "allocator.hpp"

auto main(int argc, char* argv[])-> int {
	const int width = 90;
	const int height = 60;
  const float a = 2.0f; // saxpy scaling factor

  float *y = new float[width*height];
  float *x = new float[width*height];
  for (int i = 0; i < width*height; i++) {
    x[i] = 2.5f;
    y[i] = 1.2f;
  }

  //// 父与子
  // // 父绑定设备，创建pipline。
  // // 子负责内存绑定与拷贝，内不管控gpu内存和cpu内存。
  // // 每个子对应一对内存。
  Executor f("src/shaders/saxpy.spv");
  auto d_y = vuh::Allocator<float>::fromHost(y, width*height, f.device_, f.phys_devices_[0]);
  auto d_x = vuh::Allocator<float>::fromHost(x, width*height, f.device_, f.phys_devices_[0]);

  f.execute(d_y, d_x, { width, height, a });

  float *out_tst = new float[width*height];
  d_y.to_host(out_tst, width*height); // and now out_tst should contain the result of saxpy (y = y + ax)

  for (int i = 0; i < width*height; i++) {
    std::cout << out_tst[i] << ", ";
  }



   return 0;
}
