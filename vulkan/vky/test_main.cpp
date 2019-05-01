#include "executor.h"
#include "allocator.h"

int main(int argc, char* argv[]) {
  const int len = 100;

  float *x = new float[len];
  for (int i = 0; i < len; i++) {
    x[i] = 2.5f;
  }

  for (int i = 0; i < len; i++) {
    std::cout << x[i] << ", ";
  }

   return 0;
}
