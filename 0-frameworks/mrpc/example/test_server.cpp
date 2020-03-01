#include "server.h"

double Add(double a, double b) {
  return a + b;
}

int main(int argc, char* argv[]) {

  asio::io_context io_context;
  mrpc::Server server(io_context, 8080);
  // Bind functions.
  server.Bind<float, float, float>("add", Add);
  server.Bind<int, int, int>("multiply",
    [](int a, int b) -> int { return a * b; });

  io_context.run();
  std::cout << "pass" << std::endl;

  system("pause");
  return 0;
}