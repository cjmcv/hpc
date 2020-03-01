#include "server.h"

double Add(double a, double b) {
  return a + b;
}

int main(int argc, char* argv[]) {
  short port = 8080;
  asio::io_context io_context;
  mrpc::Server server(io_context, port);
  // Bind functions.
  server.Bind<float, float, float>("add", Add);
  server.Bind<int, int, int>("multiply",
    [](int a, int b) -> int { return a * b; });

  std::cout << "Initialise an IPv4 TCP endpoint for port " << port << std::endl;
  std::cout << "Running..." << std::endl;
  io_context.run();

  return 0;
}