#include <iostream>
#include "server.h"

double Add(double a, double b) {
  return a + b;
}

int main(int argc, char* argv[]) {
  try {
    asio::io_context io_context;
    mrpc::Server server(io_context, 8080);
    server.Bind<double, double, double>("add", Add);
    server.Bind<int, int>("mul",
      [](int a) -> int { return a * a; });

    io_context.run();
    std::cout << "pass" << std::endl;
  }
  catch (std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  system("pause");

  return 0;
}