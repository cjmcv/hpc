#include <iostream>
#include "server.hpp"

int main(int argc, char* argv[]) {
  try {
    asio::io_context io_context;
    Server server(io_context, 8080);
    io_context.run();
    std::cout << "pass" << std::endl;
  }
  catch (std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  system("pause");

  return 0;
}