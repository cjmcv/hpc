#include <iostream>

#include "client.hpp"

int main(int argc, char* argv[]) {

  asio::io_context io_context;
  Client client(io_context, "localhost", "8080");
  io_context.run();

  while (1) {
    std::string func_name = "add";
    std::string A = "rpc hello world again";
    int B = 30;
    char C = 'c';

    client.Call(func_name, A, B, C);
    Sleep(1000);
  }

  system("pause");
  return 0;
}