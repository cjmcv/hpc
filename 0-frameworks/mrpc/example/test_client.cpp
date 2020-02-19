#include <iostream>

#include "client.h"

int main(int argc, char* argv[]) {

  asio::io_context io_context;
  Client client(io_context, "localhost", "8080");
  io_context.run();

  while (1) {
    std::string func_name = "add";
    double A = 30.1;
    double B = 21.2;

    client.Call(func_name, A, B);
    Sleep(1000);
  }

  system("pause");
  return 0;
}