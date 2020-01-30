#include <iostream>

#include "client.hpp"

int main(int argc, char* argv[]) {

  asio::io_context io_context;
  Client client(io_context, "localhost", "8080");
  io_context.run();

  while (1) {
    client.Call();
    Sleep(1000);
  }

  system("pause");
  return 0;
}