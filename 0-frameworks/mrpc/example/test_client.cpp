#include <iostream>

#include "client.h"

int main(int argc, char* argv[]) {

  // Send: "function name", arg1, arg2...
  // Receive: "function name", ret arg
  // Receive-Failed: "NotFound"

  asio::io_context io_context;
  mrpc::Client client(io_context, "localhost", "8080");
  io_context.run();

  while (1) {
    {
      std::string func_name = "add";
      double A = 30.1;
      double B = 21.2;
      double ret = client.Call<double>(func_name, A, B);
      std::cout << "client.Call add: " << ret << std::endl;
    }
    //
    {
      std::string func_name = "mul";
      int C = 15;
      int ret = client.Call<int>(func_name, C);
      std::cout << "client.Call mul: " << ret << std::endl;
    }

    Sleep(1000);
  }

  system("pause");
  return 0;
}