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
      float A = 30.1;
      float B = 21.2;
      float ret = client.Call<float>(func_name, A, B);
      std::cout << "client.Call: A + B = " << 
        A << " + " << B << " = " << ret << std::endl;
    }
    // 
    {
      std::string func_name = "multiply";
      int A = 15;
      int B = 12;
      int ret = client.Call<int>(func_name, A, B);
      std::cout << "client.Call: A * B = " << 
        A << " * " << B << " = " << ret << std::endl;
    }
    Sleep(1000);
  }

  system("pause");
  return 0;
}