/////////////////////////////////////////
// A simple demo for client.
// Protocol:
//  Send: "function name", arg1, arg2...
//  Receive: "function name", ret arg

#include <iostream>
#include "client.h"

int main(int argc, char* argv[]) {

  std::string host = "localhost";
  std::string service = "8080";
  asio::io_context io_context;
  mrpc::Client client(io_context);

  try {
    client.Connect(host, service);
    io_context.run();

    std::cout << "Connected to Port " << service << " of " << host << std::endl;
    while (1) {
      {
        float A = 30.1;
        float B = 21.2;
        auto ret = client.Call<float>("add", A, B);
        std::cout << "client.Call: A + B = " << A << " + " << B << " = " 
          << ret.value << ". Info: " << ret.error_str << std::endl;
      }
      //
      {
        int A = 15;
        int B = 12;
        auto ret = client.Call<int>("multiply", A, B);
        std::cout << "Call: A * B = " << A << " * " << B << " = " 
          << ret.value << ". Info: " << ret.error_str << std::endl;
      }
      Sleep(1000);
    }
  }
  catch (std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  system("pause");
  return 0;
}