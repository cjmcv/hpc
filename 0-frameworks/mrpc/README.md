# mrpc

Mini-RPC based on asio.

---

## Dependencies - Asio Only

Asio is a cross-platform C++ library for network and low-level I/O programming that provides developers with a consistent asynchronous model using a modern C++ approach.
* [Main Page](https://think-async.com/Asio/)
* [Github](https://github.com/chriskohlhoff/asio)

## Build

```bash
cd hpc\0-frameworks\mrpc
mkdir windows
cd windows
cmake -G "Visual Studio 15 2017" -DCMAKE_GENERATOR_PLATFORM=x64 ..
```

## Tested environment

* win10 + vs2017

## Examples
[Source code](https://github.com/cjmcv/hpc/tree/master/0-frameworks/mrpc/example)  

**Server** 
```cpp
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
```

**Client** 
```cpp
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
```