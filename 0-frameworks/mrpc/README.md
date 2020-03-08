# mrpc

Mini-RPC based on asio.

---

## 简介

一个小型的RPC框架，主要为了梳理RPC框架的一些技术细节，粗糙地造了一遍轮子。其中网络通信部分没写，使用了asio实现。

框架分为服务端和客户端，服务端注册函数，客户端发送函数名及其相关参数到服务端，调用服务端中已经注册了的函数，函数调用完成，由服务端将调用结果返回到客户端。

## 主要模块

1. Serializer：序列化器，服务端与客户端之间交互的数据均需要通过该类进行序列化和反序列化。发送端需将数据序列化打包发送，接收端将接收到的数据解包反序列化。

2. RpcMessage：用于管理需要传输的数据，里面使用了Serializer，封装了序列化和反序列化的一些操作，并管理发送数据的内存。

3. Processor：放置在Server中，用于注册函数、保存函数参数类型以及函数调用。以Server接收到的RpcMessage数据会放入Processor，由Processor进行参数解释与函数调用，并将调用结果重新打包成RpcMessage数据，给回到Server，再由Server发送回Client。

4. Client： 客户端基本操作接口，主要包含发送函数调用请求和接收函数调用结果。

5. Server：服务端基本操作接口，主要包含函数绑定注册和通信连接。

6. Session：在server中调用，主要包含与Client相对应的操作，即接收函数调用请求和发送函数调用结果。

## 依赖 - Asio

Asio is a cross-platform C++ library for network and low-level I/O programming that provides developers with a consistent asynchronous model using a modern C++ approach.
* [Main Page](https://think-async.com/Asio/)
* [Github](https://github.com/chriskohlhoff/asio)

## 工程构建

```bash
cd hpc\0-frameworks\mrpc
mkdir windows
cd windows
cmake -G "Visual Studio 15 2017" -DCMAKE_GENERATOR_PLATFORM=x64 ..
```

## 已测试环境

* win10 + vs2017

## Demo
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