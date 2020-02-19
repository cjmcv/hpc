#include <iostream>
#include "server.h"

int main(int argc, char* argv[]) {
  try {
    asio::io_context io_context;
    Server server(io_context, 8080);

    MRPC_BIND<double, double>("add",
      [](double a, double b) {
          printf("add = %f\n", a+b); 
        });
    MRPC_BIND<int>("mul",
      [](int a) {
      printf("mul = %d\n", a * a);
    });

    io_context.run();
    std::cout << "pass" << std::endl;
  }
  catch (std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  system("pause");

  return 0;
}

//
//int main(int argc, char* argv[]) {
//  Processor Proc;
//
//  std::function<void(double, double)> func = [](double a, double b) { 
//    double c = a + b; 
//    printf("res = %f\n", c); 
//    return;
//  };
//  Proc.Bind<double, double>("add", func);
//
//  std::function<void(long long)> func2 = [](long long a) {
//    double d = a *2;
//    printf("res2 = %f\n", d);
//    return;
//  };
//  Proc.Bind<long long>("mul", func2);
//
//  Proc.Run("add");
//  Proc.Run("mul");
//
//  //manager.Bind("sub", []() { return 3.0; });
//
//  //UnionRet ret = manager.Run("sub");
//  //std::cout << ret.dval << std::endl;
//
//  system("pause");
//
//  return 0;
//}