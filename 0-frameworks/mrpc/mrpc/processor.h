#ifndef MRPC_PROCESSOR_H_
#define MRPC_PROCESSOR_H_

#include <iostream>
#include <functional>
#include <tuple>
#include <map>

#include "message.h"

class ArgsRecorderBase {
public:
  virtual void Apply(void* functor, RpcMessage &params) = 0;
};

template<typename... Args>
class ArgsRecorder : public ArgsRecorderBase {

  using FuncT = std::function<void(Args&...)>;

public:
  template<typename T>
  static void ParamsRecover(RpcMessage &params, T& t) {
    params.GetArgs(t);
    std::string ts = typeid(T).name();
    //int id = typeid(T).hash_code();
    std::cout << "GetArgs" << t << "<" << ts << ">" << std::endl;
  }

  virtual void Apply(void* functor, RpcMessage &params) {
    // note: std::apply and "fold expression" require c++17 support.
    std::apply([&params](auto&&... args) {
      ((ParamsRecover(params, args)), ...);
    }, request_);

    FuncT* f = (FuncT*)(functor);
    std::apply(*f, request_);
  }

private:
  std::tuple<Args...> request_;
};

// TODO: 1. 封装Response作为统一输出。
// Request & Response.
class Processor {

  class Item {
  public:
    Item() {}
    Item(ArgsRecorderBase *rec, void* s, std::function<void(void)> t)
      : recorder_(rec), handler_(s), handler_delete_(t) {}
    void Execute(RpcMessage &params) {
      recorder_->Apply(handler_, params);
    }

  private:
    ArgsRecorderBase* recorder_;
    /** handler to handle a protocol */
    void* handler_;
    /** handler free function */
    std::function<void(void)> handler_delete_;
  };

  template <typename T>
  struct _identity {
    typedef T type;
  };

public:
  static Processor& Get() {
    static Processor instance;
    return instance;
  }

  template<typename... Args>
  void Bind(std::string func_name,
    typename _identity<std::function<void(Args&...)>>::type func) {

    if (items_.find(func_name) != items_.end()) {
      // Ignore it.
      printf("Duplicate: %s.", func_name.c_str());
      return;
    }

    using FuncT = std::function<void(Args&...)>;
    FuncT* fp = new FuncT(func);
    items_[func_name] =
      Item(new ArgsRecorder<Args...>(),
      (void*)fp,
        [=]() {delete fp; });
  }

  void Run(RpcMessage &message) {

    std::string func_name;
    message.GetFuncName(func_name);
    
    Item item = items_[func_name];
    item.Execute(message);
  }

private:
  std::map<std::string, Item > items_;
};

////////////////

#define MRPC_BIND Processor::Get().Bind

#endif // MRPC_PROCESSOR_H_