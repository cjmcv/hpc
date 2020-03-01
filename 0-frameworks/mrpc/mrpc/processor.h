#ifndef MRPC_PROCESSOR_H_
#define MRPC_PROCESSOR_H_

#include <iostream>
#include <functional>
#include <tuple>
#include <map>

#include "message.h"

namespace mrpc {

template <typename T>
struct _identity {
  typedef T type;
};

class Item {
public:
  virtual ~Item() {}
  virtual void Apply(RpcMessage &params) = 0;
};

template<typename Response, typename... Args>
class DerivedItem : public Item {

public:
  DerivedItem(std::string &func_name, 
    typename _identity<std::function<Response(Args&...)>>::type func) {
    func_name_ = func_name;
    handle_ = new std::function<Response(Args&...)>(func);
  }
  ~DerivedItem() { delete handle_; }

  virtual void Apply(RpcMessage &params) {
    // Fill params.
    // note: std::apply and "fold expression" require c++17 support.
    std::apply([&params](auto&&... args) {
      ((ParamsRecover(params, args)), ...);
    }, request_);

    // Calculate.
    auto response = std::apply(*handle_, request_);
    params.Pack(func_name_, response);
  }

private:
  template<typename T>
  static void ParamsRecover(RpcMessage &params, T& t) {
    params.GetArgs(t);
  }

private:
  std::string func_name_;
  std::tuple<Args...> request_;
  std::function<Response(Args&...)> *handle_;
};

// Request & Response.
class Processor {

public:
  Processor();
  ~Processor();

  template<typename Response, typename... Args>
  void Bind(std::string func_name,
    typename _identity<std::function<Response(Args&...)>>::type func) {
    if (items_.find(func_name) != items_.end()) {
      printf("Duplicate: %s.", func_name.c_str());
      return;
    }
    items_[func_name] = new DerivedItem<Response, Args...>(func_name, func);
  }

  void Run(RpcMessage &message);

private:
  std::map<std::string, Item* > items_;
};

} // namespace mrpc

#endif // MRPC_PROCESSOR_H_