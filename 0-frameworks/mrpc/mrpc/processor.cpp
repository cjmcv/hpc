#include "processor.h"

namespace mrpc {

Processor::Processor() {
  items_.clear();
}

Processor::~Processor() {
  std::map<std::string, Item* >::iterator iter;
  for (iter = items_.begin(); iter != items_.end(); iter++) {
    delete iter->second;
  }
}

void Processor::Run(RpcMessage &message) {
  std::string func_name;
  message.GetFuncName(func_name);
    
  Item *item = items_[func_name];
  if (item == nullptr) {
    std::string msg = "Can not find the function [" + func_name + "]";
    message.Pack(msg);
  }
  else {
    std::cout << "Apply function: " << func_name << std::endl;
    item->Apply(message);
  }
}

} // namespace mrpc
