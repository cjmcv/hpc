/*!
* \brief  A factory for operators.
*         allows users to create an operator using only string.
*/

#ifndef CUX_OP_FACTORY_H_
#define CUX_OP_FACTORY_H_

#include <iostream>
#include <string>
#include <map>

#include "operator.h"

namespace cux { 

class OpFactory {
  // A function pointer.
  typedef Operator *(*OpCreator)(std::string &params_str);
public:
  ~OpFactory() {};

  Operator *CreateOpByType(std::string type, std::string param_str) {
    std::map<std::string, OpCreator>::iterator it = op_creator_map_.find(type);
    if (it == op_creator_map_.end())
      return nullptr;

    OpCreator creator = it->second;
    if (!creator)
      return nullptr;

    return creator(param_str);
  }

  // Registerer, set the mapping relation between operator's class name and it's specific pointer function.
  int RegisterOpClass(std::string type, OpCreator creator) {
    if (op_creator_map_.count(type) != 0) {
      CUXLOG_WARN("Op type: %s already registered.", type.c_str());
      return -1;
    }
    op_creator_map_[type] = creator;
    return 0;
  }

  // Show which ops have been registered.
  std::string PrintList() {
    std::string out = "";
    std::map<std::string, OpCreator>::iterator it = op_creator_map_.begin();
    while (it != op_creator_map_.end()) {
      out += it->first;
      out += "  ";
      it++;
    }
    return out;
  }

  // Singleton mode. Only one OpFactory exist.
  static OpFactory& GetInstance() {
    static OpFactory factory;
    return factory;
  }

private:
  OpFactory() {};
  // <name, creater>
  std::map<std::string, OpCreator> op_creator_map_;
};

}	//namespace cux
#endif //CUX_OP_FACTORY_H_
