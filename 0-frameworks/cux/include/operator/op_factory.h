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
  typedef Operator *(*OpCreator)(OpAssistor *assistor, std::string &params_str);
public:
  ~OpFactory() {};

  Operator *CreateOpByType(std::string type, OpAssistor *assistor, std::string param_str) {
    std::map<std::string, OpCreator>::iterator it = op_creator_map_.find(type);
    if (it == op_creator_map_.end()) {
      CUXLOG_ERR("Can not find Op: %s.\n Registered Op: < %s>", 
        type.c_str(), PrintList().c_str());
      return nullptr;
    }

    OpCreator creator = it->second;
    if (!creator)
      return nullptr;

    return creator(assistor, param_str);
  }

  // Registerer, set the mapping relation between operator's class name and it's specific pointer function.
  void RegisterOpClass(std::string type, OpCreator creator) {
    if (op_creator_map_.count(type) != 0) {
      CUXLOG_WARN("Op type: %s has already been registered.", type.c_str());
      return;
    }
    op_creator_map_[type] = creator;
  }

  // Show which ops have been registered.
  std::string PrintList() {
    std::string out = "";
    std::map<std::string, OpCreator>::iterator it = op_creator_map_.begin();
    while (it != op_creator_map_.end()) {
      out += it->first;
      out += " ";
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
