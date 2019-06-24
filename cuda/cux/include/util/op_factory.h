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
  typedef Operator *(*OpCreator)(std::string &params_str);
public:
  ~OpFactory() {};

  Operator *CreateOpByType(std::string type, std::string param_str) {
    typename std::map<std::string, OpCreator>::iterator it = op_creator_map_.find(type);
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
      std::cout << "Op type :" << type << " already registered.";
      return -1;
    }
    op_creator_map_[type] = creator;
    return 0;
  }

  // Singleton mode. Only one OpFactory exist.
  static OpFactory& GetInstance() {
    static OpFactory factory;
    return factory;
  }

private:
  OpFactory() {};
  typename std::map<std::string, OpCreator> op_creator_map_;
};

}	//namespace cux
#endif //CUX_OP_FACTORY_H_
