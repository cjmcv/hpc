#ifndef HCS_PARAMS_H_
#define HCS_PARAMS_H_

#include "graph.hpp"

namespace hcs {

enum ParamsMode {
  PARAMS_IF,
  PARAMS_CXII
};

struct ParamsIF : IOParams {
  ParamsIF() :IOParams(PARAMS_IF) {}

  int i;
  float j;
};

struct ParamsCxII : IOParams {
  ParamsCxII() :IOParams(PARAMS_CXII) {}

  char *data;
  int height;
  int width;
};

class Assistor {
public:
  static int GetInput(hcs::Node &node, hcs::IOParams *input) {
    if (node.out_ == nullptr || input == nullptr) {
      std::cout << "GetInput -> node.out_ == nullptr || input == nullptr." << std::endl;
      return 0;
    }
    if (node.out_->struct_id != input->struct_id) {
      std::cout << "GetInput -> The struct_id of input is not the same as output's." << std::endl;
      return 0;
    }

    switch (node.out_->struct_id) {
    case hcs::PARAMS_IF:
      ((hcs::ParamsIF *)input)->i = ((hcs::ParamsIF *)(node.out_))->i;
      ((hcs::ParamsIF *)input)->j = ((hcs::ParamsIF *)(node.out_))->j;
      ((hcs::ParamsIF *)input)->obj_id = ((hcs::ParamsIF *)(node.out_))->obj_id;
      return 1;
    case hcs::PARAMS_CXII:
      ((hcs::ParamsCxII *)input)->data = ((hcs::ParamsCxII *)(node.out_))->data;
      ((hcs::ParamsCxII *)input)->height = ((hcs::ParamsCxII *)(node.out_))->height;
      ((hcs::ParamsCxII *)input)->width = ((hcs::ParamsCxII *)(node.out_))->width;
      ((hcs::ParamsCxII *)input)->obj_id = ((hcs::ParamsCxII *)(node.out_))->obj_id;
      return 1;
    default:
      std::cout << "GetInput -> Do not support mode " << node.out_->struct_id << std::endl;
      return 0;
    }
  }

  static int SetOutput(hcs::IOParams *input, hcs::IOParams **output) {
    if (input == nullptr) {
      std::cout << "SetOutput -> input == nullptr." << std::endl;
      return 0;
    }
    if (*output != nullptr && input->struct_id != (*output)->struct_id) {
      std::cout << "SetOutput -> The struct_id of input is not the same as output's." << std::endl;
      return 0;
    }

    switch (input->struct_id) {
    case hcs::PARAMS_IF:
      if (*output == nullptr) {
        *output = new hcs::ParamsIF();
      }
      ((hcs::ParamsIF *)(*output))->i = ((hcs::ParamsIF *)(input))->i;
      ((hcs::ParamsIF *)(*output))->j = ((hcs::ParamsIF *)(input))->j;
      ((hcs::ParamsIF *)(*output))->obj_id = ((hcs::ParamsIF *)(input))->obj_id;
      return 1;
    case hcs::PARAMS_CXII:
      if (*output == nullptr) {
        *output = new hcs::ParamsCxII();
      }
      ((hcs::ParamsCxII *)(*output))->data = ((hcs::ParamsCxII *)(input))->data;
      ((hcs::ParamsCxII *)(*output))->height = ((hcs::ParamsCxII *)(input))->height;
      ((hcs::ParamsCxII *)(*output))->width = ((hcs::ParamsCxII *)(input))->width;
      ((hcs::ParamsCxII *)(*output))->obj_id = ((hcs::ParamsCxII *)(input))->obj_id;
      return 1;
    default:
      std::cout << "SetOutput -> Do not support mode " << input->struct_id << std::endl;
      return 0;
    }
  }

};

}  // end of namespace hcs.

#endif //HCS_PARAMS_H_