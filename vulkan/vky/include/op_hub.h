////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
////////////////////////////////////////////////////////////////

#ifndef VKY_OP_HUB_H_
#define VKY_OP_HUB_H_

#include <iostream>
#include <string>
#include <map>

namespace vky { 

class OpParams {};

//////////////////////////////////////
// NormalOpParams: object accessible.
class NormalOpParams :public OpParams {
public:
  NormalOpParams() {};
  NormalOpParams(std::string shader,
    uint32_t buffer_count,
    uint32_t push_constant_count,
    uint32_t local_width,
    uint32_t local_height,
    uint32_t local_channels,
    uint32_t group_depends_id) :
    shader_(shader),
    buffer_count_(buffer_count),
    push_constant_count_(push_constant_count),
    local_width_(local_width),
    local_height_(local_height),
    local_channels_(local_channels),
    group_depends_id_(group_depends_id) {};

  std::string shader_;

  uint32_t buffer_count_;
  uint32_t push_constant_count_;

  uint32_t local_width_;
  uint32_t local_height_;
  uint32_t local_channels_;
  uint32_t group_depends_id_;
};

//
class OpHub {
public:
  OpParams *GetOpParamsByName(std::string name) {
    std::map<std::string, OpParams*>::iterator iter = op_name_params_.find(name);
    if (iter == op_name_params_.end()) {
      throw std::runtime_error(std::string("could not find shader name: ") + name);
    }
    return iter->second;
  }
  void PrintNameList() {
    std::map<std::string, OpParams*>::iterator iter = op_name_params_.begin();
    while (iter != op_name_params_.end()) {
      std::cout << iter->first << ", ";
    }
  }

  // Singleton mode. Only one instance exist.
  static OpHub& GetInstance() {
    static OpHub ins;
    return ins;
  }

private:
  OpHub() {
    op_name_params_["saxpy"] = new NormalOpParams("saxpy", 2, 3,
      32, 32, 1, 0);
    op_name_params_["add"] = new NormalOpParams("add", 2, 2,
      32, 32, 1, 0);
  }

  std::map<std::string, OpParams *> op_name_params_;

}; // class OpHub

}	//namespace dlex_cnn
#endif VKY_OP_HUB_H_
