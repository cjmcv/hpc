////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
////////////////////////////////////////////////////////////////

#ifndef VKY_OP_HUB_H_
#define VKY_OP_HUB_H_

#include <iostream>
#include <vector>
#include <string>
#include <map>

namespace vky { 

class OpParams {
public:
  int sub_op_count_;
};

//////////////////////////////////////
// NormalOpParams: object accessible.
class NormalOpParams :public OpParams {
public:
  NormalOpParams() { sub_op_count_ = 1; };
  NormalOpParams(
    std::string name,
    std::string shader_file,
    uint32_t buffer_count,
    uint32_t push_constant_count,
    uint32_t local_width,
    uint32_t local_height,
    uint32_t local_channels,
    uint32_t group_depends_id) :
    name_(name),
    shader_file_(shader_file),
    buffer_count_(buffer_count),
    push_constant_count_(push_constant_count),
    local_width_(local_width),
    local_height_(local_height),
    local_channels_(local_channels),
    group_depends_id_(group_depends_id) {
    sub_op_count_ = 1;
  };

  std::string name_;
  std::string shader_file_;

  uint32_t buffer_count_;
  uint32_t push_constant_count_;

  uint32_t local_width_;
  uint32_t local_height_;
  uint32_t local_channels_;
  uint32_t group_depends_id_;
};
//////////////////////////////////////
// HybridOpParams: object accessible.
class HybridOpParams :public OpParams {
public:
  HybridOpParams(std::string name, std::vector<NormalOpParams *> &op_params)
  : name_(name){
    op_params_.assign(op_params.begin(), op_params.end());
    sub_op_count_ = op_params_.size();
  }

public:
  std::string name_;
  std::vector<NormalOpParams *> op_params_;
};

//
class OpHub {
public:
  OpParams *GetOpParamsByName(std::string name) {
    std::map<std::string, OpParams*>::iterator iter = op_name_params_.find(name);
    if (iter == op_name_params_.end()) {
      PrintNameList();
      throw std::runtime_error(std::string("could not find op name: ") + name);
    }
    return iter->second;
  }

  void PrintNameList() {
    std::map<std::string, OpParams*>::iterator iter = op_name_params_.begin();
    std::cout << "Registered: ";
    while (iter != op_name_params_.end()) {
      std::cout << iter->first << ", ";
      iter++;
    }
    std::cout << std::endl;
  }

  // Singleton mode. Only one instance exist.
  static OpHub& GetInstance() {
    static OpHub ins;
    return ins;
  }

private:
  // TODO: Use callback function to register new op by user.(Only for NormalOp).
  OpHub() {
    // saxpy.
    NormalOpParams *saxpy = new NormalOpParams("saxpy", "saxpy.spv", 2, 3,
      32, 32, 1, 0);

    // add.
    NormalOpParams *add = new NormalOpParams("add", "add.spv", 2, 2,
      32, 32, 1, 0);

    // saxpy_add
    std::vector<NormalOpParams *> saxpy_add_in;
    saxpy_add_in.push_back(saxpy);
    saxpy_add_in.push_back(add);
    HybridOpParams *saxpy_add = new HybridOpParams("saxpy_add", saxpy_add_in);

    ///
    op_name_params_["saxpy"] = saxpy;
    op_name_params_["add"] = add;
    op_name_params_["saxpy_add"] = saxpy_add;
  }

  ~OpHub() {
    std::map<std::string, vky::OpParams *>::iterator iter_op = op_name_params_.begin();
    while (iter_op != op_name_params_.end()) {
      delete iter_op->second;
      iter_op++;
    }
  }

  std::map<std::string, OpParams*> op_name_params_;
}; // class OpHub

}	//namespace vky
#endif VKY_OP_HUB_H_
