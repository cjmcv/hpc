////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  A factory for operators.
//          allows users to create an operator only by string.
////////////////////////////////////////////////////////////////

#ifndef VKY_OP_FACTORY_H_
#define VKY_OP_FACTORY_H_

#include <iostream>
#include <string>
#include <map>
#include <memory> 

#include "operator.h"

namespace vky { 

#define GET_OP_PARAM_CLASS(type, param)                                    \
    std::shared_ptr< vky::Operator >(new vky::##type##Op(param));

class OpFactory {
public:
  OpFactory(const vk::Device device,
            const uint32_t *max_workgroup_size,
            const uint32_t max_workgroup_invocations,
            const std::string &shaders_dir)
  : device_(device),
    max_workgroup_size_(max_workgroup_size), 
    max_workgroup_invocations_(max_workgroup_invocations),
    shaders_dir_(shaders_dir){};

  ~OpFactory() {
    std::map<std::string, vk::ShaderModule>::iterator iter_shader = shaders_name_obj_.begin();
    while (iter_shader != shaders_name_obj_.end()) {
      device_.destroyShaderModule(iter_shader->second);
    }

    std::map<std::string, vky::Operator *>::iterator iter_op = ops_list_.begin();
    while (iter_op != ops_list_.end()) {
      iter_op->second->UnInitialize();
      delete iter_op->second;
    }
  };

  // Create operator instance according to the name that has been registered.
  vky::Operator *GetOpByName(std::string name) {
    typename std::map<std::string, Operator*>::iterator it = ops_list_.find(name);
    if (it != ops_list_.end())
      return it->second;

    NormalOpParams *op_params = (NormalOpParams *)OpHub::GetInstance().GetOpParamsByName(name);
    NormalOp *op = new NormalOp();
    std::string shader_file_path = shaders_dir_ + op_params->shader_file_;
    op->Initialize(device_, max_workgroup_size_, max_workgroup_invocations_, shader(name, op_params->shader_file_));

    ops_list_[name] = op;

    return op;
  }

private:
  vk::ShaderModule shader(const std::string &name, const std::string &file) {
    std::map<std::string, vk::ShaderModule>::iterator iter_shader = shaders_name_obj_.find(name);
    if (iter_shader == shaders_name_obj_.end()) {
      std::string shader_path = shaders_dir_ + file;
      vk::ShaderModule shader = CreateShaderModule(shader_path);
      shaders_name_obj_[name] = shader;
      return shader;
    }
    return iter_shader->second;
  }

  inline uint32_t div_up(uint32_t x, uint32_t y) { return (x + y - 1u) / y; }
  vk::ShaderModule CreateShaderModule(const std::string &filename) {
    // Read binary shader file into array of uint32_t. little endian assumed.
    auto fin = std::ifstream(filename.c_str(), std::ios::binary);
    if (!fin.is_open()) {
      throw std::runtime_error(std::string("could not open file ") + filename.c_str());
    }
    auto code = std::vector<char>(std::istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>());
    // Padded by 0s to a boundary of 4.
    code.resize(4 * div_up(code.size(), size_t(4)));

    vk::ShaderModuleCreateFlags flags = vk::ShaderModuleCreateFlags();
    auto shader_module_create_info = vk::ShaderModuleCreateInfo(flags, code.size(),
      reinterpret_cast<uint32_t*>(code.data()));

    return device_.createShaderModule(shader_module_create_info);
  }

private:
  std::map<std::string, Operator*> ops_list_;
  std::map<std::string, vk::ShaderModule> shaders_name_obj_;

  const vk::Device device_;
  const uint32_t *max_workgroup_size_;
  const uint32_t max_workgroup_invocations_;
  const std::string shaders_dir_;
};


}	//namespace vky
#endif
