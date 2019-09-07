#ifndef HCS_PARAMS_H_
#define HCS_PARAMS_H_

namespace hcs {

struct IOParams;

enum ParamsMode {
  PARAMS_IF,
  PARAMS_CXII
};

struct ParamsIF : IOParams {
  ParamsIF() :IOParams(PARAMS_IF) {}

  int i;
  float f;
};

struct ParamsCxII : IOParams {
  ParamsCxII() :IOParams(PARAMS_CXII) {}

  char *cx;
  int i1;
  int i2;
};

class Assistor {
public:
  static int GetInput(hcs::IOParams *former_output, hcs::IOParams *input) {
    if (former_output == nullptr || input == nullptr) {
      std::cout << "GetInput -> node.out_ == nullptr || input == nullptr." << std::endl;
      return 0;
    }
    if (former_output->struct_id != input->struct_id) {
      std::cout << "GetInput -> The struct_id of input is not the same as output's." << std::endl;
      return 0;
    }

    switch (former_output->struct_id) {
    case hcs::PARAMS_IF:
      *(hcs::ParamsIF *)input = *(hcs::ParamsIF *)former_output;
      return 1;
    case hcs::PARAMS_CXII:
      *(hcs::ParamsCxII *)input = *(hcs::ParamsCxII *)former_output;
      return 1;
    default:
      std::cout << "GetInput -> Do not support mode " << former_output->struct_id << std::endl;
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
      *(hcs::ParamsIF *)(*output) = *(hcs::ParamsIF *)input;
      return 1;
    case hcs::PARAMS_CXII:
      if (*output == nullptr) {
        *output = new hcs::ParamsCxII();
      }
      *(hcs::ParamsCxII *)(*output) = *(hcs::ParamsCxII *)(input);
      return 1;
    default:
      std::cout << "SetOutput -> Do not support mode " << input->struct_id << std::endl;
      return 0;
    }
  }
};

}  // namespace hcs.

#endif // HCS_PARAMS_H_