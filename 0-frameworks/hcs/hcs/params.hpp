#ifndef HCS_PARAMS_H_
#define HCS_PARAMS_H_

namespace hcs {

enum ParamsMode {
  PARAMS_IF = 0,
  PARAMS_FXII
};

struct IOParams {
  IOParams(int mode_id) :struct_id(mode_id) {}

  int struct_id;
  int obj_id;
};

struct ParamsIF : IOParams {
  ParamsIF() :IOParams(PARAMS_IF) {}

  int i = 0;
  float f = 0.0;
};

struct ParamsFxII : IOParams {
  ParamsFxII() :IOParams(PARAMS_FXII) {}
  ~ParamsFxII() {
    if (fx != nullptr) { delete[]fx; fx = nullptr; }
  }

  float *fx = nullptr;
  int i1 = 0;
  int i2 = 0;
};

class Assistor {
public:
  static IOParams *CreateParams(ParamsMode mode) {
    switch (mode) {
    case PARAMS_IF:
      return new ParamsIF();
    case PARAMS_FXII:
      return new ParamsFxII();
    default:
      std::cout << "CreateParams -> Do not support mode " << mode << std::endl;
      return 0;
    }
  }

  static int CopyParams(hcs::IOParams *from, hcs::IOParams *to) {
    if (from->struct_id != to->struct_id) {
      std::cout << "CopyParams -> from->struct_id != to->struct_id." << std::endl;
      return 0;
    }
    switch (from->struct_id) {
    case hcs::PARAMS_IF:
      *(hcs::ParamsIF *)to = *(hcs::ParamsIF *)from;
      return 1;
    case hcs::PARAMS_FXII:
    {
      hcs::ParamsFxII *out = (hcs::ParamsFxII *)to;
      hcs::ParamsFxII *in = (hcs::ParamsFxII *)from;
      if (out->i1 != in->i1 || out->i2 != in->i2) {
        if (out->fx != nullptr) {
          delete[] out->fx;
        }
        out->fx = new float[in->i1 * in->i2];
      }
      out->i1 = in->i1;
      out->i2 = in->i2;
      out->obj_id = in->obj_id;
      memcpy(out->fx, in->fx, sizeof(float) * in->i1 * in->i2);
      return 1;
    }
    default:
      std::cout << "CopyParams -> Do not support mode " << from->struct_id << std::endl;
      return 0;
    }
  }
};

}  // namespace hcs.

#endif // HCS_PARAMS_H_