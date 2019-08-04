/*!
* \brief LaunchConfig.
*/

#ifndef CUX_DATA_FILLER_H_
#define CUX_DATA_FILLER_H_

#include <iostream>
#include <time.h>

namespace cux { 

class DataFiller {

public:
  template <typename DType>
  static void ZeroFill(int len, DType *arr) {
    memset(arr, 0, sizeof(DType) * len);
  }

  template <typename DType>
  static void ConstantFill(float value, int len, DType *arr) {
    if (value == 0) {
      ZeroFill(len, arr);
      return;
    }
    for (int i = 0; i < len; i++) {
      arr[i] = value;
    }
  }

  template <typename DType>
  static void StepFill(int step, int len, DType *arr) {
    for (int i = 0; i < len; i++) {
      arr[i] = i % step;
    }
  }

  template <typename DType>
  static void RandomFill(int min, int max, int decimal_pose, int len, DType *arr) {
    srand(time(0));
    if (decimal_pose > 0) { max--; }

    int mod = max - min + 1;
    for (int i = 0; i < len; i++) {
      arr[i] = rand() % mod + min;  // int
    }

    if (decimal_pose > 0) {
      int s = 1;
      while (decimal_pose--) {
        s *= 10;
      }
      for (int i = 0; i < len; i++) {
        float t = (float)(rand() % s) / s; // (0, 1)
        arr[i] += t;
      }
    }
  }
};

}	//namespace cux
#endif //CUX_DATA_FILLER_H_
