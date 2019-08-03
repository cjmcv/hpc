#include "gtest/gtest.h"
#include "util/util.h"
#include "util/data_filler.h"

namespace {

void ZeroFillTest() {
  cux::DataFiller filler;

  int len = 20; 
  cux::half *arr = new cux::half[len];
  filler.ZeroFill(len, arr);
  for (int i = 0; i < len; i++) {
    EXPECT_EQ(float(arr[i]), 0);
  }

  delete[]arr;
}

template <typename DType>
void ConstantFillTest(float value, float deviation) {
  cux::DataFiller filler;

  int len = 20;
  DType *arr = new DType[len];
  filler.ConstantFill(value, len, arr);
  for (int i = 0; i < len; i++) {
    if (deviation == 0) 
      EXPECT_EQ(float(arr[i]), value);
    else {
      EXPECT_LE(float(arr[i]) - deviation, value);
      EXPECT_GE(float(arr[i]) + deviation, value);
    }
  }
  delete[]arr;
}

template <typename DType>
void RandomFillTest(int min, int max, int decimal_pose, float deviation) {
  cux::DataFiller filler;

  int len = 20;
  DType *arr = new DType[len];
  filler.RandomFill(min, max, decimal_pose, len, arr);
  for (int i = 0; i < len; i++) {
    EXPECT_LE(float(arr[i]) - deviation, max);
    EXPECT_GE(float(arr[i]) + deviation, min);
  }
  delete[]arr;
}

TEST(FillerTest, Zero) {
  ZeroFillTest();
}
TEST(FillerTest, Constant) {
  ConstantFillTest<cux::half>(1.234, 0.01);
  ConstantFillTest<float>(2.345, 0);
  ConstantFillTest<int>(3, 0);
}
TEST(FillerTest, Random) {
  RandomFillTest<cux::half>(-100, -11, 3, 0.001);
  RandomFillTest<cux::half>(-100, 200, 2, 0.001);
  RandomFillTest<cux::half>(123, 345, 1, 0.001);
  RandomFillTest<float>(-1, 1, 4, 0);
  RandomFillTest<float>(-123, 168, 5, 0);
  RandomFillTest<float>(0, 168, 5, 0);
  RandomFillTest<float>(-100, -10, 3, 0);
  RandomFillTest<int>(9, 999, 0, 0);
  RandomFillTest<int>(-99, 99, 0, 0);
  RandomFillTest<int>(-123, -23, 0, 0);
}
} // namespace