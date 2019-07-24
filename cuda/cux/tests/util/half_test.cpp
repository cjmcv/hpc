#include "util/half.h"
#include "gtest/gtest.h"

// Output should be the same as input.
float Float2Half2Float(float input) {
  return cux::half2float(cux::float2half(input));
}

// Tests factorial of 0.
TEST(Float2Half2FloatTest, FLOAT1) {
  EXPECT_EQ(9.125, Float2Half2Float(9.125));
}