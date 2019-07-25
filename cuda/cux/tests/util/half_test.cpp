#include "gtest/gtest.h"
#include "util/half.h"

namespace {

#define HALF_TEST_DEVIATION 0.01

TEST(HalfTest, HostBase) {
  float in1 = 9.125;
  cux::half a(in1);
  EXPECT_LE(in1 - HALF_TEST_DEVIATION, float(a));
  EXPECT_GE(in1 + HALF_TEST_DEVIATION, float(a));
  //
  float in2 = 1.234;
  cux::half b(in2);
  EXPECT_LE(in2 - HALF_TEST_DEVIATION, float(b));
  EXPECT_GE(in2 + HALF_TEST_DEVIATION, float(b));
}

TEST(HalfTest, HostBase2) {
  int in1 = 1234;
  cux::half a(in1);
  EXPECT_EQ(in1, float(a));
  //
  unsigned char in2 = 200;
  cux::half b(in2);
  EXPECT_EQ(in2, float(b));
}

TEST(HalfTest, HostAdd) {
  float in1 = 1.125;
  int in2 = 8;
  cux::half b(in1);
  cux::half c(in2);
  cux::half a = b + c;

  EXPECT_LE(in1 + in2 - HALF_TEST_DEVIATION, float(a));
  EXPECT_GE(in1 + in2 + HALF_TEST_DEVIATION, float(a));
}

TEST(HalfTest, HostSub) {
  float in1 = 5.678;
  float in2 = 1.987;
  cux::half b(in1);
  cux::half c(in2);
  cux::half a = b - c;

  EXPECT_LE(in1 - in2 - HALF_TEST_DEVIATION, float(a));
  EXPECT_GE(in1 - in2 + HALF_TEST_DEVIATION, float(a));
}

TEST(HalfTest, HostMul) {
  float in1 = 6.123;
  int in2 = 3;
  cux::half b(in1);
  cux::half c(in2);
  cux::half a = b * c;

  EXPECT_LE(in1 * in2 - HALF_TEST_DEVIATION, float(a));
  EXPECT_GE(in1 * in2 + HALF_TEST_DEVIATION, float(a));
}

TEST(HalfTest, HostDiv) {
  float in1 = 6.688;
  int in2 = 2;
  cux::half b(in1);
  cux::half c(in2);
  cux::half a = b / c;

  EXPECT_LE(in1 / in2 - HALF_TEST_DEVIATION, float(a));
  EXPECT_GE(in1 / in2 + HALF_TEST_DEVIATION, float(a));
}

} // namespace