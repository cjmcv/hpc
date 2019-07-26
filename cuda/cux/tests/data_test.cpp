#include "gtest/gtest.h"
#include "data.h"

namespace {

template <typename Dtype, typename Ttype>
void DataPushTest(int num, int channels, int height, int width, float scale, float beta) {

  cux::CuxData<Dtype> *in = new cux::CuxData<Dtype>(num, channels, height, width);
  int len = 1;
  for (int i = 0; i < 4; i++)
    len *= in->shape()[i];
  EXPECT_EQ(len, num * channels * height * width);

  Ttype *benchmark = new Ttype[len];
  for (int i = 0; i < len; i++)
    benchmark[i] = i * scale + beta;

  // Fill.
  Dtype *cpu_data = in->GetCpuData();
  for (int i = 0; i < len; i++) {
    cpu_data[i] = benchmark[i];
  }
  // Push.
  in->GetGpuData(cux::PUSH);
  // Clear.
  for (int i = 0; i < len; i++) {
    cpu_data[i] = 0;
  }
  // Push back and check.
  cpu_data = in->GetCpuData(cux::PUSH);
  for (int i = 0; i < len; i++) {
    EXPECT_EQ(Ttype(cpu_data[i]), benchmark[i]);
  }
  delete in;
}

template <typename Dtype, typename Ttype>
void DataBackupTest(int num, int channels, int height, int width, float scale, float beta) {

  cux::CuxData<Dtype> *in = new cux::CuxData<Dtype>(num, channels, height, width);
  int len = 1;
  for (int i = 0; i < 4; i++)
    len *= in->shape()[i];
  EXPECT_EQ(len, num * channels * height * width);

  Ttype *benchmark = new Ttype[len];
  for (int i = 0; i < len; i++)
    benchmark[i] = i * scale + beta;

  // Fill.
  Dtype *cpu_data = in->GetCpuData();
  for (int i = 0; i < len; i++) {
    cpu_data[i] = benchmark[i];
  }
  // Save.
  in->Save(cux::ON_HOST);

  // Clear.
  for (int i = 0; i < len; i++) {
    cpu_data[i] = 0;
  }
  // Restore and check.
  in->Restore(cux::ON_HOST);
  for (int i = 0; i < len; i++) {
    EXPECT_EQ(Ttype(cpu_data[i]), benchmark[i]);
  }
  delete in;
}

TEST(DataTest, PushAndBackInt) {
  DataPushTest<int, int>(1, 2, 1, 3, 1.2, -2.0);
}
TEST(DataTest, PushAndBackFloat) {
  DataPushTest<float, float>(2, 1, 1, 2, 3.5, 5);
}
TEST(DataTest, PushAndBackHalf) {
  DataPushTest<cux::half, float>(2, 1, 1, 2, 3.5, 5);
}
TEST(DataTest, PushAndBackChar) {
  DataPushTest<unsigned char, unsigned char>(1, 3, 1, 3, 123, 32);
}

TEST(DataTest, StoreFloat) {
  DataBackupTest<float, float>(1, 3, 1, 3, 12.3, 32);
}
TEST(DataTest, StoreHalf) {
  DataBackupTest<cux::half, float>(1, 2, 2, 3, 2.5, 32); // 0.5/1/1.5/2/2.5...
}
TEST(DataTest, StoreHalf2) {
  DataBackupTest<cux::half, float>(1, 2, 2, 3, 0.5, 32);
}
TEST(DataTest, StoreHalf3) {
  DataBackupTest<cux::half, float>(1, 2, 2, 3, 1.0, 32);
}
} // namespace