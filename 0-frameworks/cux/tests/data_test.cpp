#include "util/util.h"
#include "array.h"

#include "gtest/gtest.h"

namespace {

template <typename DType, typename BType>
void DataPushTest(int num, int channels, int height, int width, float scale, float beta) {

  cux::Array4D *in = new cux::Array4D(num, channels, height, width);
  int len = 1;
  for (int i = 0; i < 4; i++)
    len *= in->shape()[i];
  EXPECT_EQ(len, num * channels * height * width);

  BType *benchmark = new BType[len];
  for (int i = 0; i < len; i++)
    benchmark[i] = i * scale + beta;

  // Fill.
  DType *cpu_data = in->GetCpuData<DType>();
  for (int i = 0; i < len; i++) {
    cpu_data[i] = benchmark[i];
  }
  // Push.
  in->GetGpuData<DType>(cux::PUSH);
  // Clear.
  for (int i = 0; i < len; i++) {
    cpu_data[i] = 0;
  }
  // Push back and check.
  cpu_data = in->GetCpuData<DType>(cux::PUSH);
  for (int i = 0; i < len; i++) {
    EXPECT_EQ(BType(cpu_data[i]), benchmark[i]);
  }
  delete in;
}

template <typename SrcType, typename DstType, typename BType>
void DataFpCvtPushTest(int num, int channels, int height, int width, float scale, float beta) {

  cux::Array4D *in = new cux::Array4D(num, channels, height, width);
  int len = 1;
  for (int i = 0; i < 4; i++)
    len *= in->shape()[i];
  EXPECT_EQ(len, num * channels * height * width);

  BType *benchmark = new BType[len];
  for (int i = 0; i < len; i++)
    benchmark[i] = i * scale + beta;

  // Fill.
  SrcType *cpu_data = in->GetCpuData<SrcType>();
  for (int i = 0; i < len; i++) {
    cpu_data[i] = benchmark[i];
  }

  in->PrecsCpuCvt<SrcType, DstType>();

  DstType *dst_cpu_data = in->GetCpuData<DstType>(cux::PUSH);
  for (int i = 0; i < len; i++) {
    EXPECT_EQ(BType(dst_cpu_data[i]), benchmark[i]);
  }

  delete in;
}

template <typename DType, typename BType>
void DataBackupTest(int num, int channels, int height, int width, float scale, float beta) {

  cux::Array4D *in = new cux::Array4D(num, channels, height, width);
  int len = 1;
  for (int i = 0; i < 4; i++)
    len *= in->shape()[i];
  EXPECT_EQ(len, num * channels * height * width);

  BType *benchmark = new BType[len];
  for (int i = 0; i < len; i++)
    benchmark[i] = i * scale + beta;

  // Fill.
  DType *cpu_data = in->GetCpuData<DType>();
  for (int i = 0; i < len; i++) {
    cpu_data[i] = benchmark[i];
  }
  // Save.
  int type = cux::DataType<DType>::kFlag;
  in->Save(type, cux::ON_HOST);

  // Clear.
  for (int i = 0; i < len; i++) {
    cpu_data[i] = 0;
  }
  // Restore and check.
  in->Restore(type, cux::ON_HOST);
  for (int i = 0; i < len; i++) {
    EXPECT_EQ(BType(cpu_data[i]), benchmark[i]);
  }
  delete in;
}

// Push.
TEST(DataTest, PushAndBackInt) {
  DataPushTest<int32_t, int32_t>(1, 2, 1, 3, 1.2, -2);
}
TEST(DataTest, PushAndBackFloat) {
  DataPushTest<float, float>(2, 1, 1, 2, 3.5, 5);
}
TEST(DataTest, PushAndBackHalf) {
  DataPushTest<cux::half, float>(2, 1, 1, 2, 3.5, 5);
}
TEST(DataTest, PushAndBackChar) {
  DataPushTest<int8_t, int8_t>(1, 3, 1, 3, 123, 32);
}

// Precision convert.
TEST(DataTest, FpCvtFp32ToHalf) {
  DataFpCvtPushTest<float, cux::half, float>(1, 2, 1, 3, 1.5, -2.0);
}
TEST(DataTest, FpCvtHalfToFp32) {
  DataFpCvtPushTest<cux::half, float, float>(1, 2, 1, 2, 3.5, -3.0);
}
TEST(DataTest, FpCvtInt32ToInt8) {
  DataFpCvtPushTest<int32_t, int8_t, float>(1, 5, 1, 1, 6, -2);
}
TEST(DataTest, FpCvtInt8ToInt32) {
  DataFpCvtPushTest<int8_t, int32_t, float>(1, 1, 3, 3, 5, -3);
}

// Save and Restore.
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