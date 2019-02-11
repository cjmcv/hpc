/*!
* \brief Vector dot product: h_result = SUM(A * B).
*        V1: Record the basic usage of std::thread.
*        V2: Record the basic usage of std::sync.
*/

#include<iostream>
#include<thread>
#include<future>

void DotProductSerial(float *src0, float *src1, int len, float *dst) {
  *dst = 0;
  for (int i = 0; i < len; i++) {
    *dst += src0[i] * src1[i];
  }
}

// MT Version 1.
void DotProductMTv1(float *src0, float *src1, int len, int num_threads, float *dst) {

  auto DotProduct = [](float *src0, float *src1, int len, float *dst) {
    DotProductSerial(src0, src1, len, dst);
  };

  int sub_len = len / num_threads;

  float *dst_arr = new float[num_threads];
  memset(dst, 0, sizeof(float) * num_threads);

  // Process.
  std::thread *threads = new std::thread[num_threads];
  for (int i = 0; i < num_threads; i++) {
    threads[i] = std::thread(DotProduct, src0 + i * sub_len, src1 + i * sub_len, sub_len, dst_arr + i);
  }
  // Synchronous.
  for (int i = 0; i < num_threads; i++) {
    threads[i].join();
  }

  // Merge.
  *dst = 0;
  for (int i = 0; i < num_threads; i++) {
    *dst += dst_arr[i];
  }

  // Deal with the rest
  int rest_index = len - num_threads * sub_len;
  if (rest_index) {
    float dst_rest = 0;
    DotProduct(src0 + len - rest_index, src1 + len - rest_index, rest_index, &dst_rest);
    *dst += dst_rest;
  }

  delete[]dst_arr;
  delete[]threads;
}

// MT Version 2.
void DotProductMTv2(float *src0, float *src1, int len, int num_threads, float *dst) {
  
  auto DotProduct = [](float *src0, float *src1, int len) {
    float dst = 0;
    DotProductSerial(src0, src1, len, &dst);
    return dst;
  };

  int sub_len = len / num_threads;

  std::future<float> *dst_arr = new std::future<float>[num_threads];
  memset(dst, 0, sizeof(float) * num_threads);

  // Process.
  for (int i = 0; i < num_threads; i++) {
    dst_arr[i] = std::async(DotProduct, src0 + i * sub_len, src1 + i * sub_len, sub_len);
  }

  // Merge.
  *dst = 0;
  for (int i = 0; i < num_threads; i++) {
    *dst += dst_arr[i].get();
  }

  // Deal with the rest
  int rest_index = len - num_threads * sub_len;
  if (rest_index) {
    *dst += DotProduct(src0 + len - rest_index, src1 + len - rest_index, rest_index);
  }

  delete[]dst_arr;
}

int main() {
  int len = 12345000;
  float *src0 = new float[len];
  float *src1 = new float[len];

  for (int i = 0; i < len; i++) {
    src0[i] = src1[i] = 1;
  }

  float dst = 0;
  clock_t time = clock();
  DotProductMTv1(src0, src1, len, 12, &dst);
  std::cout << "<MT thread> result: " << dst << ", time consume: "  \
    << clock() - time << std::endl;

  dst = 0;
  time = clock();
  DotProductMTv2(src0, src1, len, 12, &dst);
  std::cout << "<MT sync> result: " << dst << ", time consume: " \
    << clock() - time << std::endl;

  dst = 0;
  time = clock();
  DotProductSerial(src0, src1, len, &dst);
  std::cout << "<Serial> result: " << dst << ", time consume: " \
    << clock() - time << std::endl;

  delete src0;
  delete src1;
  
  return 0;
}