/*!
* \brief Vector dot product: h_result = SUM(A * B).
*        Record the basic usage of std::tread.
*/

#include<iostream>
#include<thread>

void DotProduct(float *src0, float *src1, int len, float *dst) {
  *dst = 0;
  for (int i = 0; i < len; i++) {
    *dst += src0[i] * src1[i];
  }
}

void DotProductMT(float *src0, float *src1, int len, int num_threads, float *dst) {
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

int main() {
  int len = 10000000;
  float *src0 = new float[len];
  float *src1 = new float[len];

  for (int i = 0; i < len; i++) {
    src0[i] = src1[i] = 1;
  }

  float dst = 0;
  clock_t time = clock();
  DotProductMT(src0, src1, len, 5, &dst);
  std::cout << "<MT> result: " << dst << ", time consume: "  \
    << clock() - time << std::endl;

  dst = 0;
  time = clock();
  DotProduct(src0, src1, len, &dst);
  std::cout << "<Serial> result: " << dst << ", time consume: " \
    << clock() - time << std::endl;

  delete src0;
  delete src1;
}