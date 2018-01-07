/*!
* \brief Matrix Multiplication.
*/
#include<iostream>
#include "time.h"

#include "xmmintrin.h"

// Initialize the input data.
void GenMatrix(float *mat, int width) {
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < width; j++) {
      mat[i*width + j] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX*RAND_MAX);
    }
  }
}

void MatrixMulNormal(const float* matA, const float* matB, float* matRet, int width) {
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < width; j++) {
      float sum = 0;
      for (int k = 0; k < width; k++) {
        sum += matA[i * width + k] * matB[k * width + j];
      }
      matRet[i * width + j] = sum;
    }
  }
}

void MatrixMulSSE(const float* matA, const float* matB, float* matRet, int width) {
  if (width < 8)
    return;

  for (int i = 0; i < width; i++) {
    for (int j = 0; j < width; j++) {
      __m128 sum = _mm_setzero_ps();
      for (int k = 0; k < width; k += 8) {
        __m128 a0 = _mm_loadu_ps(matA + i*width + k);
        __m128 a1 = _mm_loadu_ps(matA + i*width + k + 4);

        // Notice that the order of _mm_set_ps is reversed
        __m128 b0 = _mm_set_ps(matB[(k + 3)*width + j], matB[(k + 2)*width + j], matB[(k + 1)*width + j], matB[k*width + j]);
        __m128 b1 = _mm_set_ps(matB[(k + 7)*width + j], matB[(k + 6)*width + j], matB[(k + 5)*width + j], matB[(k + 4)*width + j]);
        sum = _mm_add_ps(sum, _mm_add_ps(_mm_mul_ps(a0, b0), _mm_mul_ps(a1, b1)));
      }
      matRet[i*width + j] = sum.m128_f32[0] + sum.m128_f32[1] + sum.m128_f32[2] + sum.m128_f32[3];
    }
  }
}

// Just for checking the result.
float GetMean(const float* mat, const int height, const int width) {
  int num = height * width;
  float total = 0;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      total += mat[i*width + j];
    }
  }
  return total / num;
}

void MatrixPrint(const float* mat, const int height, const int width) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      std::cout << mat[i*width + j] << ",";
    }
    std::cout << std::endl;
  }
}

int main() {
  int width = 128;
  float *matA = new float[width * width];
  float *matB = new float[width * width];
  float *matRetOrg = new float[width * width];
  float *matRetSSE = new float[width * width];

  srand(0);
  GenMatrix(matA, width);
  GenMatrix(matB, width);

  time_t stime;

  stime = clock();
  for (int i = 0; i < 100; i++) {
    MatrixMulNormal(matA, matB, matRetOrg, width);
  }
  std::cout << "Normal ->  time: " << clock() - stime << std::endl;

  stime = clock();
  for (int i = 0; i < 100; i++) {
    MatrixMulSSE(matA, matB, matRetSSE, width);
  }
  std::cout << "SSE ->  time: " << clock() - stime << std::endl;

  printf("avg = %f \n", GetMean(matRetOrg, width, width));
  printf("avg2 = %f \n", GetMean(matRetSSE, width, width));

  //std::cout << "\n Org result" << std::endl;
  //MatrixPrint(matRetOrg, width, width);

  //std::cout << "\n SSE result \n" << std::endl;
  //MatrixPrint(matRetSSE, width, width);

  delete[] matA;
  delete[] matB;
  delete[] matRetOrg;
  delete[] matRetSSE;

  system("pause");
  return 0;
}