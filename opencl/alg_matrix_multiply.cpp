/*!
* \brief gemm: C = A * B.
*/

#include "ocl_util.h"

// Initialize the input data.
void GenMatrix(const int height, const int width, float *mat) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      mat[i*width + j] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX*RAND_MAX);
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

// Just for checking the result too.
void MatrixPrint(const float* mat, const int height, const int width) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      std::cout << mat[i*width + j] << ",";
    }
    std::cout << std::endl;
  }
}

// Normal version in cpu as a reference
void MatrixMulHost(const int M, const int N, const int K, const float ALPHA,
  const float *A, const int lda,
  const float *B, const int ldb,
  float *C, const int ldc) {
  int i, j, k;
  memset(C, 0, sizeof(float) * ldc * M);
  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      register float A_PART = ALPHA*A[i*lda + k];
      for (j = 0; j < N; ++j) {
        C[i*ldc + j] += A_PART*B[k*ldb + j];
      }
    }
  }
}

int main(int argc, char **argv) {
  cl_int err_code;

  size_t height_a = 16, width_a = 32;
  size_t height_b = 32, width_b = 32;
  if (width_a != height_b) {
    printf("width_a should be equal to height_b.\n");
    return 1;
  }

  // set and log Global and Local work size dimensions
  size_t local_work_size[2] = {16, 16}; // x, y
  // 2D var for Total # of work items - (N = height_a, M = width_b)
  size_t global_work_size[2] = 
  { cjmcv_ocl_util::GetRoundUpMultiple(width_b, local_work_size[0]) * local_work_size[0],
    cjmcv_ocl_util::GetRoundUpMultiple(height_a, local_work_size[1]) * local_work_size[1] };

  printf("Global Work Size = {%zu, %zu}, Local Work Size = {%zu, %zu}, # of Work Groups = {%zu, %zu}\n\n",
    global_work_size[0], global_work_size[1], local_work_size[0], local_work_size[1], 
    global_work_size[0] / local_work_size[0], global_work_size[1] / local_work_size[1]);

  // Allocate and initialize host arrays.
  int size_a = sizeof(cl_float) * height_a * width_a;
  int size_b = sizeof(cl_float) * height_b * width_b;
  int size_c = sizeof(cl_float) * height_a * width_b;

  cl_float *h_a = (cl_float *)malloc(size_a);
  cl_float *h_b = (cl_float *)malloc(size_b);
  cl_float *h_c = (cl_float *)malloc(size_c);
  cl_float *h_c4cl = (cl_float *)malloc(size_c);
  cl_float *h_c_init = (cl_float *)malloc(size_c);

  // Initialize 
  srand(0);
  GenMatrix(height_a, width_a, h_a);
  GenMatrix(height_b, width_b, h_b);
  memset(h_c_init, 0, size_c);

  // Load CL source.
  cjmcv_ocl_util::KernelLoader *loader = new cjmcv_ocl_util::KernelLoader;
  loader->Load("../../alg_matrix_multiply.cl");

  // Get an OpenCL platform.
  cl_uint num_platforms;
  OCL_CHECK(clGetPlatformIDs(5, NULL, &num_platforms));
  printf("There are ( %d ) platforms that support OpenCL.\n\n", num_platforms);

  cl_platform_id *platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
  OCL_CHECK(clGetPlatformIDs(num_platforms, platforms, NULL));
  
  std::string prog[2] = { "MatrixMulDeviceV1", "MatrixMulDeviceV2" };
  for (int pi = 0; pi < 2; pi++) {

    for (cl_uint i = 0; i < num_platforms; i++) {
      cjmcv_ocl_util::PrintPlatBasicInfo(platforms[i]);
      printf("Running program %s.\n", prog[pi].c_str());

      // Get the devices
      cl_device_id device;
      OCL_CHECK(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL));

      // Create the context
      cl_context context = clCreateContext(0, 1, &device, NULL, NULL, &err_code);
      OCL_CHECK(err_code);

      // Get Kernel.
      cl_kernel kernel;
      loader->CreateProgram(context);
      loader->GetKernel(prog[pi].c_str(), &kernel);

      // Allocate the OpenCL buffer memory objects for source and result on the device GMEM
      cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, size_a, NULL, &err_code);
      OCL_CHECK(err_code);
      cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, size_b, NULL, &err_code);
      OCL_CHECK(err_code);
      cl_mem d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size_c, NULL, &err_code);
      OCL_CHECK(err_code);

      // Set the Argument values
      /*
        const int M, const int N, const int K, const float ALPHA,
        __global const float *A, const int lda,
        __global const float *B, const int ldb,
        __global float *C, const int ldc
      */
      float ALPHA = 1.0;
      OCL_CHECK(clSetKernelArg(kernel, 0, sizeof(int), &height_a)); // M
      OCL_CHECK(clSetKernelArg(kernel, 1, sizeof(int), &width_b));  // N
      OCL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), &width_a));  // K
      OCL_CHECK(clSetKernelArg(kernel, 3, sizeof(float), &ALPHA));  // ALPHA
      OCL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&d_a)); // A
      OCL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), &width_a));  // 
      OCL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&d_b)); // B
      OCL_CHECK(clSetKernelArg(kernel, 7, sizeof(int), &width_b));  // 
      OCL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*)&d_c)); // C
      OCL_CHECK(clSetKernelArg(kernel, 9, sizeof(int), &width_b));  // 

      //--------------------------------------------------------
      // Create a command-queue
      cl_command_queue command_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err_code);
      //cl_command_queue command_queue = clCreateCommandQueue(context, device, 0, &err_code);
      OCL_CHECK(err_code);

      // Asynchronous write of data to GPU device
      OCL_CHECK(clEnqueueWriteBuffer(command_queue, d_a, CL_FALSE, 0, size_a, h_a, 0, NULL, NULL));
      OCL_CHECK(clEnqueueWriteBuffer(command_queue, d_b, CL_FALSE, 0, size_b, h_b, 0, NULL, NULL));
      OCL_CHECK(clEnqueueWriteBuffer(command_queue, d_c, CL_FALSE, 0, size_c, h_c_init, 0, NULL, NULL));

      // Launch kernel
      cl_event ev;
      OCL_CHECK(clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &ev));
      //OCL_CHECK(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL));

      // Synchronous/blocking read of results, and check accumulated errors
      OCL_CHECK(clEnqueueReadBuffer(command_queue, d_c, CL_TRUE, 0, size_c, h_c4cl, 0, NULL, NULL));

      // Block until all tasks in command_queue have been completed.
      clFinish(command_queue);

      // Gets the running time of the kernel function.
      cjmcv_ocl_util::PrintCommandElapsedTime(ev);
      //--------------------------------------------------------

      // Compute and compare results on host.
      MatrixMulHost(height_a, width_b, width_a, ALPHA, h_a, width_a, h_b, width_b, h_c, width_b);
      float meam_cl = GetMean(h_c4cl, height_a, width_b);
      float mean = GetMean(h_c, height_a, width_b);

      float err = meam_cl - mean;
      printf("Test: %s (%f, %f)\n \n", ((err > -0.00001 && err < 0.00001) ? "PASS" : "FAILED"), meam_cl, mean);

      // Cleanup
      if (kernel) OCL_CHECK(clReleaseKernel(kernel));
      if (command_queue) OCL_CHECK(clReleaseCommandQueue(command_queue));
      if (context) OCL_CHECK(clReleaseContext(context));
      if (d_a) OCL_CHECK(clReleaseMemObject(d_a));
      if (d_b) OCL_CHECK(clReleaseMemObject(d_b));
      if (d_c) OCL_CHECK(clReleaseMemObject(d_c));
    } // for each platform
  } // for each program

  if (loader) {
    loader->UnLoad();
    delete loader;
  }
  if (h_a) free(h_a);
  if (h_b) free(h_b);
  if (h_c) free(h_c);
  if (h_c_init) free(h_c_init);
  if (h_c4cl) free(h_c4cl);

  return 0;
}
