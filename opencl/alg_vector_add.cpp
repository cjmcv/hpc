/*!
* \brief Vector addition: C = A + B.
*/

#include "ocl_util.h"

void VectorAddHost(const float* src1, const float* src2, float* h_dst, size_t num_elements) {
  for (int i = 0; i < num_elements; i++) {
    h_dst[i] = src1[i] + src2[i];
  }
}

int main(int argc, char **argv) {
  cl_int err_code;
  size_t num_elements = 65537;
  // set and log Global and Local work size dimensions
  size_t local_work_size = 256;
  // 1D var for Total # of work items
  size_t global_work_size = cjmcv_ocl_util::GetRoundUpMultiple(num_elements, local_work_size) * local_work_size;

  printf("Global Work Size = %zu, Local Work Size = %zu, # of Work Groups = %zu\n\n",
    global_work_size, local_work_size, global_work_size / local_work_size);

  // Allocate and initialize host arrays.
  float *h_src1 = (float *)malloc(sizeof(cl_float) * num_elements);
  float *h_src2 = (float *)malloc(sizeof(cl_float) * num_elements);
  float *h_dst = (float *)malloc(sizeof(cl_float) * num_elements);
  float *h_dst4cl = (float *)malloc(sizeof(cl_float) * num_elements);
  for (size_t i = 0; i < num_elements; i++) {
    h_src1[i] = i;
    h_src2[i] = i;
  }

  // Load CL source.
  cjmcv_ocl_util::KernelLoader *loader = new cjmcv_ocl_util::KernelLoader;
  loader->Load("../../alg_vector_add.cl");

  //Get an OpenCL platform
  cl_uint num_platforms;
  OCL_CHECK(clGetPlatformIDs(5, NULL, &num_platforms));
  printf("There are ( %d ) platforms that support OpenCL.\n\n", num_platforms);

  cl_platform_id *platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
  OCL_CHECK(clGetPlatformIDs(num_platforms, platforms, NULL));

  for (cl_uint i = 0; i < num_platforms; i++) {
    cjmcv_ocl_util::PrintPlatBasicInfo(platforms[i]);

    // Get the devices
    cl_device_id device;
    OCL_CHECK(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL));

    //Create the context
    cl_context context = clCreateContext(0, 1, &device, NULL, NULL, &err_code);
    OCL_CHECK(err_code);

    // Get Kernel.
    cl_kernel kernel;
    loader->CreateProgram(context);
    loader->GetKernel("VectorAdd", &kernel);

    // Allocate the OpenCL buffer memory objects for source and result on the device GMEM
    cl_mem d_src1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * num_elements, NULL, &err_code);
    OCL_CHECK(err_code);
    cl_mem d_src2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * num_elements, NULL, &err_code);
    OCL_CHECK(err_code);
    cl_mem d_dst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * num_elements, NULL, &err_code);
    OCL_CHECK(err_code);

    // Set the Argument values
    OCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&d_src1));
    OCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&d_src2));
    OCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&d_dst));
    OCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&num_elements));

    //--------------------------------------------------------
    // Create a command-queue
    // CL_QUEUE_PROFILING_ENABLE: Opencl operation time can only be calculated when this flag bit is turned on
    cl_command_queue command_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err_code);
    OCL_CHECK(err_code);
    
    // Asynchronous write of data to GPU device
    OCL_CHECK(clEnqueueWriteBuffer(command_queue, d_src1, CL_FALSE, 0, sizeof(cl_float) * num_elements, h_src1, 0, NULL, NULL));
    OCL_CHECK(clEnqueueWriteBuffer(command_queue, d_src2, CL_FALSE, 0, sizeof(cl_float) * num_elements, h_src2, 0, NULL, NULL));

    // Launch kernel
    cl_event ev;
    OCL_CHECK(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &ev));

    // Synchronous/blocking read of results, and check accumulated errors
    OCL_CHECK(clEnqueueReadBuffer(command_queue, d_dst, CL_TRUE, 0, sizeof(cl_float) * num_elements, h_dst4cl, 0, NULL, NULL));
    
    // Block until all tasks in command_queue have been completed.
    clFinish(command_queue);
  
    // Gets the running time of the kernel function.
    cjmcv_ocl_util::PrintCommandElapsedTime(ev);
    //--------------------------------------------------------

    // Compute and compare results on host.
    VectorAddHost(h_src1, h_src2, h_dst, num_elements);
    bool is_equal = true;
    for (int ni = 0; ni < num_elements; ni++) {
      if (h_dst[ni] != h_dst4cl[ni]) {
        is_equal = false;
        break;
      }
    }
    printf("Test: %s \n \n", (is_equal ? "PASS":"FAILED"));

    // Cleanup
    if (kernel) OCL_CHECK(clReleaseKernel(kernel));
    if (command_queue) OCL_CHECK(clReleaseCommandQueue(command_queue));
    if (context) OCL_CHECK(clReleaseContext(context));
    if (d_src1) OCL_CHECK(clReleaseMemObject(d_src1));
    if (d_src2) OCL_CHECK(clReleaseMemObject(d_src2));
    if (d_dst) OCL_CHECK(clReleaseMemObject(d_dst));
  }

  if (loader) {
    loader->UnLoad();
    delete loader;
  }
  if (h_src1) free(h_src1);
  if (h_src2) free(h_src2);
  if (h_dst) free(h_dst);
  if (h_dst4cl) free(h_dst4cl);

  return 0;
}
