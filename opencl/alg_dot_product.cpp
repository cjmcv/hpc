/*!
* \brief Vector dot product: h_result = SUM(A * B).
*/

#include "ocl_util.h"

void DotProductHost(const int* src1, const int* src2, int* dst, size_t num_elements) {
  *dst = 0;
  for (size_t i = 0; i < num_elements; i++) {
    (*dst) += src1[i] * src2[i];
  }
}

int main(int argc, char **argv) {
  cl_int err_code;
  size_t num_elements = 1000000;
  // set and log Global and Local work size dimensions
  size_t local_work_size = 256;
  // 1D var for Total # of work items
  size_t global_work_size = cjmcv_ocl_util::GetRoundUpMultiple(num_elements, local_work_size) * local_work_size;

  printf("Global Work Size = %zu, Local Work Size = %zu, # of Work Groups = %zu\n\n",
    global_work_size, local_work_size, global_work_size / local_work_size);

  // Allocate and initialize host arrays.
  int *h_src1 = (int *)malloc(sizeof(cl_int) * num_elements);
  int *h_src2 = (int *)malloc(sizeof(cl_int) * num_elements);
  int h_dst = 0;
  int h_dst4cl = 0;
  for (int i = 0; i < num_elements; i++) {
    h_src1[i] = i;
    h_src2[i] = i;
  }

  // Load CL source.
  cjmcv_ocl_util::KernelLoader *loader = new cjmcv_ocl_util::KernelLoader;
  loader->Load("../../alg_dot_product.cl");

  // Get an OpenCL platform.
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

    // Create the context
    cl_context context = clCreateContext(0, 1, &device, NULL, NULL, &err_code);
    OCL_CHECK(err_code);

    // Get Kernel.
    cl_kernel kernel;
    loader->CreateProgram(context);
    loader->GetKernel("DotProductDevice", &kernel);

    // Allocate the OpenCL buffer memory objects for source and result on the device GMEM
    cl_mem d_src1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * num_elements, NULL, &err_code);
    OCL_CHECK(err_code);
    cl_mem d_src2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * num_elements, NULL, &err_code);
    OCL_CHECK(err_code);
    cl_mem d_dst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int), NULL, &err_code);
    OCL_CHECK(err_code);

    // Set the Argument values
    OCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&d_src1));
    OCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&d_src2));
    OCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&d_dst));

    //--------------------------------------------------------
    // Create a command-queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err_code);
    //cl_command_queue command_queue = clCreateCommandQueue(context, device, 0, &err_code);
    OCL_CHECK(err_code);

    // Asynchronous write of data to GPU device
    OCL_CHECK(clEnqueueWriteBuffer(command_queue, d_src1, CL_FALSE, 0, sizeof(cl_int) * num_elements, h_src1, 0, NULL, NULL));
    OCL_CHECK(clEnqueueWriteBuffer(command_queue, d_src2, CL_FALSE, 0, sizeof(cl_int) * num_elements, h_src2, 0, NULL, NULL));

    // Launch kernel
    cl_event ev;
    OCL_CHECK(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &ev));
    //OCL_CHECK(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL));

    // Synchronous/blocking read of results, and check accumulated errors
    OCL_CHECK(clEnqueueReadBuffer(command_queue, d_dst, CL_TRUE, 0, sizeof(cl_int), &h_dst4cl, 0, NULL, NULL));
	
	  // Block until all tasks in command_queue have been completed.
    clFinish(command_queue);

    // Gets the running time of the kernel function.
    cjmcv_ocl_util::PrintCommandElapsedTime(ev);
    //--------------------------------------------------------

    // Compute and compare results on host.
    DotProductHost(h_src1, h_src2, &h_dst, num_elements);
    printf("Test: %s (%d, %d)\n \n", (h_dst4cl == h_dst ? "PASS" : "FAILED"), h_dst4cl, h_dst);

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

  return 0;
}
