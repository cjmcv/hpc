/*!
* \brief Vector dot product: h_result = SUM(A * B).
*/

#include "ocl_util.h"

void DotProductHost(const int* src1, const int* src2, int* dst, int num_elements) {
  *dst = 0;
  for (int i = 0; i < num_elements; i++) {
    (*dst) += src1[i] * src2[i];
  }
}

int main(int argc, char **argv) {
  cl_int err_code;
  // set and log Global and Local work size dimensions
  size_t local_work_size = 256;
  // 1D var for Total # of work items
  size_t global_work_size = local_work_size * 2;

  printf("Global Work Size = %u, Local Work Size = %u, # of Work Groups = %u\n\n",
    global_work_size, local_work_size, (global_work_size % local_work_size + global_work_size / local_work_size));

  // Allocate and initialize host arrays.
  int *h_src1 = (int *)malloc(sizeof(cl_int) * global_work_size);
  int *h_src2 = (int *)malloc(sizeof(cl_int) * global_work_size);
  int h_dst = 0;
  int h_dst4cl = 0;
  for (int i = 0; i < global_work_size; i++) {
    h_src1[i] = i;
    h_src2[i] = i;
  }

  //Get an OpenCL platform
  cl_uint num_platforms;
  OCL_CHECK(clGetPlatformIDs(5, NULL, &num_platforms));
  printf("There are ( %d ) platforms that support OpenCL.\n\n", num_platforms);

  cl_platform_id *platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
  OCL_CHECK(clGetPlatformIDs(num_platforms, platforms, NULL));

  for (int i = 0; i < num_platforms; i++) {
    cjmcv_ocl_util::PrintPlatBasicInfo(platforms[i]);

    //Get the devices
    cl_device_id device;
    OCL_CHECK(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL));

    //Create the context
    cl_context context = clCreateContext(0, 1, &device, NULL, NULL, &err_code);
    OCL_CHECK(err_code);

    // Allocate the OpenCL buffer memory objects for source and result on the device GMEM
    cl_mem d_src1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * global_work_size, NULL, &err_code);
    OCL_CHECK(err_code);
    cl_mem d_src2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * global_work_size, NULL, &err_code);
    OCL_CHECK(err_code);
    cl_mem d_dst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int), NULL, &err_code);
    OCL_CHECK(err_code);

    // Read the OpenCL kernel in from source file
    const char* source_file = "../../alg_dot_product.cl";
    size_t kernel_length;			// Byte size of kernel code
                              // Buffer to hold source for compilation
    char* kernel_source = cjmcv_ocl_util::LoadProgSource(source_file, "", &kernel_length);
    if (kernel_source == NULL) {
      printf("LoadProgSource Failed.\n"); return -1;
    }
    else {
      printf("LoadProgSource (%s): Succeed\n", source_file);
    }

    // Create the program
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, &kernel_length, &err_code);
    OCL_CHECK(err_code);
    OCL_CHECK(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));

    // Create the kernel
    cl_kernel kernel = clCreateKernel(program, "DotProductDevice", &err_code);
    OCL_CHECK(err_code);

    // Set the Argument values
    OCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&d_src1));
    OCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&d_src2));
    OCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&d_dst));

    //--------------------------------------------------------
    // Create a command-queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device, 0, &err_code);
    OCL_CHECK(err_code);

    // Asynchronous write of data to GPU device
    OCL_CHECK(clEnqueueWriteBuffer(command_queue, d_src1, CL_FALSE, 0, sizeof(cl_int) * global_work_size, h_src1, 0, NULL, NULL));
    OCL_CHECK(clEnqueueWriteBuffer(command_queue, d_src2, CL_FALSE, 0, sizeof(cl_int) * global_work_size, h_src2, 0, NULL, NULL));

    // Launch kernel
    OCL_CHECK(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL));

    // Synchronous/blocking read of results, and check accumulated errors
    OCL_CHECK(clEnqueueReadBuffer(command_queue, d_dst, CL_TRUE, 0, sizeof(cl_int), &h_dst4cl, 0, NULL, NULL));
    //--------------------------------------------------------

    // Compute and compare results on host.
    DotProductHost(h_src1, h_src2, &h_dst, global_work_size);
    printf("Test: %s (%d, %d)\n \n", (h_dst4cl == h_dst ? "PASS" : "FAILED"), h_dst4cl, h_dst);

    // Cleanup
    if (kernel) OCL_CHECK(clReleaseKernel(kernel));
    if (program) OCL_CHECK(clReleaseProgram(program));
    if (command_queue) OCL_CHECK(clReleaseCommandQueue(command_queue));
    if (context) OCL_CHECK(clReleaseContext(context));
    if (d_src1) OCL_CHECK(clReleaseMemObject(d_src1));
    if (d_src2) OCL_CHECK(clReleaseMemObject(d_src2));
    if (d_dst) OCL_CHECK(clReleaseMemObject(d_dst));

    if (kernel_source) free(kernel_source);
  }

  if (h_src1) free(h_src1);
  if (h_src2) free(h_src2);
}
