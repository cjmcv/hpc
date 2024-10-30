/*!
* \brief Basic demo: Vector addition C = A + B.
*     Introduce the basic calling method and process of OpenCL API (without using pocket-ai).
*/

#include <iostream>
#include <CL/cl.h>

void VectorAddHost(const float* src1, const float* src2, float* h_dst, size_t num_elements) {
    for (size_t i = 0; i < num_elements; i++) {
        h_dst[i] = src1[i] + src2[i];
    }
}

//  Loads a Program file and prepends the preamble to the code.
char* LoadProgSource(const char* file_name, const char* preamble, size_t* final_length) {
    // Locals 
    FILE* file_stream = NULL;
    size_t source_length;

    // Open the OpenCL source code file
#ifdef _WIN32   // Windows version
    if (fopen_s(&file_stream, file_name, "rb") != 0) {
        printf("Can not open the file : %s.\n", file_name);
        return NULL;
    }
#else           // Linux version
    file_stream = fopen(file_name, "rb");
    if (file_stream == 0) {
        printf("Can not open the file : %s.\n", file_name);
        return NULL;
    }
#endif

    size_t preamble_length = strlen(preamble);

    // get the length of the source code
    fseek(file_stream, 0, SEEK_END);
    source_length = ftell(file_stream);
    fseek(file_stream, 0, SEEK_SET);

    // allocate a buffer for the source code string and read it in
    char* source_string = (char *)malloc(source_length + preamble_length + 1);
    memcpy(source_string, preamble, preamble_length);
    if (fread((source_string)+preamble_length, source_length, 1, file_stream) != 1) {
        fclose(file_stream);
        free(source_string);
        return NULL;
    }

    // close the file and return the total length of the combined (preamble + source) string
    fclose(file_stream);
    if (final_length != 0) {
        *final_length = source_length + preamble_length;
    }
    source_string[source_length + preamble_length] = '\0';

    return source_string;
}

size_t GetRoundUpMultiple(size_t dividend, size_t divisor) {
    return (dividend + divisor - 1) / divisor;
}

int main(int argc, char **argv) {
    cl_int err_code;
    size_t num_elements = 65537;
    // set and log Global and Local work size dimensions
    size_t local_work_size = 256;
    // 1D var for Total # of work items
    size_t global_work_size = GetRoundUpMultiple(num_elements, local_work_size) * local_work_size;

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
    // Byte size of kernel code. Buffer to hold source for compilation
    size_t program_length;
    char *program_source = LoadProgSource("../basic_demo.cl", "", &program_length);
    if (program_source == NULL) {
        printf("LoadProgSource Failed.\n"); 
        return -1;
    }

    //Get an OpenCL platform
    cl_uint num_platforms;
    clGetPlatformIDs(5, NULL, &num_platforms);
    printf("There are ( %d ) platforms that support OpenCL.\n\n", num_platforms);

    cl_platform_id *platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    clGetPlatformIDs(num_platforms, platforms, NULL);

    for (cl_uint i = 0; i < num_platforms; i++) {
        {
            size_t ext_size;
            clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, 0, NULL, &ext_size);
            char *name = (char*)malloc(ext_size);
            clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, ext_size, name, NULL);

            char *version = (char*)malloc(ext_size);
            clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, ext_size, version, NULL);

            printf("The name of the platform is <%s> with version <%s>.\n", name, version);

            free(name);
            free(version);            
        }

        // Get the devices
        cl_device_id device;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL);

        //Create the context
        cl_context context = clCreateContext(0, 1, &device, NULL, NULL, &err_code);

        // Get Kernel.
        cl_int err_code;
        cl_program program = clCreateProgramWithSource(context, 1, (const char **)&program_source, &program_length, &err_code);
        clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

        cl_kernel kernel = clCreateKernel(program, "VectorAdd", &err_code);

        // Allocate the OpenCL buffer memory objects for source and result on the device GMEM
        cl_mem d_src1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * num_elements, NULL, &err_code);
        cl_mem d_src2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * num_elements, NULL, &err_code);
        cl_mem d_dst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * num_elements, NULL, &err_code);
        // Set the Argument values
        clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&d_src1);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&d_src2);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&d_dst);
        clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&num_elements);

        //--------------------------------------------------------
        // Create a command-queue
        // CL_QUEUE_PROFILING_ENABLE: Opencl operation time can only be calculated when this flag bit is turned on
        cl_command_queue command_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err_code);

        // Asynchronous write of data to GPU device
        clEnqueueWriteBuffer(command_queue, d_src1, CL_FALSE, 0, sizeof(cl_float) * num_elements, h_src1, 0, NULL, NULL);
        clEnqueueWriteBuffer(command_queue, d_src2, CL_FALSE, 0, sizeof(cl_float) * num_elements, h_src2, 0, NULL, NULL);

        // Launch kernel
        cl_event ev;
        clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &ev);

        // Synchronous/blocking read of results, and check accumulated errors
        clEnqueueReadBuffer(command_queue, d_dst, CL_TRUE, 0, sizeof(cl_float) * num_elements, h_dst4cl, 0, NULL, NULL);
        
        // Block until all tasks in command_queue have been completed.
        clFinish(command_queue);
      
        // Gets the running time of the kernel function.
        cl_ulong start_time = 0, end_time = 0;
        // It requires that the CL_QUEUE_PROFILING_ENABLE flag is set in the clCreateCommandQueue function.
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
        printf("Command elapsed time: %f ms\n", (end_time - start_time)*1e-6);
        //--------------------------------------------------------

        // Compute and compare results on host.
        VectorAddHost(h_src1, h_src2, h_dst, num_elements);
        bool is_equal = true;
        for (size_t ni = 0; ni < num_elements; ni++) {
            if (h_dst[ni] != h_dst4cl[ni]) {
                is_equal = false;
                break;
            }
        }
        printf("Test: %s \n \n", (is_equal ? "PASS":"FAILED"));

        // Cleanup
        if (kernel) clReleaseKernel(kernel);
        if (command_queue) clReleaseCommandQueue(command_queue);
        if (context) clReleaseContext(context);
        if (d_src1) clReleaseMemObject(d_src1);
        if (d_src2) clReleaseMemObject(d_src2);
        if (d_dst) clReleaseMemObject(d_dst);
        
        if (program) clReleaseProgram(program);
    }

    if (program_source) free(program_source);

    if (h_src1) free(h_src1);
    if (h_src2) free(h_src2);
    if (h_dst) free(h_dst);
    if (h_dst4cl) free(h_dst4cl);

    return 0;
}
