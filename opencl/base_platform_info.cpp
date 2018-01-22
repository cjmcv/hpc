/*!
* \brief Query OpenCL platform information.
*/

#include <iostream>
#include <CL/cl.h>

#define OCL_CHECK(condition) \
  do { \
    cl_int error = condition; \
    if (error != CL_SUCCESS) { \
      fprintf(stderr, "OCL_CHECK error in line %d of file %s : code %d \n", \
              __LINE__, __FILE__, error); \
      exit(EXIT_FAILURE); \
    } \
  } while(0);

int main() {
  // Platform index.
  cl_platform_id *platforms;

  // Query how many devices support OpenCL.
  cl_uint num_platforms;
  OCL_CHECK(clGetPlatformIDs(5, NULL, &num_platforms));
  printf("There are ( %d ) platforms that support OpenCL.\n\n", num_platforms);

  // Giving the number of platform to get platforms' index.
  platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
  OCL_CHECK(clGetPlatformIDs(num_platforms, platforms, NULL));

  // Get information about the OpenCL platforms.
  // Traverse all platforms. 
  for (cl_int i = 0; i < num_platforms; i++) {
    // Get buffer size.
    size_t ext_size;
    OCL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS,
      0, NULL, &ext_size));

    // Get the extensions supported on the platform.
    char* ext_data = (char*)malloc(ext_size);
    OCL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS,
      ext_size, ext_data, NULL));
    printf("The extensions supported on the platform %d includes : %s\n", i, ext_data);

    // Get the platform's name. 
    char *name = (char*)malloc(ext_size);
    OCL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,
      ext_size, name, NULL));
    printf("The name of platform %d is : %s\n", i, name);

    // Get the producer name of the video card. 
    char *vendor = (char*)malloc(ext_size);
    OCL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR,
      ext_size, vendor, NULL));
    printf("The producer of platform %d is : %s\n", i, vendor);

    // Get information about the platform.
    char *version = (char*)malloc(ext_size);
    OCL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION,
      ext_size, version, NULL));
    printf("The version information of platform %d is : %s\n", i, version);

    // Check whether the video card is an embedded video card or not.
    // Only full profile or embeded profile.
    char *profile = (char*)malloc(ext_size);
    OCL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE,
      ext_size, profile, NULL));
    printf("The profile of platform %d is : %s\n", i, profile);

    free(ext_data);
    free(name);
    free(vendor);
    free(version);
    free(profile);

    printf("\n\n");
  }

  free(platforms);
  return 0;
}