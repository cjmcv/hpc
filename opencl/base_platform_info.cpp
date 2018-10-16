/*!
* \brief Query OpenCL platform information.
*/

#include"ocl_util.h"

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

    if (ext_data) free(ext_data);
    if (name) free(name);
    if (vendor) free(vendor);
    if (version) free(version);
    if (profile) free(profile);

    printf("\n\n");
  }

  if (platforms) free(platforms);
  return 0;
}