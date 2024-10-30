/*!
* \brief 
*/

#include "pocket-ai/engine/cl/engine.hpp"

using namespace pai;

void SetParams4Gemm(cl::KernelParams *params) {    
    params->io_attri = {
        {0,                 false,  sizeof(uint32_t)}, // 0
        {0,                 false,  sizeof(uint32_t)}, // 1
        {0,                 false,  sizeof(uint32_t)}, // 2
        {CL_MEM_READ_ONLY,  false,  sizeof(cl_mem)},   // 3
        {0,                 false,  sizeof(uint32_t)}, // 4
        {CL_MEM_READ_ONLY,  false,  sizeof(cl_mem)},   // 5
        {0,                 false,  sizeof(uint32_t)}, // 6
        {CL_MEM_WRITE_ONLY, false,  sizeof(cl_mem)},   // 7
        {0,                 false,  sizeof(uint32_t)}  // 8
    };
}

void SetImageParams4Gemm(cl::KernelParams *params) {    
    params->io_attri = {
        {0,                 false,  sizeof(uint32_t)}, // 0
        {0,                 false,  sizeof(uint32_t)}, // 1
        {0,                 false,  sizeof(uint32_t)}, // 2
        {CL_MEM_READ_ONLY,  true,   sizeof(cl_mem)},   // 3
        {0,                 false,  sizeof(uint32_t)}, // 4
        {CL_MEM_READ_ONLY,  true,   sizeof(cl_mem)},   // 5
        {0,                 false,  sizeof(uint32_t)}, // 6
        {CL_MEM_WRITE_ONLY, true,   sizeof(cl_mem)},   // 7
        {0,                 false,  sizeof(uint32_t)}  // 8
    };
}

void TestGemm(cl::Engine *engine, std::string kernel_name, int step, bool transpose_a, bool use_image = false) {
    PAI_LOGS(">>> %s: ", kernel_name.c_str());

    cl::Kernel *kernel = engine->GetKernel(kernel_name, true);

    // 1024
    uint32_t height_a = 960, width_a = 1280;
    uint32_t height_b = 1280, width_b = 6400;
    // set and log Global and Local work size dimensions
    size_t local_work_size[2] = {16, 16}; // x, y
    size_t global_work_size[2] =
        {cl::GetRoundUpMultiple(width_b/step, local_work_size[0]) * local_work_size[0],
         cl::GetRoundUpMultiple(height_a/step, local_work_size[1]) * local_work_size[1]};
    PAI_LOGS("(%zu, %zu). ", global_work_size[0], global_work_size[1]);

    if (!use_image) {
        uint32_t lda = width_a;
        if (transpose_a) lda = height_a;

        std::vector<size_t> size = {
            height_a, width_b, width_a,                     // 0-M 1-N 2-K
            sizeof(cl_float) * height_a * width_a, lda,     // 3-A 4-lda
            sizeof(cl_float) * height_b * width_b, width_b, // 5-B 6-ldb
            sizeof(cl_float) * height_a * width_b, width_b  // 7-C 8-ldc
        };  
        kernel->CreateBuffer(size);      
    }
    else {
        const uint32_t image_chn = 4; // RGBA
        uint32_t lda = width_a;
        cl::BufferArgs a_arg = {};
        if (transpose_a) {
            lda = height_a;
            a_arg = {sizeof(cl_float), width_a, height_a/image_chn};
        }
        else {
            lda = width_a;
            a_arg = {sizeof(cl_float), height_a, width_a/image_chn};
        }
        std::vector<cl::BufferArgs> args = {
            {height_a}, {width_b}, {width_a},                 // 0-M 1-N 2-K
            a_arg, {lda},                                     // 3-A 4-lda
            {sizeof(cl_float), height_b, width_b/image_chn}, {width_b}, // 5-B 6-ldb
            {sizeof(cl_float), height_a, width_b/image_chn}, {width_b}  // 7-C 8-ldc
        };
        kernel->CreateBuffer(args);
    }

    cl_float *hA_map = (cl_float *)kernel->MapBuffer(CL_TRUE, 3); // A
    cl_float *hB_map = (cl_float *)kernel->MapBuffer(CL_TRUE, 5); // B
    for (uint32_t i = 0; i < height_a * width_a; i++)
        hA_map[i] = 1.2345f+i/23.f+i%12;
    for (uint32_t i = 0; i < height_b * width_b; i++)
        hB_map[i] = 2.3456f+i/33.f+i%22;

    if (transpose_a) {
        cl_float *temp = (float *)malloc(height_a * width_a * sizeof(cl_float));
        memcpy(temp, hA_map, height_a * width_a * sizeof(cl_float));
        for (uint32_t i = 0; i < height_a; i++) {
            for (uint32_t j = 0; j < width_a; j++) {
                hA_map[j*height_a+i] = temp[i*width_a+j];
            }
        }
        free(temp);
    }
    kernel->UnmapBuffer(3);
    kernel->UnmapBuffer(5);

    // Launch kernel
    engine->AsyncRun(kernel, 2, global_work_size, local_work_size, true);

    cl_float *hC_map = (cl_float *)kernel->MapBuffer(CL_TRUE, 7);

    float mean = 0.f;
    for (uint32_t i = 0; i < height_a; i++) {
        // PAI_LOGS("\n");
        for (uint32_t j = 0; j < width_b; j++) {
            mean += hC_map[i * width_b + j];
            // PAI_LOGS("%f, ", hC_map[i * width_b + j]);
            // if (hC_map[i * width_b + j] != 3200)
            //     PAI_LOGS("%f(%d, %d), ", hC_map[i * width_b + j], i, j);
        }
        
    }
    PAI_LOGS(" <<< Out: %f.\n", mean / (height_a * width_b));
    kernel->UnmapBuffer(7);
    kernel->ReleaseBuffer();
    engine->FinishQueue();
}

int main(int argc, char **argv) {
    std::vector<std::tuple<std::string, std::string, cl::pSetParamsFuncs>> kernels_name;
    // kernels_name.push_back(std::make_tuple("dot_product", "DotProductDevice", SetParams4DotProduct));
    kernels_name.push_back(std::make_tuple("gemm_fp32", "GemmDeviceV1", SetParams4Gemm));
    kernels_name.push_back(std::make_tuple("gemm_fp32", "GemmDeviceV2", SetParams4Gemm));
    kernels_name.push_back(std::make_tuple("gemm_fp32", "GemmDeviceV3", SetParams4Gemm));
    kernels_name.push_back(std::make_tuple("gemm_fp32", "GemmDeviceV4", SetParams4Gemm));
    kernels_name.push_back(std::make_tuple("gemm_fp32", "GemmDeviceV5_0", SetParams4Gemm));
    kernels_name.push_back(std::make_tuple("gemm_fp32", "GemmDeviceV5_1", SetParams4Gemm));
    kernels_name.push_back(std::make_tuple("gemm_fp32", "GemmDeviceV5_2", SetParams4Gemm));

    kernels_name.push_back(std::make_tuple("gemm_mobile_fp32", "GemmMobileDeviceV1", SetParams4Gemm));
    kernels_name.push_back(std::make_tuple("gemm_mobile_fp32", "GemmMobileDeviceV2", SetParams4Gemm));
    kernels_name.push_back(std::make_tuple("gemm_mobile_fp32", "GemmMobileDeviceV3_0", SetParams4Gemm));
    kernels_name.push_back(std::make_tuple("gemm_mobile_fp32", "GemmMobileDeviceV3_1", SetParams4Gemm));
    kernels_name.push_back(std::make_tuple("gemm_mobile_fp32", "GemmMobileDeviceV3_2", SetParams4Gemm));
    kernels_name.push_back(std::make_tuple("gemm_mobile_fp32", "GemmMobileDeviceV4", SetImageParams4Gemm));
    kernels_name.push_back(std::make_tuple("gemm_mobile_fp32", "GemmMobileDeviceV5", SetImageParams4Gemm));

    cl::Engine engine;
    engine.Init("../", kernels_name, 0, true);

    PAI_LOGS("\n##############################\n");

    TestGemm(&engine, "GemmDeviceV1", 1, false);
    TestGemm(&engine, "GemmDeviceV2", 1, false);
    TestGemm(&engine, "GemmDeviceV3", 2, false);
    TestGemm(&engine, "GemmDeviceV4", 4, false);
    TestGemm(&engine, "GemmDeviceV5_0", 4, false);
    TestGemm(&engine, "GemmDeviceV5_1", 4, false);
    TestGemm(&engine, "GemmDeviceV5_2", 4, false);

    TestGemm(&engine, "GemmMobileDeviceV1", 1, false);
    TestGemm(&engine, "GemmMobileDeviceV2", 4, false);
    TestGemm(&engine, "GemmMobileDeviceV3_0", 4, false);
    TestGemm(&engine, "GemmMobileDeviceV3_1", 4, false);
    TestGemm(&engine, "GemmMobileDeviceV3_2", 4, true);
    TestGemm(&engine, "GemmMobileDeviceV4", 4, true, true);
    TestGemm(&engine, "GemmMobileDeviceV5", 8, true, true);

    engine.Deinit();
    PAI_LOGS("Exit...\n");
    return 0;
}