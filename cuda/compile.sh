# ncu --set full --target-processes all -o report ./a.out
# ncu-ui report.ncu-rep
nvcc --optimize 3 -arch=sm_89 -I../ gemm_fp16_wmma.cu -o a.out && ./a.out