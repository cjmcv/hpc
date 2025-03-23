# ncu --set full --target-processes all -o report ./a.out
# ncu-ui report.ncu-rep
# nvcc -ptx --optimize 3 -arch=sm_89 -I../ reduce_fp32.cu -o a.ptx
nvcc --optimize 3 -arch=sm_89 -I../ reduce_fp32.cu -o a.out && ./a.out