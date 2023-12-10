
adb push a.out /data/local/tmp/gemm/a.out
adb push libc++_shared.so /data/local/tmp/gemm/libc++_shared.so

adb shell "chmod 777 -R ./data/local/tmp/gemm/ && export LD_LIBRARY_PATH=/data/local/tmp/gemm/:$LD_LIBRARY_PATH && ./data/local/tmp/gemm/a.out"
