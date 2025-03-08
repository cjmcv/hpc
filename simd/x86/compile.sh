#!/usr/bin/env bash

# valgrind --leak-check=full --show-leak-kinds=all ./a.out
# 自动检测：-march=native
# 手动指令：-msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mavx -mavx2 -mavx512f -mavx512cd -mavx512er -mavx512pf -mavx512vl -mavx512vnni
g++ -O3 -march=native linear.cpp -o a.out && ./a.out