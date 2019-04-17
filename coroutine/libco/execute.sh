#!/bin/bash
# Usage: sh execute.sh demo.cpp

source_file=$1
target_name=${source_file%.*}".o"

echo ${target_name}

if [ -f ${target_name} ];then
  rm ${target_name}
fi

g++ ${source_file} libco/build/libcolib.a  -l pthread -l dl -I libco/include/ -o ${target_name}
./${target_name}
