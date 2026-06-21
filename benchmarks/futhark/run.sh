#!/bin/bash

export LIBRARY_PATH=/usr/local/cuda-13.2/targets/x86_64-linux/lib/stubs:/usr/local/cuda-13.2/targets/x86_64-linux/lib
export CPATH=/usr/local/cuda-13.2/targets/x86_64-linux/include

mkdir -p data

printf "SOFTMAX\n"

SIZES=(1024 4096 16384)

for size in ${SIZES[@]}; do
  # ParPy
  nsys profile -f true -o data/parpy.nsys-rep -- python softmax.py run -n $size -o data/parpy-softmax-$size.txt > /dev/null
  nsys stats -q --force-export=true --format json --report cuda_gpu_kern_sum data/parpy.nsys-rep > data/parpy-softmax-$size.json

  # Generate input data for Futhark
  python softmax.py generate-input -o data/futhark-softmax-in.txt -n $size

  # Futhark
  futhark cuda --library softmax.fut
  nvcc -x c softmax_runner.c -x c softmax.c -O3 -lcuda -lcudart -lnvrtc -o softmax -arch=native
  nsys profile -f true -o data/futhark.nsys-rep -- ./softmax 100 $size < data/futhark-softmax-in.txt > /dev/null 2> data/futhark-softmax-$size.txt
  nsys stats -q --force-export=true --format json --report cuda_gpu_kern_sum data/futhark.nsys-rep > data/futhark-softmax-$size.json
done

python softmax.py print-results ${SIZES[@]}

printf "\nSTENCIL\n"

NX=(128 512)
NY=(128 512)
NZ=(32 64)
ITERATIONS=100

for i in {0..1}; do
  # Naive ParPy
  nsys profile -f true -o data/parpy.nsys-rep -- python stencil.py run --output-file data/parpy-stencil-${NX[$i]}.txt --nx ${NX[$i]} --ny ${NY[$i]} --nz ${NZ[$i]} --iterations $ITERATIONS > /dev/null
  nsys stats -q --force-export=true --format json --report cuda_gpu_kern_sum data/parpy.nsys-rep > data/parpy-stencil-${NX[$i]}.json

  # Optimized ParPy
  nsys profile -f true -o data/parpy.nsys-rep -- python stencil.py run --optimized --output-file data/parpy-opt-stencil-${NX[$i]}.txt --nx ${NX[$i]} --ny ${NY[$i]} --nz ${NZ[$i]} --iterations $ITERATIONS > /dev/null
  nsys stats -q --force-export=true --format json --report cuda_gpu_kern_sum data/parpy.nsys-rep > data/parpy-opt-stencil-${NX[$i]}.json

  # Generate input data for Futhark
  python stencil.py generate-input -o data/futhark-stencil-in.txt --nx ${NX[$i]} --ny ${NY[$i]} --nz ${NZ[$i]} --iterations $ITERATIONS

  # Futhark
  futhark cuda stencil.fut
  nsys profile -f true -o data/futhark.nsys-rep -- ./stencil < data/futhark-stencil-in.txt > data/futhark-stencil-${NX[$i]}.txt
  nsys stats -q --force-export=true --format json --report cuda_gpu_kern_sum data/futhark.nsys-rep > data/futhark-stencil-${NX[$i]}.json
done

python stencil.py print-results --iterations $ITERATIONS ${NX[@]}
