#include <iostream>
#include "hello_cpu.h"
#include "hello_gpu.h"

int main(){
  helloCPU();
  helloGPU<<<1,1>>>();
  cudaDeviceSynchronize();
}