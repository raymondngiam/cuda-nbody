#include "hello_gpu.h"

__global__
void helloGPU()
{
  printf("Hello also from the GPU.\n");
}