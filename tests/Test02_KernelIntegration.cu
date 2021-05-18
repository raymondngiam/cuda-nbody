#include "gtest/gtest.h"
#include "nbody_common.h"
#include "nbody_gpu.h"
#include "test_utils.h"
#include <chrono>
#include <fmt/core.h>

namespace{
  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations
} // namespace

void NBody_GPU_V2(int nBodies,bool verbose){
  int bytes = nBodies * sizeof(Body);
  int thread_size = 256;

  Body *baseline = (Body*)malloc(bytes);
  test_utils::GetBaseline(baseline,nBodies);  

  Body *buf = (Body*)malloc(bytes);

  nbody_common::initBodies(buf, nBodies); // Init pos / vel data

  cudaError_t errSync, errAsync;

  Body *p;
  cudaMallocManaged((void**)&p, bytes);
  errSync = cudaGetLastError();
  if (errSync!=cudaSuccess){printf("Malloc error: %s\n",cudaGetErrorString(errSync));}
  memcpy(p, buf, bytes);

  auto st0 = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < nIters; iter++) {
    dim3 num_of_blocks((nBodies-1)/thread_size + 1,1,1);
    dim3 threads_per_block(thread_size,1,1);
    bodyForce_v1<<<num_of_blocks,threads_per_block>>>(p, dt, nBodies); // compute interbody forces
    errSync = cudaGetLastError();
    if (errSync!=cudaSuccess){printf("Sync error: %s\n",cudaGetErrorString(errSync));}
    errAsync = cudaDeviceSynchronize();
    if (errAsync!=cudaSuccess){printf("Async error: %s\n",cudaGetErrorString(errAsync));}

    integrateBody<<<num_of_blocks,threads_per_block>>>(p, dt, nBodies);
    errSync = cudaGetLastError();
    if (errSync!=cudaSuccess){printf("Sync error: %s\n",cudaGetErrorString(errSync));}
    errAsync = cudaDeviceSynchronize();
    if (errAsync!=cudaSuccess){printf("Async error: %s\n",cudaGetErrorString(errAsync));}
  }
  auto st1 = std::chrono::high_resolution_clock::now();
  auto elapsed_ms = 1e-6 * (st1-st0).count()/ nIters ;
  fmt::print("{:d} Bodies: average execution time = {:.6f} milliseconds\n",nBodies,elapsed_ms);

  float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / (elapsed_ms/1000.0);
  fmt::print("{:d} Bodies: average {:0.6f} Billion Interactions / second\n",nBodies,billionsOfOpsPerSecond);

  if (verbose){
    fmt::print("Output:\n");
    test_utils::PrintNBodies(p,nBodies);
    fmt::print("\n");
    fmt::print("Baseline:\n");
    test_utils::PrintNBodies(baseline,nBodies);
  }
  EXPECT_TRUE(test_utils::AlmostEqual(p,baseline,nBodies,1e-4));

  cudaFree(p);
  free(buf);
}

TEST(Test02_KernelIntegration,TwoBodiesTest){
  const int nBodies = 2;
  NBody_GPU_V2(nBodies, true);
}

TEST(Test02_KernelIntegration,EightBodiesTest){
  const int nBodies = 8;
  NBody_GPU_V2(nBodies, true);
}

TEST(Test02_KernelIntegration,_32BodiesTest){
  const int nBodies = 32;
  NBody_GPU_V2(nBodies, false);
}

TEST(Test02_KernelIntegration,_4096BodiesTest){
  const int nBodies = 4096;
  NBody_GPU_V2(nBodies, false);
}