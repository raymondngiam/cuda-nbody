#include "gtest/gtest.h"
#include "nbody_common.h"
#include "nbody_gpu.h"
#include "test_utils.h"
#include <chrono>
#include <fmt/core.h>

namespace{
  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations
  constexpr bool multithreadBodies = true;
} // namespace

void NBody_GPU_V5(int nBodies,int P, int Q, bool verbose){
  int bytes = nBodies * sizeof(Body);

  Body *baseline = (Body*)malloc(bytes);
  test_utils::GetBaseline(baseline,nBodies);  

  Body *buf = (Body*)malloc(bytes);

  nbody_common::initBodies(buf, nBodies); // Init pos / vel data

  cudaError_t errSync, errAsync;
  int deviceId;
  cudaGetDevice(&deviceId);

  Body *p;
  cudaMallocManaged((void**)&p, bytes);
  errSync = cudaGetLastError();
  if (errSync!=cudaSuccess){printf("Malloc error [p]: %s\n",cudaGetErrorString(errSync));}

  cudaMemPrefetchAsync(p, bytes, deviceId);  // Prefetch to GPU device.
  memcpy(p, buf, bytes);

  auto st0 = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < nIters; iter++) {
    dim3 num_of_blocks((nBodies-1)/P + 1,1,1);
    dim3 threads_per_block(P,Q,1);
    size_t shared_mem_size = P * Q * sizeof(float3);
    integrateBodySM<multithreadBodies><<<num_of_blocks,threads_per_block, shared_mem_size>>>(p, dt, nBodies);
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

  cudaMemPrefetchAsync(p, bytes, cudaCpuDeviceId);  // Prefetch to host. `cudaCpuDeviceId` is a
                                                    // built-in CUDA variable. 

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

TEST(Test05_SharedMemory2D,TwoBodiesTest){
  const int nBodies = 2;
  int P = 1;
  int Q= 2;
  NBody_GPU_V5(nBodies, P, Q, true);
}

TEST(Test05_SharedMemory2D,EightBodiesTest){
  const int nBodies = 8;
  int P = 4;
  int Q= 2;
  NBody_GPU_V5(nBodies, P, Q, true);
}

TEST(Test05_SharedMemory2D,_32BodiesTest){
  const int nBodies = 32;
  int P = 16;
  int Q= 2;
  NBody_GPU_V5(nBodies, P, Q, false);
}

TEST(Test05_SharedMemory2D,_4096BodiesTest){
  const int nBodies = 4096;
  int P = 256;
  int Q= 4;
  NBody_GPU_V5(nBodies, P, Q, false);
}