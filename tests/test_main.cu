#include "gtest/gtest.h"
#include "nbody_common.h"
#include "nbody_gpu.h"
#include <math.h>
#include <string>
#include <fstream>
#include <fmt/core.h>
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>

void PrintNBodies(Body *p, int nBodies){
  for (int i = 0 ; i < nBodies; i++) {
    fmt::print("{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n", 
      p[i].pos.x, 
      p[i].pos.y, 
      p[i].pos.z, 
      p[i].vel.x, 
      p[i].vel.y, 
      p[i].vel.z);
  }
}

bool AlmostEqual(Body *p, Body *q, int nBodies, float epsilon){
  bool almostEqual = true;
  float *p_float = (float*)p;
  float *q_float = (float*)q;
  for (int i = 0 ; i < 6*nBodies; i++) {
    if (fabs(p_float[i]-q_float[i]) > epsilon)
    {
      almostEqual = false;
      break;
    }      
  }
  return almostEqual;
}

TEST(Test01_RefactorBodyForce,TwoBodiesTest){
  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations
  const int nBodies = 2;

  int bytes = nBodies * sizeof(Body);

  Body *baseline = (Body*)malloc(bytes);

  std::ifstream f(fmt::format("../data/data_{:04d}.txt",nBodies));
  std::string line;
  boost::regex pat( "(.*),(.*),(.*),(.*),(.*),(.*)" );
  std::getline(f,line);  //skip header line

  int index=0;
  while(std::getline(f,line)){
    boost::smatch matches;
    if (boost::regex_match(line, matches, pat)){
      baseline[index].pos.x = boost::lexical_cast<float>(matches[1]);
      baseline[index].pos.y = boost::lexical_cast<float>(matches[2]);
      baseline[index].pos.z = boost::lexical_cast<float>(matches[3]);
      baseline[index].vel.x = boost::lexical_cast<float>(matches[4]);
      baseline[index].vel.y = boost::lexical_cast<float>(matches[5]);
      baseline[index].vel.z = boost::lexical_cast<float>(matches[6]);
      index += 1;
    }
  }  

  Body *buf = (Body*)malloc(bytes);

  nbody_common::initBodies(buf, nBodies); // Init pos / vel data

  cudaError_t errSync, errAsync;

  Body *p;
  cudaMallocManaged((void**)&p, bytes);
  errSync = cudaGetLastError();
  if (errSync!=cudaSuccess){printf("Malloc error: %s\n",cudaGetErrorString(errSync));}
  memcpy(p, buf, bytes);

  for (int iter = 0; iter < nIters; iter++) {
    dim3 num_of_blocks(32,1,1);
    dim3 threads_per_block(256,1,1);
    bodyForce_v1<<<num_of_blocks,threads_per_block>>>(p, dt, nBodies); // compute interbody forces
    errSync = cudaGetLastError();
    if (errSync!=cudaSuccess){printf("Sync error: %s\n",cudaGetErrorString(errSync));}
    errAsync = cudaDeviceSynchronize();
    if (errAsync!=cudaSuccess){printf("Async error: %s\n",cudaGetErrorString(errAsync));}

    for (int i = 0 ; i < nBodies; i++) { // integrate position
      p[i].pos.x += p[i].vel.x*dt;
      p[i].pos.y += p[i].vel.y*dt;
      p[i].pos.z += p[i].vel.z*dt;
    }
  }
  fmt::print("Output:\n");
  PrintNBodies(p,nBodies);
  fmt::print("\n");
  fmt::print("Baseline:\n");
  PrintNBodies(baseline,nBodies);
  EXPECT_TRUE(AlmostEqual(p,baseline,nBodies,1e-4));

  cudaFree(&p);
  free(buf);
}