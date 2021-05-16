#include <iostream>
#include <fstream>
#include <chrono>
#include <math.h>
#include <stdlib.h> 
#include <fmt/core.h>
#include <fmt/printf.h>
#include "nbody_common.h"

using Body = nbody_common::Body;

int main(const int argc, const char** argv) {

  if (argc!=2){
    fmt::print("Invalid input argument.\n");
    fmt::print("Please input number of n-body as program argument.\n");
    return -1;
  }  
  int nBodies =  atoi(argv[1]);
  if (nBodies == 0)
  {
    fmt::print("Invalid input argument.\n");
    fmt::print("Please input number of n-body as program argument.\n");
    return -1;
  }  

  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations

  int bytes = nBodies * sizeof(Body);

  Body *p = (Body*)malloc(bytes);

  nbody_common::initBodies(p, nBodies); // Init pos / vel data
  auto pos0 = p[0].pos;
  auto vel0 = p[0].vel;
  fmt::print("{}-th body\n",0);
  fmt::print("x,y,z:[{:.3f},{:.3f},{:.3f}]\n", pos0.x, pos0.y, pos0.z);
  fmt::print("vx,vy,vz:[{:.3f},{:.3f},{:.3f}]\n", vel0.x, vel0.y, vel0.z);
  auto posN = p[nBodies-1].pos;
  auto velN = p[nBodies-1].vel;
  fmt::print("{}-th body\n",nBodies);
  fmt::print("x,y,z:[{:.3f},{:.3f},{:.3f}]\n", posN.x, posN.y, posN.z);
  fmt::print("vx,vy,vz:[{:.3f},{:.3f},{:.3f}]\n", velN.x, velN.y, velN.z);

  auto st0 = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < nIters; iter++) {

    nbody_common::bodyForce(p, dt, nBodies); // compute interbody forces

    for (int i = 0 ; i < nBodies; i++) { // integrate position
      p[i].pos.x += p[i].vel.x*dt;
      p[i].pos.y += p[i].vel.y*dt;
      p[i].pos.z += p[i].vel.z*dt;
    }
  }
  auto st1 = std::chrono::high_resolution_clock::now();
  fmt::print("Average execution time = {:.6f} milliseconds\n",1e-6 * (st1-st0).count() / nIters);

  std::ofstream f(fmt::format("../data/data_{:04d}.txt",nBodies));
  fmt::fprintf(f,"x,y,z,vx,vy,vz\n");
  for(size_t i=0; i < nBodies; i++){
    fmt::fprintf(f,"%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",\
    p[i].pos.x, \
    p[i].pos.y, \
    p[i].pos.z, \
    p[i].vel.x, \
    p[i].vel.y, \
    p[i].vel.z);
  }

  free(p);
}
