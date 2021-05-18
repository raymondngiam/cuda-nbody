#pragma once
#include "nbody_common.h"

using Body = nbody_common::Body;

// Macros to simplify shared memory addressing
#define SX(i) sharedPos[i+blockDim.x*threadIdx.y]
// This macro is only used when multithreadBodies is true (below)
#define SX_SUM(i,j) sharedPos[i+blockDim.x*j]

/* 
   bodyForce_v1: 
   Refactor cpu version of bodyForce into gpu kernel version
*/
__global__ void bodyForce_v1(Body *p, float dt, int n);

/* 
   integrateBody: 
   GPU kernel version for integrating body forces
*/
__global__ void integrateBody(Body *p, float dt, int n);

__device__ float3 bodyBodyInteraction(float3 ai, float3 bi, float3 bj);

__device__ 
float3 gravitation(float3 myPos, float3 accel);

template <bool multithreadBodies>
__device__
float3 bodyForceSM(float3 pos, Body *p, float dt, int n);

template <bool multithreadBodies>
__global__
void integrateBodySM(Body *p, float dt, int n);