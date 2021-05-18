#include "nbody_gpu.h"

__global__
void bodyForce_v1(Body *p, float dt, int n) {
  int i_id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_i = gridDim.x * blockDim.x;
  
  for (int i = i_id; i < n; i+=stride_i) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].pos.x - p[i].pos.x;
      float dy = p[j].pos.y - p[i].pos.y;
      float dz = p[j].pos.z - p[i].pos.z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = rsqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;      

      Fx += dx * invDist3; 
      Fy += dy * invDist3; 
      Fz += dz * invDist3;
    }
    
    p[i].vel.x += dt*Fx; 
    p[i].vel.y += dt*Fy; 
    p[i].vel.z += dt*Fz;
  }
}

__global__
void integrateBody(Body *p, float dt, int n){
  int i_id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_i = gridDim.x * blockDim.x;
  
  for (int i = i_id; i < n; i+=stride_i) {
    p[i].pos.x += p[i].vel.x*dt;
    p[i].pos.y += p[i].vel.y*dt;
    p[i].pos.z += p[i].vel.z*dt;
  }
}

__device__ 
float3 bodyBodyInteraction(float3 ai, float3 bi, float3 bj) {
    float3 r;

    r.x = bi.x - bj.x;
    r.y = bi.y - bj.y;
    r.z = bi.z - bj.z;
    float distSqr = r.x*r.x + r.y*r.y + r.z*r.z + SOFTENING;
    float invDist = rsqrtf(distSqr);
    float invDist3 = invDist * invDist * invDist;      

    ai.x += r.x * invDist3;
    ai.y += r.y * invDist3;
    ai.z += r.z * invDist3;

    return ai;
}

__device__ 
float3 gravitation(float3 myPos, float3 accel)
{
    extern __shared__ float3 sharedPos[];
    int i;

    for (i = 0; i < blockDim.x; ) 
    {
        accel = bodyBodyInteraction(accel, SX(i), myPos); i += 1;
        // Here we unroll the loop if needed
        //accel = bodyBodyInteraction(accel, SX(i), myPos); i += 1;
        //accel = bodyBodyInteraction(accel, SX(i), myPos); i += 1;
        //accel = bodyBodyInteraction(accel, SX(i), myPos); i += 1;
    }

    return accel;
}

template <bool multithreadBodies>
__device__
float3 bodyForceSM(float3 pos, Body *p, float dt, int n) {
  extern __shared__ float3 sharedPos[];
  
  float3 acc = {0.0f, 0.0f, 0.0f};
  
  for (int i = 0; i < gridDim.x; i++) {
    auto data = p[i*blockDim.x + threadIdx.x];
    sharedPos[threadIdx.x+blockDim.x*threadIdx.y] = data.pos;
    __syncthreads();

    // This is the "tile_calculation" function from the GPUG3 article.
    acc = gravitation(pos, acc);
    __syncthreads();
    
  }

  if (multithreadBodies)
  {
      SX_SUM(threadIdx.x, threadIdx.y) = acc;

      __syncthreads();

      // Save the result in global memory for the integration step
      if (threadIdx.y == 0) {
          for (int i = 1; i < blockDim.y; i++) {
              acc.x += SX_SUM(threadIdx.x,i).x;
              acc.y += SX_SUM(threadIdx.x,i).y;
              acc.z += SX_SUM(threadIdx.x,i).z;
          }
      }
  }

  return acc;
}

template <bool multithreadBodies>
__global__
void integrateBodySM(Body *p, float dt, int n){
  int i_id = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i_id < n){
    Body bodyCurrent = p[i_id];   
    float3 pos = bodyCurrent.pos;
    float3 vel = bodyCurrent.vel;
    
    float3 force = bodyForceSM<multithreadBodies>(pos, p, dt, n);    
    vel.x += force.x * dt;
    vel.y += force.y * dt;
    vel.z += force.z * dt;  
        
    // new position = old position + velocity * deltaTime
    pos.x += vel.x * dt;
    pos.y += vel.y * dt;
    pos.z += vel.z * dt;

    // store new position and velocity
    p[i_id].pos = pos;
    p[i_id].vel = vel;
  }
}

template __device__ float3 bodyForceSM<false>(float3 pos, Body *p, float dt, int n);
template __device__ float3 bodyForceSM<true>(float3 pos, Body *p, float dt, int n);
template __global__ void integrateBodySM<false>(Body *p, float dt, int n);
template __global__ void integrateBodySM<true>(Body *p, float dt, int n);