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