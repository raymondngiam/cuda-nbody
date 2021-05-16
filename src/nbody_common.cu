#include "nbody_common.h"

void nbody_common::initBodies(Body *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i].pos.x = (float)i;
    data[i].pos.y = (float)i;
    data[i].pos.z = (float)i;
    data[i].vel.x = 1.0f;
    data[i].vel.y = 1.0f;
    data[i].vel.z = 1.0f;
  }
}

/*
 * This function calculates the gravitational impact of all bodies in the system
 * on all others, but does not update their positions.
 */

void nbody_common::bodyForce(Body *p, float dt, int n) {
  for (int i = 0; i < n; ++i) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].pos.x - p[i].pos.x;
      float dy = p[j].pos.y - p[i].pos.y;
      float dz = p[j].pos.z - p[i].pos.z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = rsqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    p[i].vel.x += dt*Fx; p[i].vel.y += dt*Fy; p[i].vel.z += dt*Fz;
  }
}