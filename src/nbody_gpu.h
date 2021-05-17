#pragma once
#include "nbody_common.h"

using Body = nbody_common::Body;

/* 
   v1: Refactor cpu version of bodyForce into gpu kernel version
*/
__global__ void bodyForce_v1(Body *p, float dt, int n);