#pragma once
#include <math.h>

#define SOFTENING 1e-9f

namespace nbody_common
{
/*
 * Each body contains `pos` and `vel` variables,
 * whereby each contains x, y, and z components.
 */

typedef struct { float3 pos, vel; } Body;

/*
 * Initialize `pos` with vector (n, n, n), where n is the index of the body
 * and `vel` with vector (1,1,1).
 */

void initBodies(Body *data, int n);

/*
 * This function calculates the gravitational impact of all bodies in the system
 * on all others, but does not update their positions.
 */

void bodyForce(Body *p, float dt, int n);
} // namespace nbody_common