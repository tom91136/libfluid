
#ifndef LIBFLUID_SPH_H
#define LIBFLUID_SPH_H

#ifndef __OPENCL_C_VERSION__

#include <CL/cl_platform.h>

typedef cl_float4 float4;
typedef cl_float3 float3;
typedef cl_int3 int3;
typedef cl_char3 char3;
typedef cl_uint uint;
typedef cl_uint2 uint2;
typedef cl_uint3 uint3;

#else


#endif

typedef struct Entry {
	int3 key;
	size_t value;
} __attribute__ ((aligned)) Entry;

typedef struct ClSphConfig {
	float scale;
	float dt;
	size_t iteration;
	float3 constForce;
}  __attribute__ ((aligned)) ClSphConfig;

typedef enum ClSphType {
	Fluid, Obstacle
} ClSphType;

typedef struct ClSphParticle {
	size_t id;
	ClSphType type;
	float mass;
	float3 position;
	float3 velocity;
}  __attribute__ ((aligned)) ClSphParticle;

typedef struct ClSphAtom {

	ClSphParticle particle;

	float3 pStar;
	float3 deltaP;
	float3 omega;
	float lambda;
	size_t zIndex;
}  __attribute__ ((aligned)) ClSphAtom;



#endif //LIBFLUID_SPH_H
