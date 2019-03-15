
#ifndef LIBFLUID_SPH_H
#define LIBFLUID_SPH_H

#ifndef __OPENCL_C_VERSION__

#include <CL/cl_platform.h>

typedef cl_float4 float4;
typedef cl_float3 float3;
typedef cl_int3 int3;
typedef cl_uint uint;
typedef cl_uint2 uint2;

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
}  __attribute__ ((aligned)) ClSphConfig;

typedef enum ClSphType {
	Fluid, Obstacle
} ClSphType;


typedef struct ClSphAtom {

	size_t id;
	ClSphType type;
	float mass;
	float3 position;
	float3 velocity;

	float3 now;
	float3 deltaP;
	float3 omega;
	float lambda;

	size_t neighbourOffset;
	size_t neighbourCount;

}  __attribute__ ((aligned)) ClSphAtom;

typedef struct ClSphResult {

	float3 position;
	float3 velocity;

}  __attribute__ ((aligned)) ClSphResult;


#endif //LIBFLUID_SPH_H
