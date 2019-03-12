
#ifndef LIBFLUID_SPH_H
#define LIBFLUID_SPH_H

#ifndef __OPENCL_C_VERSION__

#include <CL/cl_platform.h>

typedef cl_float4 float4;
typedef cl_float3 float3;
typedef cl_uint uint;

#endif

typedef struct Atom {

	float3 now;
	float3 velocity;

	float3 deltaP;
	float3 omega;

	float mass;
	float lambda;

	int id;
	uint neighbourOffset;
	uint neighbourCount;

}  __attribute__ ((aligned)) Atom;


#endif //LIBFLUID_SPH_H
