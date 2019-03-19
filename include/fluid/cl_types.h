#ifndef LIBFLUID_CL_TYPES
#define LIBFLUID_CL_TYPES

#ifndef __OPENCL_C_VERSION__

#include <CL/cl_platform.h>

// non CL compiler
typedef cl_float4 float4;
typedef cl_float3 float3;
typedef cl_int3 int3;
typedef cl_char3 char3;
typedef cl_uint uint;
typedef cl_uint2 uint2;
typedef cl_uint3 uint3;

#define M_PI_F 3.1415926f
#define global
#define kernel
#define constant

#else

#define global global
#define kernel kernel
#define constant constant

#endif


#if defined(_MSC_VER)
#define ALIGNED_(x) __declspec(align(x))
#else
#define ALIGNED_(x) __attribute__ ((aligned(x)))
#endif

#define _ALIGNED_TYPE(t, x) typedef t ALIGNED_(x)


#endif //LIBFLUID_CL_TYPES
