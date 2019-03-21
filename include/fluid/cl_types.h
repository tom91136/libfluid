#ifndef LIBFLUID_CL_TYPES
#define LIBFLUID_CL_TYPES

#ifndef __OPENCL_C_VERSION__

#include <CL/cl_platform.h>

// non CL compiler
typedef cl_float4 float4;
typedef cl_float3 float3;
typedef cl_int3 int3;
typedef cl_int4 int4;
typedef cl_char3 char3;
typedef cl_uint uint;
typedef cl_uchar uchar;
typedef cl_uint2 uint2;
typedef cl_uint3 uint3;
typedef cl_uint4 uint4;

#define M_PI_F 3.1415926f
#define global
#define kernel
#define constant
#define __private

#if defined(_MSC_VER)
#define ALIGNED_(x) __declspec(align(x))
#else
#define ALIGNED_(x) __attribute__ ((aligned(x)))
#endif

#else

#define global global
#define kernel kernel
#define constant constant

#define ALIGNED_(x) __attribute__ ((aligned(x)))

#endif

#endif //LIBFLUID_CL_TYPES
