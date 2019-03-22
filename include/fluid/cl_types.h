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

#define NEW_ERROR_TYPE(ERR) {ERR, #ERR}

typedef struct ClErrorType {
	cl_int code;
	const char *name;
} ClErrorType;

const static ClErrorType CL_ERROR_LUT[63] = {
		NEW_ERROR_TYPE(CL_SUCCESS),
		NEW_ERROR_TYPE(CL_DEVICE_NOT_FOUND),
		NEW_ERROR_TYPE(CL_DEVICE_NOT_AVAILABLE),
		NEW_ERROR_TYPE(CL_COMPILER_NOT_AVAILABLE),
		NEW_ERROR_TYPE(CL_MEM_OBJECT_ALLOCATION_FAILURE),
		NEW_ERROR_TYPE(CL_OUT_OF_RESOURCES),
		NEW_ERROR_TYPE(CL_OUT_OF_HOST_MEMORY),
		NEW_ERROR_TYPE(CL_PROFILING_INFO_NOT_AVAILABLE),
		NEW_ERROR_TYPE(CL_MEM_COPY_OVERLAP),
		NEW_ERROR_TYPE(CL_IMAGE_FORMAT_MISMATCH),
		NEW_ERROR_TYPE(CL_IMAGE_FORMAT_NOT_SUPPORTED),
		NEW_ERROR_TYPE(CL_BUILD_PROGRAM_FAILURE),
		NEW_ERROR_TYPE(CL_MAP_FAILURE),
		NEW_ERROR_TYPE(CL_MISALIGNED_SUB_BUFFER_OFFSET),
		NEW_ERROR_TYPE(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST),
		NEW_ERROR_TYPE(CL_COMPILE_PROGRAM_FAILURE),
		NEW_ERROR_TYPE(CL_LINKER_NOT_AVAILABLE),
		NEW_ERROR_TYPE(CL_LINK_PROGRAM_FAILURE),
		NEW_ERROR_TYPE(CL_DEVICE_PARTITION_FAILED),
		NEW_ERROR_TYPE(CL_KERNEL_ARG_INFO_NOT_AVAILABLE),
		NEW_ERROR_TYPE(CL_INVALID_VALUE),
		NEW_ERROR_TYPE(CL_INVALID_DEVICE_TYPE),
		NEW_ERROR_TYPE(CL_INVALID_PLATFORM),
		NEW_ERROR_TYPE(CL_INVALID_DEVICE),
		NEW_ERROR_TYPE(CL_INVALID_CONTEXT),
		NEW_ERROR_TYPE(CL_INVALID_QUEUE_PROPERTIES),
		NEW_ERROR_TYPE(CL_INVALID_COMMAND_QUEUE),
		NEW_ERROR_TYPE(CL_INVALID_HOST_PTR),
		NEW_ERROR_TYPE(CL_INVALID_MEM_OBJECT),
		NEW_ERROR_TYPE(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR),
		NEW_ERROR_TYPE(CL_INVALID_IMAGE_SIZE),
		NEW_ERROR_TYPE(CL_INVALID_SAMPLER),
		NEW_ERROR_TYPE(CL_INVALID_BINARY),
		NEW_ERROR_TYPE(CL_INVALID_BUILD_OPTIONS),
		NEW_ERROR_TYPE(CL_INVALID_PROGRAM),
		NEW_ERROR_TYPE(CL_INVALID_PROGRAM_EXECUTABLE),
		NEW_ERROR_TYPE(CL_INVALID_KERNEL_NAME),
		NEW_ERROR_TYPE(CL_INVALID_KERNEL_DEFINITION),
		NEW_ERROR_TYPE(CL_INVALID_KERNEL),
		NEW_ERROR_TYPE(CL_INVALID_ARG_INDEX),
		NEW_ERROR_TYPE(CL_INVALID_ARG_VALUE),
		NEW_ERROR_TYPE(CL_INVALID_ARG_SIZE),
		NEW_ERROR_TYPE(CL_INVALID_KERNEL_ARGS),
		NEW_ERROR_TYPE(CL_INVALID_WORK_DIMENSION),
		NEW_ERROR_TYPE(CL_INVALID_WORK_GROUP_SIZE),
		NEW_ERROR_TYPE(CL_INVALID_WORK_ITEM_SIZE),
		NEW_ERROR_TYPE(CL_INVALID_GLOBAL_OFFSET),
		NEW_ERROR_TYPE(CL_INVALID_EVENT_WAIT_LIST),
		NEW_ERROR_TYPE(CL_INVALID_EVENT),
		NEW_ERROR_TYPE(CL_INVALID_OPERATION),
		NEW_ERROR_TYPE(CL_INVALID_GL_OBJECT),
		NEW_ERROR_TYPE(CL_INVALID_BUFFER_SIZE),
		NEW_ERROR_TYPE(CL_INVALID_MIP_LEVEL),
		NEW_ERROR_TYPE(CL_INVALID_GLOBAL_WORK_SIZE),
		NEW_ERROR_TYPE(CL_INVALID_PROPERTY),
		NEW_ERROR_TYPE(CL_INVALID_IMAGE_DESCRIPTOR),
		NEW_ERROR_TYPE(CL_INVALID_COMPILER_OPTIONS),
		NEW_ERROR_TYPE(CL_INVALID_LINKER_OPTIONS),
		NEW_ERROR_TYPE(CL_INVALID_DEVICE_PARTITION_COUNT),
		NEW_ERROR_TYPE(CL_INVALID_PIPE_SIZE),
		NEW_ERROR_TYPE(CL_INVALID_DEVICE_QUEUE),
		NEW_ERROR_TYPE(CL_INVALID_SPEC_ID),
		NEW_ERROR_TYPE(CL_MAX_SIZE_RESTRICTION_EXCEEDED),
};

const char *clResolveError(const cl_int error) {
	for (size_t i = 0; i < 63; ++i) {
		if (CL_ERROR_LUT[i].code == error) return CL_ERROR_LUT[i].name;
	}
	return "<Unknown>";
}

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
