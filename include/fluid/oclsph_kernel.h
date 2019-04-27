
#define CURVE_UINT3_TYPE uint3
#define CURVE_UINT3_CTOR(x, y, z) ((uint3)((x), (y), (z)))

#include "oclsph_type.h"
#include "oclsph_collision.h"
#include "curves.h"
#include "mc.h"


#define DEBUG
#undef DEBUG


#ifndef SPH_H
#error SPH_H is not set
#endif


const constant float VD = 0.49f;// Velocity dampening;
const constant float RHO = 6378.0f; // Reference density;
const constant float RHO_RECIP = 1.f / RHO;

const constant float EPSILON = 0.00000001f;
const constant float CFM_EPSILON = 600.0f; // CFM propagation;
const constant float CorrDeltaQ = 0.3f * SPH_H;

const constant float C = 0.00001f;
const constant float VORTICITY_EPSILON = 0.0005f;
const constant float CorrK = 0.0001f;
const constant float CorrN = 4.f;

const constant float H2 = SPH_H * 2;
const constant float HH = SPH_H * SPH_H;
const constant float HHH = SPH_H * SPH_H * SPH_H;

const constant uint3 NEIGHBOUR_OFFSETS[27] = {
		(uint3) (-1, -1, -1), (uint3) (+0, -1, -1), (uint3) (+1, -1, -1),
		(uint3) (-1, +0, -1), (uint3) (+0, +0, -1), (uint3) (+1, +0, -1),
		(uint3) (-1, +1, -1), (uint3) (+0, +1, -1), (uint3) (+1, +1, -1),
		(uint3) (-1, -1, +0), (uint3) (+0, -1, +0), (uint3) (+1, -1, +0),
		(uint3) (-1, +0, +0), (uint3) (+0, +0, +0), (uint3) (+1, +0, +0),
		(uint3) (-1, +1, +0), (uint3) (+0, +1, +0), (uint3) (+1, +1, +0),
		(uint3) (-1, -1, +1), (uint3) (+0, -1, +1), (uint3) (+1, -1, +1),
		(uint3) (-1, +0, +1), (uint3) (+0, +0, +1), (uint3) (+1, +0, +1),
		(uint3) (-1, +1, +1), (uint3) (+0, +1, +1), (uint3) (+1, +1, +1)
};


const constant float Poly6Factor = 315.f / (64.f * M_PI_F * (HHH * HHH * HHH));
const constant float SpikyKernelFactor = -(45.f / (M_PI_F * HHH * HHH));

inline float poly6Kernel(const float r) {
	return select(0.f, Poly6Factor * pown(HH - r * r, 3), r <= SPH_H);
//	return r <= SPH_H ? Poly6Factor * pown(HH - r * r, 3) : 0.f;
}


#define BETWEEN(a, x, b) ((x)-(a)*(x)-(b)) > 0.f

inline float3 spikyKernelGradient(const float3 x, const float3 y, const float r) {



//	min(max(r, EPSILON), H)


	return (r >= EPSILON && r <= SPH_H) ?
	       (x - y) * (SpikyKernelFactor * native_divide(pown(SPH_H - r, 2), r)) :
	       (float3) (0.f);
}

#define FOR_SINGLE_GRID(zIndex, b, atoms, gridTable, gridTableN, op) \
{ \
    const size_t __offset = (zIndex); \
    const size_t __start = (gridTable)[__offset]; \
    const size_t __end = ((__offset + 1) < (gridTableN)) ? (gridTable)[__offset + 1] : (__start); \
    for (size_t __ni = __start; __ni < __end; ++__ni)  { \
        const global ClSphAtom *b = &(atoms)[__ni]; \
        op; \
    } \
} \

#define SWP_(x, y, d) \
{ \
    const size_t __tmp = min((d)[(y)], (d)[(x)]); \
    (d)[(y)] = max((d)[(y)], (d)[(x)]); \
    (d)[(x)] = __tmp; \
}; \


inline void sortArray27(size_t d[27]) {
	// optimal sort network
	//@formatter:off
	SWP_(1, 2, d) SWP_(0, 2, d) SWP_(0, 1, d) SWP_(4, 5, d) SWP_(3, 5, d) SWP_(3, 4, d)
	SWP_(0, 3, d) SWP_(1, 4, d) SWP_(2, 5, d) SWP_(2, 4, d) SWP_(1, 3, d) SWP_(2, 3, d)
	SWP_(7, 8, d) SWP_(6, 8, d) SWP_(6, 7, d) SWP_(9, 10, d) SWP_(11, 12, d) SWP_(9, 11, d)
	SWP_(10, 12, d) SWP_(10, 11, d) SWP_(6, 10, d) SWP_(6, 9, d) SWP_(7, 11, d) SWP_(8, 12, d)
	SWP_(8, 11, d) SWP_(7, 9, d) SWP_(8, 10, d) SWP_(8, 9, d) SWP_(0, 7, d) SWP_(0, 6, d)
	SWP_(1, 8, d) SWP_(2, 9, d) SWP_(2, 8, d) SWP_(1, 6, d) SWP_(2, 7, d) SWP_(2, 6, d)
	SWP_(3, 10, d) SWP_(4, 11, d) SWP_(5, 12, d) SWP_(5, 11, d) SWP_(4, 10, d) SWP_(5, 10, d)
	SWP_(3, 7, d) SWP_(3, 6, d) SWP_(4, 8, d) SWP_(5, 9, d) SWP_(5, 8, d) SWP_(4, 6, d)
	SWP_(5, 7, d) SWP_(5, 6, d) SWP_(14, 15, d) SWP_(13, 15, d) SWP_(13, 14, d) SWP_(16, 17, d)
	SWP_(18, 19, d) SWP_(16, 18, d) SWP_(17, 19, d) SWP_(17, 18, d) SWP_(13, 17, d) SWP_(13, 16, d)
	SWP_(14, 18, d) SWP_(15, 19, d) SWP_(15, 18, d) SWP_(14, 16, d) SWP_(15, 17, d) SWP_(15, 16, d)
	SWP_(21, 22, d) SWP_(20, 22, d) SWP_(20, 21, d) SWP_(23, 24, d) SWP_(25, 26, d) SWP_(23, 25, d)
	SWP_(24, 26, d) SWP_(24, 25, d) SWP_(20, 24, d) SWP_(20, 23, d) SWP_(21, 25, d) SWP_(22, 26, d)
	SWP_(22, 25, d) SWP_(21, 23, d) SWP_(22, 24, d) SWP_(22, 23, d) SWP_(13, 20, d) SWP_(14, 21, d)
	SWP_(15, 22, d) SWP_(15, 21, d) SWP_(14, 20, d) SWP_(15, 20, d) SWP_(16, 23, d) SWP_(17, 24, d)
	SWP_(17, 23, d) SWP_(18, 25, d) SWP_(19, 26, d) SWP_(19, 25, d) SWP_(18, 23, d) SWP_(19, 24, d)
	SWP_(19, 23, d) SWP_(16, 20, d) SWP_(17, 21, d) SWP_(17, 20, d) SWP_(18, 22, d) SWP_(19, 22, d)
	SWP_(18, 20, d) SWP_(19, 21, d) SWP_(19, 20, d) SWP_(0, 14, d) SWP_(0, 13, d) SWP_(1, 15, d)
	SWP_(2, 16, d) SWP_(2, 15, d) SWP_(1, 13, d) SWP_(2, 14, d) SWP_(2, 13, d) SWP_(3, 17, d)
	SWP_(4, 18, d) SWP_(5, 19, d) SWP_(5, 18, d) SWP_(4, 17, d) SWP_(5, 17, d) SWP_(3, 14, d)
	SWP_(3, 13, d) SWP_(4, 15, d) SWP_(5, 16, d) SWP_(5, 15, d) SWP_(4, 13, d) SWP_(5, 14, d)
	SWP_(5, 13, d) SWP_(6, 20, d) SWP_(7, 21, d) SWP_(8, 22, d) SWP_(8, 21, d) SWP_(7, 20, d)
	SWP_(8, 20, d) SWP_(9, 23, d) SWP_(10, 24, d) SWP_(10, 23, d) SWP_(11, 25, d) SWP_(12, 26, d)
	SWP_(12, 25, d) SWP_(11, 23, d) SWP_(12, 24, d) SWP_(12, 23, d) SWP_(9, 20, d) SWP_(10, 21, d)
	SWP_(10, 20, d) SWP_(11, 22, d) SWP_(12, 22, d) SWP_(11, 20, d) SWP_(12, 21, d) SWP_(12, 20, d)
	SWP_(6, 13, d) SWP_(7, 14, d) SWP_(8, 15, d) SWP_(8, 14, d) SWP_(7, 13, d) SWP_(8, 13, d)
	SWP_(9, 16, d) SWP_(10, 17, d) SWP_(10, 16, d) SWP_(11, 18, d) SWP_(12, 19, d) SWP_(12, 18, d)
	SWP_(11, 16, d) SWP_(12, 17, d) SWP_(12, 16, d) SWP_(9, 13, d) SWP_(10, 14, d) SWP_(10, 13, d)
	SWP_(11, 15, d) SWP_(12, 15, d) SWP_(11, 13, d) SWP_(12, 14, d) SWP_(12, 13, d)
	//@formatter:on
}

inline size_t zCurveGridIndexAtCoordU3(const uint3 v) {
	return zCurveGridIndexAtCoord(v.x, v.y, v.z);
}

#define SORTED

#ifdef SORTED

// FIXME OOB?
#define FOR_EACH_NEIGHBOUR__BAD(zIndex, gridTable, gridTableN, op) \
{ \
    const uint3 __delta = (uint3)(coordAtZCurveGridIndex0((zIndex)) , coordAtZCurveGridIndex1((zIndex)), coordAtZCurveGridIndex2((zIndex))); \
    for(size_t __i = 0; __i < 27; __i++) { \
        const size_t __offset = zCurveGridIndexAtCoordU3(NEIGHBOUR_OFFSETS[__i] + __delta); \
        const size_t __start = (gridTable)[__offset]; \
        const size_t __end = ((__offset + 1) < (gridTableN)) ? (gridTable)[__offset + 1] : (__start); \
        for (size_t b = __start; b < __end; ++b)  { \
                op; \
        } \
    } \
} \


#define FOR_EACH_NEIGHBOUR__(zIndex, gridTable, gridTableN, op) \
{ \
    const size_t __x = coordAtZCurveGridIndex0((zIndex)); \
    const size_t __y = coordAtZCurveGridIndex1((zIndex)); \
    const size_t __z = coordAtZCurveGridIndex2((zIndex)); \
    size_t __offsets[27] = { \
        zCurveGridIndexAtCoord(__x - 1, __y - 1, __z - 1), \
        zCurveGridIndexAtCoord(__x + 0, __y - 1, __z - 1), \
        zCurveGridIndexAtCoord(__x + 1, __y - 1, __z - 1), \
        zCurveGridIndexAtCoord(__x - 1, __y + 0, __z - 1), \
        zCurveGridIndexAtCoord(__x + 0, __y + 0, __z - 1), \
        zCurveGridIndexAtCoord(__x + 1, __y + 0, __z - 1), \
        zCurveGridIndexAtCoord(__x - 1, __y + 1, __z - 1), \
        zCurveGridIndexAtCoord(__x + 0, __y + 1, __z - 1), \
        zCurveGridIndexAtCoord(__x + 1, __y + 1, __z - 1), \
        zCurveGridIndexAtCoord(__x - 1, __y - 1, __z + 0), \
        zCurveGridIndexAtCoord(__x + 0, __y - 1, __z + 0), \
        zCurveGridIndexAtCoord(__x + 1, __y - 1, __z + 0), \
        zCurveGridIndexAtCoord(__x - 1, __y + 0, __z + 0), \
        zCurveGridIndexAtCoord(__x + 0, __y + 0, __z + 0), \
        zCurveGridIndexAtCoord(__x + 1, __y + 0, __z + 0), \
        zCurveGridIndexAtCoord(__x - 1, __y + 1, __z + 0), \
        zCurveGridIndexAtCoord(__x + 0, __y + 1, __z + 0), \
        zCurveGridIndexAtCoord(__x + 1, __y + 1, __z + 0), \
        zCurveGridIndexAtCoord(__x - 1, __y - 1, __z + 1), \
        zCurveGridIndexAtCoord(__x + 0, __y - 1, __z + 1), \
        zCurveGridIndexAtCoord(__x + 1, __y - 1, __z + 1), \
        zCurveGridIndexAtCoord(__x - 1, __y + 0, __z + 1), \
        zCurveGridIndexAtCoord(__x + 0, __y + 0, __z + 1), \
        zCurveGridIndexAtCoord(__x + 1, __y + 0, __z + 1), \
        zCurveGridIndexAtCoord(__x - 1, __y + 1, __z + 1), \
        zCurveGridIndexAtCoord(__x + 0, __y + 1, __z + 1), \
        zCurveGridIndexAtCoord(__x + 1, __y + 1, __z + 1) \
    }; \
    for(size_t __i = 0; __i < 27; __i++) { \
        const size_t __offset = (__offsets[__i]); \
        const size_t __start = (gridTable)[__offset]; \
        const size_t __end = ((__offset + 1) < (gridTableN)) ? (gridTable)[__offset + 1] : (__start); \
        for (size_t b = __start; b < __end; ++b)  { \
                op; \
        } \
    } \
} \

#define FOR_EACH_NEIGHBOUR(zIndex, b, atoms, atomN, gridTable, gridTableN, op) \
{ \
    const size_t __x = coordAtZCurveGridIndex0((zIndex)); \
    const size_t __y = coordAtZCurveGridIndex1((zIndex)); \
    const size_t __z = coordAtZCurveGridIndex2((zIndex)); \
    size_t __offsets[27] = { \
        zCurveGridIndexAtCoord(__x - 1, __y - 1, __z - 1), \
        zCurveGridIndexAtCoord(__x + 0, __y - 1, __z - 1), \
        zCurveGridIndexAtCoord(__x + 1, __y - 1, __z - 1), \
        zCurveGridIndexAtCoord(__x - 1, __y + 0, __z - 1), \
        zCurveGridIndexAtCoord(__x + 0, __y + 0, __z - 1), \
        zCurveGridIndexAtCoord(__x + 1, __y + 0, __z - 1), \
        zCurveGridIndexAtCoord(__x - 1, __y + 1, __z - 1), \
        zCurveGridIndexAtCoord(__x + 0, __y + 1, __z - 1), \
        zCurveGridIndexAtCoord(__x + 1, __y + 1, __z - 1), \
        zCurveGridIndexAtCoord(__x - 1, __y - 1, __z + 0), \
        zCurveGridIndexAtCoord(__x + 0, __y - 1, __z + 0), \
        zCurveGridIndexAtCoord(__x + 1, __y - 1, __z + 0), \
        zCurveGridIndexAtCoord(__x - 1, __y + 0, __z + 0), \
        zCurveGridIndexAtCoord(__x + 0, __y + 0, __z + 0), \
        zCurveGridIndexAtCoord(__x + 1, __y + 0, __z + 0), \
        zCurveGridIndexAtCoord(__x - 1, __y + 1, __z + 0), \
        zCurveGridIndexAtCoord(__x + 0, __y + 1, __z + 0), \
        zCurveGridIndexAtCoord(__x + 1, __y + 1, __z + 0), \
        zCurveGridIndexAtCoord(__x - 1, __y - 1, __z + 1), \
        zCurveGridIndexAtCoord(__x + 0, __y - 1, __z + 1), \
        zCurveGridIndexAtCoord(__x + 1, __y - 1, __z + 1), \
        zCurveGridIndexAtCoord(__x - 1, __y + 0, __z + 1), \
        zCurveGridIndexAtCoord(__x + 0, __y + 0, __z + 1), \
        zCurveGridIndexAtCoord(__x + 1, __y + 0, __z + 1), \
        zCurveGridIndexAtCoord(__x - 1, __y + 1, __z + 1), \
        zCurveGridIndexAtCoord(__x + 0, __y + 1, __z + 1), \
        zCurveGridIndexAtCoord(__x + 1, __y + 1, __z + 1) \
    }; \
    sortArray27(__offsets);  \
    for(size_t __i = 0; __i < 27; __i++) \
        FOR_SINGLE_GRID(__offsets[__i], b, atoms, gridTable, gridTableN, op);  \
} \

#else

#define FOR_EACH_NEIGHBOUR(zIndex, b, atoms, atomN, gridTable, gridTableN, op) \
{ \
	const uint __x = coordAtZCurveGridIndex0((zIndex)); \
	const uint __y = coordAtZCurveGridIndex1((zIndex)); \
	const uint __z = coordAtZCurveGridIndex2((zIndex)); \
	FOR_SINGLE_GRID(zCurveGridIndexAtCoord(__x - 1, __y - 1, __z - 1), b, atoms, gridTable, gridTableN, op); \
	FOR_SINGLE_GRID(zCurveGridIndexAtCoord(__x + 0, __y - 1, __z - 1), b, atoms, gridTable, gridTableN, op); \
	FOR_SINGLE_GRID(zCurveGridIndexAtCoord(__x + 1, __y - 1, __z - 1), b, atoms, gridTable, gridTableN, op); \
	FOR_SINGLE_GRID(zCurveGridIndexAtCoord(__x - 1, __y + 0, __z - 1), b, atoms, gridTable, gridTableN, op); \
	FOR_SINGLE_GRID(zCurveGridIndexAtCoord(__x + 0, __y + 0, __z - 1), b, atoms, gridTable, gridTableN, op); \
	FOR_SINGLE_GRID(zCurveGridIndexAtCoord(__x + 1, __y + 0, __z - 1), b, atoms, gridTable, gridTableN, op); \
	FOR_SINGLE_GRID(zCurveGridIndexAtCoord(__x - 1, __y + 1, __z - 1), b, atoms, gridTable, gridTableN, op); \
	FOR_SINGLE_GRID(zCurveGridIndexAtCoord(__x + 0, __y + 1, __z - 1), b, atoms, gridTable, gridTableN, op); \
	FOR_SINGLE_GRID(zCurveGridIndexAtCoord(__x + 1, __y + 1, __z - 1), b, atoms, gridTable, gridTableN, op); \
	FOR_SINGLE_GRID(zCurveGridIndexAtCoord(__x - 1, __y - 1, __z + 0), b, atoms, gridTable, gridTableN, op); \
	FOR_SINGLE_GRID(zCurveGridIndexAtCoord(__x + 0, __y - 1, __z + 0), b, atoms, gridTable, gridTableN, op); \
	FOR_SINGLE_GRID(zCurveGridIndexAtCoord(__x + 1, __y - 1, __z + 0), b, atoms, gridTable, gridTableN, op); \
	FOR_SINGLE_GRID(zCurveGridIndexAtCoord(__x - 1, __y + 0, __z + 0), b, atoms, gridTable, gridTableN, op); \
	FOR_SINGLE_GRID(zCurveGridIndexAtCoord(__x + 0, __y + 0, __z + 0), b, atoms, gridTable, gridTableN, op); \
	FOR_SINGLE_GRID(zCurveGridIndexAtCoord(__x + 1, __y + 0, __z + 0), b, atoms, gridTable, gridTableN, op); \
	FOR_SINGLE_GRID(zCurveGridIndexAtCoord(__x - 1, __y + 1, __z + 0), b, atoms, gridTable, gridTableN, op); \
	FOR_SINGLE_GRID(zCurveGridIndexAtCoord(__x + 0, __y + 1, __z + 0), b, atoms, gridTable, gridTableN, op); \
	FOR_SINGLE_GRID(zCurveGridIndexAtCoord(__x + 1, __y + 1, __z + 0), b, atoms, gridTable, gridTableN, op); \
	FOR_SINGLE_GRID(zCurveGridIndexAtCoord(__x - 1, __y - 1, __z + 1), b, atoms, gridTable, gridTableN, op); \
	FOR_SINGLE_GRID(zCurveGridIndexAtCoord(__x + 0, __y - 1, __z + 1), b, atoms, gridTable, gridTableN, op); \
	FOR_SINGLE_GRID(zCurveGridIndexAtCoord(__x + 1, __y - 1, __z + 1), b, atoms, gridTable, gridTableN, op); \
	FOR_SINGLE_GRID(zCurveGridIndexAtCoord(__x - 1, __y + 0, __z + 1), b, atoms, gridTable, gridTableN, op); \
	FOR_SINGLE_GRID(zCurveGridIndexAtCoord(__x + 0, __y + 0, __z + 1), b, atoms, gridTable, gridTableN, op); \
	FOR_SINGLE_GRID(zCurveGridIndexAtCoord(__x + 1, __y + 0, __z + 1), b, atoms, gridTable, gridTableN, op); \
	FOR_SINGLE_GRID(zCurveGridIndexAtCoord(__x - 1, __y + 1, __z + 1), b, atoms, gridTable, gridTableN, op); \
	FOR_SINGLE_GRID(zCurveGridIndexAtCoord(__x + 0, __y + 1, __z + 1), b, atoms, gridTable, gridTableN, op); \
	FOR_SINGLE_GRID(zCurveGridIndexAtCoord(__x + 1, __y + 1, __z + 1), b, atoms, gridTable, gridTableN, op); \
} \

#endif

inline float fast_length_sq(float3 v) {
	return (v.x * v.x) + (v.y * v.y) + (v.z * v.z);
//	return mad(a.x, a.x, mad(a.y, a.y, a.z * a.z));
}

kernel void check_size(global size_t *sizes) {
	sizes[get_global_id(0)] = _SIZES[get_global_id(0)];
}


kernel void sph_lambda(
		const constant ClSphConfig *config,
		const global uint *zIndex, const global uint *gridTable, const uint gridTableN,
		const global float3 *pStar,
		const global float *mass,
		global float *lambda
) {

	const size_t a = get_global_id(0);

	float3 norm2V = (float3) (0.f);
	float rho = 0.f;

	FOR_EACH_NEIGHBOUR__(zIndex[a], gridTable, gridTableN, {
		const float r = fast_distance(pStar[a], pStar[b]);
		norm2V = mad(spikyKernelGradient(pStar[a], pStar[b], r), RHO_RECIP, norm2V);
		rho = mad(mass[a], poly6Kernel(r), rho);
	});

	float norm2 = fast_length_sq(norm2V); // dot self = length2
	float C1 = (rho / RHO - 1.f);
	lambda[a] = -C1 / (norm2 + CFM_EPSILON);
}


kernel void sph_delta(
		const constant ClSphConfig *config,
		const global uint *zIndex, const global uint *gridTable, const uint gridTableN,
		const global ClSphTraiangle *mesh, const uint meshN,
		global float3 *pStar,
		const global float *lambda,
		const global float3 *position,
		global float3 *velocity,
		global float3 *deltaP

) {
	const size_t a = get_global_id(0);

	const float p6DeltaQ = poly6Kernel(CorrDeltaQ);

	float3 deltaPacc = (float3) (0.f);

	FOR_EACH_NEIGHBOUR__(zIndex[a], gridTable, gridTableN, {
		const float r = fast_distance(pStar[a], pStar[b]);
		const float corr = -CorrK * pow(poly6Kernel(r) / p6DeltaQ, CorrN);
		const float factor = (lambda[a] + lambda[b] + corr) / RHO;
		deltaPacc = mad(spikyKernelGradient(pStar[a], pStar[b], r), factor, deltaPacc);
	});

	deltaP[a] = deltaPacc;

	ClSphResponse resp;
	resp.position = (pStar[a] + deltaP[a]) * config->scale;
	resp.velocity = velocity[a];

	collideTriangle2(mesh, meshN, position[a], &resp);

	// clamp to extent
	resp.position = min(config->maxBound, max(config->minBound, resp.position));


	pStar[a] = resp.position / config->scale;
	velocity[a] = resp.velocity;


#ifdef DEBUG
	printf("[%ld] config { scale=%f, dt=%f} p={id=%ld, mass=%f lam=%f, deltaP=(%f,%f,%f)}\n",
		   id, config.scale, config.dt,
		   a->particle.id, a->particle.mass, a->lambda,
		   a->deltaP.x, a->deltaP.y, a->deltaP.z);
#endif
}


//kernel void sph_initialise(
//		const constant ClSphConfig *config, float3 constForce,
//		global ClSphAtom *atoms,
//) {
//	const size_t id = get_global_id(0);
//	global ClSphAtom *a = &atoms[id];
//
//	a->velocity = config.constForce(p) * config.dt + a->velocity;
//	a->now = (a->velocity * config.dt) + (a->position / config.scale);
//	a->zIndex = zCurveGridIndexAtCoord()
//
//	const float3 deltaX = a->now - a->position / config.scale;
//	results[id].position = a->now * config.scale;
//	results[id].velocity = (deltaX * (1.f / config.dt) + a->velocity) * VD;
//}


kernel void sph_finalise(
		const constant ClSphConfig *config,
		const global float3 *pStar,
		global float3 *position,
		global float3 *velocity
) {
	const size_t a = get_global_id(0);
	const float3 deltaX = pStar[a] - position[a] / config->scale;
	position[a] = pStar[a] * config->scale;
	velocity[a] = mad(deltaX, (1.f / config->dt), velocity[a]) * VD;
}

// mcCube -> TrigSumN
//

kernel void sph_evalLattice(
		const constant ClSphConfig *config, const constant ClMcConfig *mcConfig,
		const global uint *gridTable, uint gridTableN,
		const float3 min, const uint3 sizes, const uint3 gridExtent,
		const global float3 *position,
		global float4 *lattice) {


	const size_t x = get_global_id(0);
	const size_t y = get_global_id(1);
	const size_t z = get_global_id(2);


	const float3 pos = (float3) (x, y, z);
	const float step = SPH_H / mcConfig->sampleResolution;
	const float3 a = (min + (pos * step)) * config->scale;

	const size_t zIndex = zCurveGridIndexAtCoord(
			(size_t) (pos.x / mcConfig->sampleResolution),
			(size_t) (pos.y / mcConfig->sampleResolution),
			(size_t) (pos.z / mcConfig->sampleResolution));

	const float threshold = SPH_H * config->scale * 1;


	const size_t __x = coordAtZCurveGridIndex0(zIndex);
	const size_t __y = coordAtZCurveGridIndex1(zIndex);
	const size_t __z = coordAtZCurveGridIndex2(zIndex);

	if (__x == gridExtent.x && __y == gridExtent.y && __z == gridExtent.z) {
		// XXX there is exactly one case where this may happens: the last element of the z-curve
		return;
	}

	const size_t x_l = clamp(((int) __x) - 1, 0, (int) gridExtent.x - 1);
	const size_t x_r = clamp(((int) __x) + 1, 0, (int) gridExtent.x - 1);
	const size_t y_l = clamp(((int) __y) - 1, 0, (int) gridExtent.y - 1);
	const size_t y_r = clamp(((int) __y) + 1, 0, (int) gridExtent.y - 1);
	const size_t z_l = clamp(((int) __z) - 1, 0, (int) gridExtent.z - 1);
	const size_t z_r = clamp(((int) __z) + 1, 0, (int) gridExtent.z - 1);

	size_t offsets[27] = {
			zCurveGridIndexAtCoord(x_l, y_l, z_l),
			zCurveGridIndexAtCoord(__x, y_l, z_l),
			zCurveGridIndexAtCoord(x_r, y_l, z_l),
			zCurveGridIndexAtCoord(x_l, __y, z_l),
			zCurveGridIndexAtCoord(__x, __y, z_l),
			zCurveGridIndexAtCoord(x_r, __y, z_l),
			zCurveGridIndexAtCoord(x_l, y_r, z_l),
			zCurveGridIndexAtCoord(__x, y_r, z_l),
			zCurveGridIndexAtCoord(x_r, y_r, z_l),
			zCurveGridIndexAtCoord(x_l, y_l, __z),
			zCurveGridIndexAtCoord(__x, y_l, __z),
			zCurveGridIndexAtCoord(x_r, y_l, __z),
			zCurveGridIndexAtCoord(x_l, __y, __z),
			zCurveGridIndexAtCoord(__x, __y, __z),
			zCurveGridIndexAtCoord(x_r, __y, __z),
			zCurveGridIndexAtCoord(x_l, y_r, __z),
			zCurveGridIndexAtCoord(__x, y_r, __z),
			zCurveGridIndexAtCoord(x_r, y_r, __z),
			zCurveGridIndexAtCoord(x_l, y_l, z_r),
			zCurveGridIndexAtCoord(__x, y_l, z_r),
			zCurveGridIndexAtCoord(x_r, y_l, z_r),
			zCurveGridIndexAtCoord(x_l, __y, z_r),
			zCurveGridIndexAtCoord(__x, __y, z_r),
			zCurveGridIndexAtCoord(x_r, __y, z_r),
			zCurveGridIndexAtCoord(x_l, y_r, z_r),
			zCurveGridIndexAtCoord(__x, y_r, z_r),
			zCurveGridIndexAtCoord(x_r, y_r, z_r)
	};

//	sortArray27(offsets);

	float v = 0.f;
	float3 normal = (float3) (0);
	for (size_t i = 0; i < 27; i++) {
		const size_t offset = offsets[i];
		const size_t start = gridTable[offset];
		const size_t end = (offset + 1 < gridTableN) ? gridTable[offset + 1] : start;
		for (size_t b = start; b < end; ++b) {
			if (fast_distance(position[b], a) < threshold) {
				const float3 l = (position[b]) - a;
				const float len = fast_length(l);
				const float denominator = pow(len, mcConfig->particleInfluence);

				normal += (-mcConfig->particleInfluence) *
				          mcConfig->particleSize *
				          (l / denominator);
				v += (mcConfig->particleSize / denominator);
			}
		}
	}
	normal = fast_normalize(normal);
	global float4 *out = &lattice[index3d(x, y, z, sizes.x, sizes.y, sizes.z)];
	out->s0 = v;
	out->s1 = normal.s0;
	out->s2 = normal.s1;
	out->s3 = normal.s2;
}


const constant uint3 CUBE_OFFSETS[8] = {
		(uint3) (0, 0, 0),
		(uint3) (1, 0, 0),
		(uint3) (1, 1, 0),
		(uint3) (0, 1, 0),
		(uint3) (0, 0, 1),
		(uint3) (1, 0, 1),
		(uint3) (1, 1, 1),
		(uint3) (0, 1, 1)
};


kernel void mc_size(
		const constant ClMcConfig *mcConfig,
		const uint3 sizes,
		const global float4 *values,
		local uint *localSums,
		global uint *partialSums
) {


	const uint3 marchRange = sizes - (uint3) (1);

	uint nVert = 0;
	// because global size needs to be divisible by local group size (CL1.2), we discard the padding
	if (get_global_id(0) >= (marchRange.x * marchRange.y * marchRange.z)) {
		// NOOP
	}else{
		const uint3 pos = to3d(get_global_id(0), marchRange.x, marchRange.y, marchRange.z);
		const float isolevel = mcConfig->isolevel;
		uint ci = 0u;

		for (int i = 0; i < 8; ++i) {
			const uint3 offset = CUBE_OFFSETS[i] + pos;
			const float v = values[index3d(
					offset.x, offset.y, offset.z, sizes.x, sizes.y, sizes.z)].s0;
			ci = select(ci, ci | (1 << i), v < isolevel);
		}
		nVert = select((uint) NumVertsTable[ci] / 3, 0u, EdgeTable[ci] == 0);
	}

	const uint localId = get_local_id(0);
	const uint groupSize = get_local_size(0);

	// zero out local memory first, this is needed because workgroup size might not divide
	// group size perfectly; we need to zero out trailing cells
	if (localId == 0) {
		for (size_t i = 0; i < get_local_size(0); ++i) localSums[i] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	localSums[localId] = nVert;

	for (uint stride = groupSize / 2; stride > 0; stride >>= 1u) {
		barrier(CLK_LOCAL_MEM_FENCE);
		if (localId < stride) localSums[localId] += localSums[localId + stride];
	}


	if (localId == 0) {
		partialSums[get_group_id(0)] = localSums[0];
	}

}

inline float3 lerp(const float isolevel,
                   const float3 p1, const float3 p2,
                   const float v1, const float v2) {
	return mix(p1, p2, ((isolevel - v1) / (v2 - v1)));
}

// FIXME nvidia GPUs output broken triangles for some reason, Intel and AMD works fine
kernel void mc_eval(
		const constant ClSphConfig *config, const constant ClMcConfig *mcConfig,
		const float3 min, const uint3 sizes,
		const global float4 *values,
		volatile global uint *trigCounter,

		const uint acc,

		global float3 *outVxs,
		global float3 *outVys,
		global float3 *outVzs,

		global float3 *outNxs,
		global float3 *outNys,
		global float3 *outNzs
) {

	const uint3 pos = to3d(get_global_id(0), sizes.x - 1, sizes.y - 1, sizes.z - 1);
	const float isolevel = mcConfig->isolevel;
	const float step = SPH_H / mcConfig->sampleResolution;

	float vertices[8];
	float3 normals[8];
	float3 offsets[8];

	uint ci = 0;
	for (int i = 0; i < 8; ++i) {
		const uint3 offset = CUBE_OFFSETS[i] + pos;
		const float4 point = values[index3d(
				offset.x, offset.y, offset.z, sizes.x, sizes.y, sizes.z)];

		vertices[i] = point.s0;
		normals[i] = (float3) (point.s1, point.s2, point.s3);
		offsets[i] = (min + (convert_float3(offset) * step)) * config->scale;

		ci = select(ci, ci | (1 << i), vertices[i] < isolevel);
	}

	float3 ts[12];
	float3 ns[12];

	const uint edge = EdgeTable[ci];

	if (edge & 1 << 0) {
		ts[0] = lerp(isolevel, offsets[0], offsets[1], vertices[0], vertices[1]);
		ns[0] = lerp(isolevel, normals[0], normals[1], vertices[0], vertices[1]);
	}
	if (edge & 1 << 1) {
		ts[1] = lerp(isolevel, offsets[1], offsets[2], vertices[1], vertices[2]);
		ns[1] = lerp(isolevel, normals[1], normals[2], vertices[1], vertices[2]);
	}
	if (edge & 1 << 2) {
		ts[2] = lerp(isolevel, offsets[2], offsets[3], vertices[2], vertices[3]);
		ns[2] = lerp(isolevel, normals[2], normals[3], vertices[2], vertices[3]);
	}
	if (edge & 1 << 3) {
		ts[3] = lerp(isolevel, offsets[3], offsets[0], vertices[3], vertices[0]);
		ns[3] = lerp(isolevel, normals[3], normals[0], vertices[3], vertices[0]);
	}
	if (edge & 1 << 4) {
		ts[4] = lerp(isolevel, offsets[4], offsets[5], vertices[4], vertices[5]);
		ns[4] = lerp(isolevel, normals[4], normals[5], vertices[4], vertices[5]);
	}
	if (edge & 1 << 5) {
		ts[5] = lerp(isolevel, offsets[5], offsets[6], vertices[5], vertices[6]);
		ns[5] = lerp(isolevel, normals[5], normals[6], vertices[5], vertices[6]);
	}
	if (edge & 1 << 6) {
		ts[6] = lerp(isolevel, offsets[6], offsets[7], vertices[6], vertices[7]);
		ns[6] = lerp(isolevel, normals[6], normals[7], vertices[6], vertices[7]);
	}
	if (edge & 1 << 7) {
		ts[7] = lerp(isolevel, offsets[7], offsets[4], vertices[7], vertices[4]);
		ns[7] = lerp(isolevel, normals[7], normals[4], vertices[7], vertices[4]);
	}
	if (edge & 1 << 8) {
		ts[8] = lerp(isolevel, offsets[0], offsets[4], vertices[0], vertices[4]);
		ns[8] = lerp(isolevel, normals[0], normals[4], vertices[0], vertices[4]);
	}
	if (edge & 1 << 9) {
		ts[9] = lerp(isolevel, offsets[1], offsets[5], vertices[1], vertices[5]);
		ns[9] = lerp(isolevel, normals[1], normals[5], vertices[1], vertices[5]);
	}
	if (edge & 1 << 10) {
		ts[10] = lerp(isolevel, offsets[2], offsets[6], vertices[2], vertices[6]);
		ns[10] = lerp(isolevel, normals[2], normals[6], vertices[2], vertices[6]);
	}
	if (edge & 1 << 11) {
		ts[11] = lerp(isolevel, offsets[3], offsets[7], vertices[3], vertices[7]);
		ns[11] = lerp(isolevel, normals[3], normals[7], vertices[3], vertices[7]);
	}


	for (size_t i = 0; TriTable[ci][i] != 255; i += 3) {
		const uint trigIndex = atomic_inc(trigCounter);
		const int x = TriTable[ci][i + 0];
		const int y = TriTable[ci][i + 1];
		const int z = TriTable[ci][i + 2];
		outVxs[trigIndex] = ts[x];
		outVys[trigIndex] = ts[y];
		outVzs[trigIndex] = ts[z];
		outNxs[trigIndex] = ns[x];
		outNys[trigIndex] = ns[y];
		outNzs[trigIndex] = ns[z];
	}
//	printf("trigIdx: %d -> %d", (*trigCounter), trigIndex);




}



