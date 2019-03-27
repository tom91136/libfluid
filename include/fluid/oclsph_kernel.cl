#include "oclsph_type.h"
#include "oclsph_collision.h"
#include "zcurve.h"


#define DEBUG
#undef DEBUG


#ifndef H
#error H is not set
#endif


const constant float VD = 0.49f;// Velocity dampening;
const constant float RHO = 6378.0f; // Reference density;
const constant float EPSILON = 0.00000001f;
const constant float CFM_EPSILON = 600.0f; // CFM propagation;

const constant float C = 0.00001f;
const constant float VORTICITY_EPSILON = 0.0005f;
const constant float CorrK = 0.0001f;
const constant float CorrN = 4.f;

const constant float H2 = H * 2;
const constant float HH = H * H;
const constant float HHH = H * H * H;

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
	return r <= H ? Poly6Factor * pown(HH - r * r, 3) : 0.f;
}

inline float3 spikyKernelGradient(const float3 x, const float3 y, const float r) {
	return (r >= EPSILON && r <= H) ?
	       (x - y) * (SpikyKernelFactor * (pown(H - r, 2) / r)) :
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

#define SORTED

#ifdef SORTED

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

kernel void sph_lambda(
		const ClSphConfig config,
		global ClSphAtom *atoms, uint atomN,
		const global uint *gridTable, uint gridTableN
) {
	const size_t id = get_global_id(0);
	global ClSphAtom *a = &atoms[id];
	float3 norm2V = (float3) (0.f);
	float rho = 0.f;
	const float oneOverRho = 1.f / RHO;
	FOR_EACH_NEIGHBOUR(a->zIndex, b, atoms, atomN, gridTable, gridTableN, {
		const float r = fast_distance(a->pStar, b->pStar);
		norm2V = mad(spikyKernelGradient(a->pStar, b->pStar, r), oneOverRho, norm2V);
		rho = mad(b->particle.mass, poly6Kernel(r), rho);
	});


	float norm2 = fast_length_sq(norm2V); // dot self = length2
	float C1 = (rho / RHO - 1.f);
	a->lambda = -C1 / (norm2 + CFM_EPSILON);
}


kernel void sph_delta(
		const ClSphConfig config,
		global ClSphAtom *atoms, const uint atomN,
		const global uint *gridTable, const uint gridTableN,
		const global ClSphTraiangle *mesh, const uint meshN
) {
	const size_t id = get_global_id(0);

	global ClSphAtom *a = &atoms[id];

	const float CorrDeltaQ = 0.3f * H;
	const float p6DeltaQ = poly6Kernel(CorrDeltaQ);

	float3 deltaP = (float3) (0.f);


	FOR_EACH_NEIGHBOUR(a->zIndex, b, atoms, atomN, gridTable, gridTableN, {
		const float r = fast_distance(a->pStar, b->pStar);
		const float corr = -CorrK * pow(poly6Kernel(r) / p6DeltaQ, CorrN);
		const float factor = (a->lambda + b->lambda + corr) / RHO;
		deltaP = mad(spikyKernelGradient(a->pStar, b->pStar, r), factor, deltaP);
	});
	a->deltaP = deltaP;

	// collision


	ClSphResponse resp;
	resp.position = (a->pStar + a->deltaP) * config.scale;
	resp.velocity = a->particle.velocity;

//	collideTriangle2(mesh, meshN, a->particle.position, &resp);

	// clamp to extent
	resp.position = min(config.maxBound, max(config.minBound, resp.position));


	a->pStar = resp.position / config.scale;
	a->particle.velocity = resp.velocity;


#ifdef DEBUG
	printf("[%ld] config { scale=%f, dt=%f} p={id=%ld, mass=%f lam=%f, deltaP=(%f,%f,%f)}\n",
		   id, config.scale, config.dt,
		   a->particle.id, a->particle.mass, a->lambda,
		   a->deltaP.x, a->deltaP.y, a->deltaP.z);
#endif
}


//kernel void sph_initialise(
//		const ClSphConfig config, float3 constForce,
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
		const ClSphConfig config,
		global ClSphAtom *atoms,
		global ClSphParticle *results
) {
	const size_t id = get_global_id(0);
	global ClSphAtom *a = &atoms[id];
	const float3 deltaX = a->pStar - a->particle.position / config.scale;
	a->particle.position = a->pStar * config.scale;
	a->particle.velocity = mad(deltaX, (1.f / config.dt), a->particle.velocity) * VD;
	results[id] = a->particle;
}

kernel void sph_create_field(
		const ClSphConfig config,
		const global ClSphAtom *atoms, uint atomN,
		const global uint *gridTable, uint gridTableN,
		const float3 min, const ClMcConfig mcConfig,
		global float *field, const uint3 sizes, const uint3 gridExtent) {


	const size_t x = get_global_id(0);
	const size_t y = get_global_id(1);
	const size_t z = get_global_id(2);


	const float3 pos = (float3) (x, y, z);
	const float step = H / mcConfig.sampleResolution;
	const float3 a = (min + (pos * step)) * config.scale;

	const size_t zIndex = zCurveGridIndexAtCoord(
			(size_t) (pos.x / mcConfig.sampleResolution),
			(size_t) (pos.y / mcConfig.sampleResolution),
			(size_t) (pos.z / mcConfig.sampleResolution));

	const float sN = pown(mcConfig.particleSize, 2);
	const float threshold = H * config.scale * 1;
	float v = 0.f;


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

	for (size_t i = 0; i < 27; i++) {
		FOR_SINGLE_GRID(offsets[i], b, atoms, gridTable, gridTableN, {
			if (fast_distance(b->particle.position, a) < threshold) {
				const float3 l = (b->particle.position) - a;
				const float len = fast_length_sq(l);
				v += (sN / pow(len, mcConfig.particleInfluence));
			}
		});
	}

	field[index3d(x, y, z, sizes.x, sizes.y, sizes.z)] = v;
}
