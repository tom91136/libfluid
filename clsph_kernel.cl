


#ifndef __OPENCL_C_VERSION__

#include <tgmath.h>
#include "clsph_types.h"
#include "zcurve.h"

#define global
#define kernel
#define constant
#define M_PI_F 3.1415926f
#define INCL
#else

#include "clsph_types.h"
#include "zcurve.h"

#define global global
#define kernel kernel
#define constant constant
#endif

#define DEBUG
#undef DEBUG


const constant float VD = 0.49f;// Velocity dampening;
const constant float RHO = 6378.0f; // Reference density;
const constant float EPSILON = 0.00000001f;
const constant float CFM_EPSILON = 600.0f; // CFM propagation;

const constant float C = 0.00001f;
const constant float VORTICITY_EPSILON = 0.0005f;
const constant float CorrK = 0.0001f;
const constant float CorrN = 4.f;


const constant float H = 0.1f;
const constant float H2 = H * 2;
const constant float HH = H * H;
const constant float HHH = H * H * H;
#define  NEIGHBOUR_SIZE  (1 + (2) * 2) // n[L] + C + n[R]
const constant float NEIGHBOURS[NEIGHBOUR_SIZE] = {-H2, -H, 0, H, H2};


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
	return r <= H ? Poly6Factor * pow(HH - r * r, 3.f) : 0.f;
}

inline float3 spikyKernelGradient(const float3 x, const float3 y, const float r) {
	return (r >= EPSILON && r <= H) ?
	       (x - y) * (SpikyKernelFactor * (pow(H - r, 2.f) / r)) :
	       (float3) (0.f);
}

#define FOR_EACH_NEIGHBOUR_BEGIN(a, b, atoms, atomN, gridTable, gridTableN) \
{ \
    uint3 __coord = (uint3) ( \
            coordAtZCurveGridIndex0(a->zIndex), \
            coordAtZCurveGridIndex1(a->zIndex), \
            coordAtZCurveGridIndex2(a->zIndex)); \
    for (size_t __i = 0; __i < 27; ++__i) { \
        const uint3 __delta = __coord + NEIGHBOUR_OFFSETS[__i]; \
        const size_t __offset = zCurveGridIndexAtCoord(__delta.x, __delta.y, __delta.z); \
        const size_t __start = (gridTable)[__offset]; \
        const size_t __end = ((__offset + 1) < (gridTableN)) ? (gridTable)[__offset + 1] : (atomN); \
        for (size_t __ni = __start; __ni < __end; ++__ni)  { \
            const global ClSphAtom *b = &(atoms)[__ni]; \

#define FOR_EACH_NEIGHBOUR_END  \
        } \
    } \
} \

kernel void sph_lambda(
		const ClSphConfig config,
		global ClSphAtom *atoms, uint atomN,
		const global uint *gridTable, uint gridTableN
) {
	const size_t id = get_global_id(0);
	global ClSphAtom *a = &atoms[id];
	float3 norm2V = (float3) (0.f);
	float rho = 0.f;

	FOR_EACH_NEIGHBOUR_BEGIN(a, b, atoms, atomN, gridTable, gridTableN)
				const float r = distance(a->pStar, b->pStar);
				norm2V += spikyKernelGradient(a->pStar, b->pStar, r) * (1.f / RHO);
				rho += b->particle.mass * poly6Kernel(r);
	FOR_EACH_NEIGHBOUR_END

	float norm2 = dot(norm2V, norm2V); // dot self = length2
	float C1 = (rho / RHO - 1.f);
	a->lambda = -C1 / (norm2 + CFM_EPSILON);
}


kernel void sph_delta(
		const ClSphConfig config,
		global ClSphAtom *atoms, uint atomN,
		const global uint *gridTable, uint gridTableN
) {
	const size_t id = get_global_id(0);

	global ClSphAtom *a = &atoms[id];

	const float CorrDeltaQ = 0.3f * H;
	const float p6DeltaQ = poly6Kernel(CorrDeltaQ);

	float3 deltaP = (float3) (0.f);


	FOR_EACH_NEIGHBOUR_BEGIN(a, b, atoms, atomN, gridTable, gridTableN)
				const float r = distance(a->pStar, b->pStar);
				const float corr = -CorrK * pow(poly6Kernel(r) / p6DeltaQ, CorrN);
				const float factor = (a->lambda + b->lambda + corr) / RHO;
				deltaP += spikyKernelGradient(a->pStar, b->pStar, r) * factor;
	FOR_EACH_NEIGHBOUR_END

	a->deltaP = deltaP;

	// collision
	float3 currentP = (a->pStar + a->deltaP) * config.scale;
	float3 currentV = a->particle.velocity;
	currentP = clamp(currentP, -500.f, 500.f);
	// TODO handle colliders

	a->pStar = currentP / config.scale;
	a->particle.velocity = currentV;

//	size_t x = 0;
//	for (size_t i = neighbourStart(a); i < neighbourEnd(a); i++) {
//		x += atoms[neighbours[i]].id;
//	}



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
	a->particle.velocity = (deltaX * (1.f / config.dt) + a->particle.velocity) * VD;
	results[id] = a->particle;
}
