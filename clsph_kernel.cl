#include "clsph_type.h"
#include "zcurve.h"


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

#define SORTED

#ifdef SORTED

const static void sortArray27(size_t d[27]) {
	// optimal sort network
#define SWAP(x, y) { const size_t tmp = min(d[y], d[x]); d[y] = max(d[y], d[x]); d[x] = tmp; };
	//@formatter:off
	SWAP(1, 2) SWAP(0, 2) SWAP(0, 1) SWAP(4, 5) SWAP(3, 5) SWAP(3, 4)
	SWAP(0, 3) SWAP(1, 4) SWAP(2, 5) SWAP(2, 4) SWAP(1, 3) SWAP(2, 3)
	SWAP(7, 8) SWAP(6, 8) SWAP(6, 7) SWAP(9, 10) SWAP(11, 12) SWAP(9, 11)
	SWAP(10, 12) SWAP(10, 11) SWAP(6, 10) SWAP(6, 9) SWAP(7, 11) SWAP(8, 12)
	SWAP(8, 11) SWAP(7, 9) SWAP(8, 10) SWAP(8, 9) SWAP(0, 7) SWAP(0, 6)
	SWAP(1, 8) SWAP(2, 9) SWAP(2, 8) SWAP(1, 6) SWAP(2, 7) SWAP(2, 6)
	SWAP(3, 10) SWAP(4, 11) SWAP(5, 12) SWAP(5, 11) SWAP(4, 10) SWAP(5, 10)
	SWAP(3, 7) SWAP(3, 6) SWAP(4, 8) SWAP(5, 9) SWAP(5, 8) SWAP(4, 6)
	SWAP(5, 7) SWAP(5, 6) SWAP(14, 15) SWAP(13, 15) SWAP(13, 14) SWAP(16, 17)
	SWAP(18, 19) SWAP(16, 18) SWAP(17, 19) SWAP(17, 18) SWAP(13, 17) SWAP(13, 16)
	SWAP(14, 18) SWAP(15, 19) SWAP(15, 18) SWAP(14, 16) SWAP(15, 17) SWAP(15, 16)
	SWAP(21, 22) SWAP(20, 22) SWAP(20, 21) SWAP(23, 24) SWAP(25, 26) SWAP(23, 25)
	SWAP(24, 26) SWAP(24, 25) SWAP(20, 24) SWAP(20, 23) SWAP(21, 25) SWAP(22, 26)
	SWAP(22, 25) SWAP(21, 23) SWAP(22, 24) SWAP(22, 23) SWAP(13, 20) SWAP(14, 21)
	SWAP(15, 22) SWAP(15, 21) SWAP(14, 20) SWAP(15, 20) SWAP(16, 23) SWAP(17, 24)
	SWAP(17, 23) SWAP(18, 25) SWAP(19, 26) SWAP(19, 25) SWAP(18, 23) SWAP(19, 24)
	SWAP(19, 23) SWAP(16, 20) SWAP(17, 21) SWAP(17, 20) SWAP(18, 22) SWAP(19, 22)
	SWAP(18, 20) SWAP(19, 21) SWAP(19, 20) SWAP(0, 14) SWAP(0, 13) SWAP(1, 15)
	SWAP(2, 16) SWAP(2, 15) SWAP(1, 13) SWAP(2, 14) SWAP(2, 13) SWAP(3, 17)
	SWAP(4, 18) SWAP(5, 19) SWAP(5, 18) SWAP(4, 17) SWAP(5, 17) SWAP(3, 14)
	SWAP(3, 13) SWAP(4, 15) SWAP(5, 16) SWAP(5, 15) SWAP(4, 13) SWAP(5, 14)
	SWAP(5, 13) SWAP(6, 20) SWAP(7, 21) SWAP(8, 22) SWAP(8, 21) SWAP(7, 20)
	SWAP(8, 20) SWAP(9, 23) SWAP(10, 24) SWAP(10, 23) SWAP(11, 25) SWAP(12, 26)
	SWAP(12, 25) SWAP(11, 23) SWAP(12, 24) SWAP(12, 23) SWAP(9, 20) SWAP(10, 21)
	SWAP(10, 20) SWAP(11, 22) SWAP(12, 22) SWAP(11, 20) SWAP(12, 21) SWAP(12, 20)
	SWAP(6, 13) SWAP(7, 14) SWAP(8, 15) SWAP(8, 14) SWAP(7, 13) SWAP(8, 13)
	SWAP(9, 16) SWAP(10, 17) SWAP(10, 16) SWAP(11, 18) SWAP(12, 19) SWAP(12, 18)
	SWAP(11, 16) SWAP(12, 17) SWAP(12, 16) SWAP(9, 13) SWAP(10, 14) SWAP(10, 13)
	SWAP(11, 15) SWAP(12, 15) SWAP(11, 13) SWAP(12, 14) SWAP(12, 13)
	//@formatter:on
#undef SWAP
}

#define FOR_EACH_NEIGHBOUR_BEGIN(a, b, atoms, atomN, gridTable, gridTableN) \
{ \
    uint3 __coord = (uint3) ( \
            coordAtZCurveGridIndex0(a->zIndex), \
            coordAtZCurveGridIndex1(a->zIndex), \
            coordAtZCurveGridIndex2(a->zIndex)); \
    size_t __offsets[27] = { \
        zCurveGridIndexAtCoord(__coord.x - 1, __coord.y - 1, __coord.z - 1), \
        zCurveGridIndexAtCoord(__coord.x + 0, __coord.y - 1, __coord.z - 1), \
        zCurveGridIndexAtCoord(__coord.x + 1, __coord.y - 1, __coord.z - 1), \
        zCurveGridIndexAtCoord(__coord.x - 1, __coord.y + 0, __coord.z - 1), \
        zCurveGridIndexAtCoord(__coord.x + 0, __coord.y + 0, __coord.z - 1), \
        zCurveGridIndexAtCoord(__coord.x + 1, __coord.y + 0, __coord.z - 1), \
        zCurveGridIndexAtCoord(__coord.x - 1, __coord.y + 1, __coord.z - 1), \
        zCurveGridIndexAtCoord(__coord.x + 0, __coord.y + 1, __coord.z - 1), \
        zCurveGridIndexAtCoord(__coord.x + 1, __coord.y + 1, __coord.z - 1), \
        zCurveGridIndexAtCoord(__coord.x - 1, __coord.y - 1, __coord.z + 0), \
        zCurveGridIndexAtCoord(__coord.x + 0, __coord.y - 1, __coord.z + 0), \
        zCurveGridIndexAtCoord(__coord.x + 1, __coord.y - 1, __coord.z + 0), \
        zCurveGridIndexAtCoord(__coord.x - 1, __coord.y + 0, __coord.z + 0), \
        zCurveGridIndexAtCoord(__coord.x + 0, __coord.y + 0, __coord.z + 0), \
        zCurveGridIndexAtCoord(__coord.x + 1, __coord.y + 0, __coord.z + 0), \
        zCurveGridIndexAtCoord(__coord.x - 1, __coord.y + 1, __coord.z + 0), \
        zCurveGridIndexAtCoord(__coord.x + 0, __coord.y + 1, __coord.z + 0), \
        zCurveGridIndexAtCoord(__coord.x + 1, __coord.y + 1, __coord.z + 0), \
        zCurveGridIndexAtCoord(__coord.x - 1, __coord.y - 1, __coord.z + 1), \
        zCurveGridIndexAtCoord(__coord.x + 0, __coord.y - 1, __coord.z + 1), \
        zCurveGridIndexAtCoord(__coord.x + 1, __coord.y - 1, __coord.z + 1), \
        zCurveGridIndexAtCoord(__coord.x - 1, __coord.y + 0, __coord.z + 1), \
        zCurveGridIndexAtCoord(__coord.x + 0, __coord.y + 0, __coord.z + 1), \
        zCurveGridIndexAtCoord(__coord.x + 1, __coord.y + 0, __coord.z + 1), \
        zCurveGridIndexAtCoord(__coord.x - 1, __coord.y + 1, __coord.z + 1), \
        zCurveGridIndexAtCoord(__coord.x + 0, __coord.y + 1, __coord.z + 1), \
        zCurveGridIndexAtCoord(__coord.x + 1, __coord.y + 1, __coord.z + 1) \
    }; \
    sortArray27(__offsets);  \
    size_t __lastStart = 0; \
    size_t __lastEnd = 0; \
    for (size_t __i = 0; __i < 27; ++__i) { \
        size_t __offset = __offsets[__i]; \
        size_t __start = (gridTable)[__offset]; \
        size_t __end = ((__offset + 1) < (gridTableN)) ? (gridTable)[__offset + 1] : (atomN); \
        for (size_t __ni = __start; __ni < __end; ++__ni) { \
            const global ClSphAtom *b = &(atoms)[__ni]; \


#define FOR_EACH_NEIGHBOUR_END  \
        } \
        __lastStart = __start; \
        __lastEnd = __end; \
    } \
} \

#else

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


#endif
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
				const float r = fast_distance(a->pStar, b->pStar);
				norm2V = fma(spikyKernelGradient(a->pStar, b->pStar, r), 1.f / RHO, norm2V);
				rho = fma(b->particle.mass, poly6Kernel(r), rho);
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
				const float r = fast_distance(a->pStar, b->pStar);
				const float corr = -CorrK * pow(poly6Kernel(r) / p6DeltaQ, CorrN);
				const float factor = (a->lambda + b->lambda + corr) / RHO;
				deltaP = fma(spikyKernelGradient(a->pStar, b->pStar, r), factor, deltaP);
	FOR_EACH_NEIGHBOUR_END

	a->deltaP = deltaP;

	// collision


	float3 currentP = (a->pStar + a->deltaP) * config.scale;
	float3 currentV = a->particle.velocity;
	currentP = clamp(currentP, -500.f, 500.f);
	// TODO handle colliders

	a->pStar = currentP / config.scale;
	a->particle.velocity = currentV;


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
	a->particle.velocity = fma(deltaX, (1.f / config.dt), a->particle.velocity) * VD;
	results[id] = a->particle;
}
