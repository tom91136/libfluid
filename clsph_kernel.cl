


#ifndef __OPENCL_C_VERSION__

#include <tgmath.h>
#include "clsph_types.h"

#define global
#define kernel
#define constant
#define global
#define M_PI_F 3.1415926f
#define INCL
#else

#include "clsph_types.h"

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




// 1 hash pass one:
//  in : Atom[N]
//  in/out : uint[N] ( |neighbour size * 27| )

// 2 host : prefix sum[N]

// 3 hash pass two
//  in : Atom[N]
//  in/out : uint[N] (prefix sum)
//  in/out : uint[N] (prefix sum)


inline int3 snap(float3 a) {
	return convert_int3_rtp(a / H);
}

inline size_t hash(int3 a) {
	size_t res = 17;
	res = res * 31 + a.x;
	res = res * 31 + a.y;
	res = res * 31 + a.z;
	return res;
}


inline size_t findSlotUnfenced(const global Entry *buckets, const size_t limit, const int3 x) {

	size_t i = hash(x) % limit;
	while (buckets[i].value != 0 && all(buckets[i].key != x))
		i = (i + 1) % limit;
	return i;
}


kernel void nn_phase_size(
		const global ClSphAtom *atoms,
		global Entry *entries, const uint limit
) {
	const size_t id = get_global_id(0);
	entries[id].value = 0;
	const global ClSphAtom *a = &atoms[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (size_t x = 0; x < NEIGHBOUR_SIZE; x++)
		for (size_t y = 0; y < NEIGHBOUR_SIZE; y++)
			for (size_t z = 0; z < NEIGHBOUR_SIZE; z++) {
				int3 snapped = snap(
						a->now + (float3) (NEIGHBOURS[x], NEIGHBOURS[y], NEIGHBOURS[z]));
				mem_fence(CLK_LOCAL_MEM_FENCE);
				size_t slot = findSlotUnfenced(entries, limit, snapped);
				global Entry *e = &entries[slot];
				if (e->value == 0) e->key = snapped;
				e->value++;

//				printf("[%d], <(%d,%d,%d), %d>", id , e->key.x,e->key.y,e->key.z, e->value );
//				mem_fence(CLK_LOCAL_MEM_FENCE);
			}
//	barrier(CLK_GLOBAL_MEM_FENCE);
//	printf("[%ld] atom=%d <(%d,%d,%d), %d>", id, a.id, )
}

//kernel void nn_phase_store(
//		const global ClSphAtom *atoms,
//		const global Entry *entries, const uint limit,
//		const global uint2 *offsetLengthPair,
//		const uint *neighbours
//) {
//	const size_t id = get_global_id(0);
//
//	const uint2 pair = offsetLengthPair[id];
//	const uint offset = pair.x
//	const uint length = pair.y
//
//	const global ClSphAtom *a = &atoms[id];
//	size_t i = offset;
//	for (size_t x = 0; x < NEIGHBOUR_SIZE; x++)
//		for (size_t y = 0; y < NEIGHBOUR_SIZE; y++)
//			for (size_t z = 0; z < NEIGHBOUR_SIZE; z++) {
//				float3 snapped = snap(a->now + (float3) (x, y, z));
//				size_t slot = findSlotUnfenced(buckets, limit, snapped);
//				global Entry *e = &buckets[slot];
//
//				neighbours[i++] = e->value
//
//
//			}
//
//}


inline size_t neighbourStart(const global ClSphAtom *atom) { return atom->neighbourOffset; }

inline size_t neighbourEnd(const global ClSphAtom *atom) {
	return neighbourStart(atom) + atom->neighbourCount;
};


kernel void sph_lambda(
		const ClSphConfig config,
		global ClSphAtom *atoms,
		const global uint *neighbours
) {
	const size_t id = get_global_id(0);
	global ClSphAtom *a = &atoms[id];
	float3 norm2V = (float3) (0.f);
	float rho = 0.f;
	for (size_t i = neighbourStart(a); i < neighbourEnd(a); i++) {
		const global ClSphAtom *b = &atoms[neighbours[i]];
		const float r = distance(a->now, b->now);
		norm2V += spikyKernelGradient(a->now, b->now, r) * (1.f / RHO);
		rho += b->mass * poly6Kernel(r);
	}
	float norm2 = dot(norm2V, norm2V); // dot self = length2
	float C1 = (rho / RHO - 1.f);
	a->lambda = -C1 / (norm2 + CFM_EPSILON);
}


kernel void sph_delta(
		const ClSphConfig config,
		global ClSphAtom *atoms,
		const global uint *neighbours
) {
	const size_t id = get_global_id(0);

	global ClSphAtom *a = &atoms[id];

	const float CorrDeltaQ = 0.3f * H;
	const float p6DeltaQ = poly6Kernel(CorrDeltaQ);

	float3 deltaP = (float3) (0.f);
	for (size_t i = neighbourStart(a); i < neighbourEnd(a); i++) {
		const global ClSphAtom *b = &atoms[neighbours[i]];
		const float r = distance(a->now, b->now);
		const float corr = -CorrK * pow(poly6Kernel(r) / p6DeltaQ, CorrN);
		const float factor = (a->lambda + b->lambda + corr) / RHO;
		deltaP += spikyKernelGradient(a->now, b->now, r) * factor;
	}
	a->deltaP = deltaP;

	// collision
	float3 currentP = (a->now + a->deltaP) * config.scale;
	float3 currentV = a->velocity;
	currentP = clamp(currentP, -500.f, 500.f);
	// TODO handle colliders

	a->now = currentP / config.scale;
	a->velocity = currentV;

//	size_t x = 0;
//	for (size_t i = neighbourStart(a); i < neighbourEnd(a); i++) {
//		x += atoms[neighbours[i]].id;
//	}



#ifdef DEBUG
	printf("[%ld] config { scale=%f, dt=%f} p={id=%ld, mass=%f lam=%f, deltaP=(%f,%f,%f)}\n",
		   id, config.scale, config.dt,
		   a->id, a->mass, a->lambda,
		   a->deltaP.x, a->deltaP.y, a->deltaP.z);
#endif
}


kernel void sph_finalise(
		const ClSphConfig config,
		global ClSphAtom *atoms,
		global ClSphResult *results
) {
	const size_t id = get_global_id(0);

	global ClSphAtom *a = &atoms[id];


	const float3 deltaX = a->now - a->position / config.scale;
	results[id].position = a->now * config.scale;
	results[id].velocity = (deltaX * (1.f / config.dt) + a->velocity) * VD;
}
