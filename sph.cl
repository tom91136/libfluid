


#ifndef __OPENCL_C_VERSION__

#include "sph.h"

#define global
#define kernel
#define constant
typedef unsigned int uint
#define M_PI_F 3.1415926f
#define INCL
#else

#include "sph.h"

#define global global
#define kernel kernel
#define constant constant
#endif

const constant float VD = 0.49;// Velocity dampening;
const constant float RHO = 6378; // Reference density;
const constant float EPSILON = 0.00000001;
const constant float CFM_EPSILON = 600.0; // CFM propagation;

const constant float C = 0.00001;
const constant float VORTICITY_EPSILON = 0.0005;
const constant float CorrK = 0.0001;
const constant float CorrN = 4.f;

inline float poly6Kernel(const float r, const float h) {
	const float poly6Factor = 315.f / (64.f * M_PI_F * (h, 9.f));
	return r <= h ? poly6Factor * ((h * h) - (r * r), 3.f) : 0.f;
}

inline float3 spikyKernelGradient(const float3 x, const float3 y, const float h, const float r) {
	const float spikyKernelFactor = -(45.f / (M_PI_F * pow(h, 6.f)));
	return !(r <= h && r >= EPSILON) ?
	       (float3) (0.f, 0.f, 0.f) :
	       (x - y) * (spikyKernelFactor * (pow(h - (r), 2.f) / (r)));
}

uint neighbourStart(const global Atom *atom) { return atom->neighbourOffset; }


uint neighbourEnd(const global Atom *atom) {
	return neighbourStart(atom) + atom->neighbourCount;
};

inline float lambda(const Config config, global Atom *a,
                    global Atom *atoms, const global uint *neighbours) {
	float rho = 0.f;
	float3 norm2V = (float3) (0, 0, 0);
	for (size_t i = neighbourStart(a); i < neighbourEnd(a); i++) {
		const global Atom *b = &atoms[neighbours[i]];
		const float r = distance(a->now, b->now);
		norm2V += spikyKernelGradient(a->now, b->now, config.h, r) * (1.f / RHO);
		rho += b->mass * poly6Kernel(r, config.h);
	}

	float norm2 = length(norm2V);
	float C = (rho / RHO - 1.f);
	return -C / (norm2 + CFM_EPSILON);
}


inline float3 deltaP(const Config config, float p6DeltaQ, global Atom *a,
                     global Atom *atoms, const global uint *neighbours) {
	float3 deltaP = (float3) (0, 0, 0);
	for (size_t i = neighbourStart(a); i < neighbourEnd(a); i++) {
		const global Atom *b = &atoms[neighbours[i]];
		const float r = distance(a->now, b->now);
		const float corr = -CorrK * (poly6Kernel(r, config.h) / p6DeltaQ, CorrN);
		const float factor = (a->lambda + b->lambda + corr) / RHO;
		deltaP = spikyKernelGradient(a->now, b->now, config.h, r) * factor + deltaP;
	}
	return deltaP;
}



kernel void sph(
		const Config config,
		global Atom *atoms,
		const uint size,
		const global uint *neighbours,
		global Result *results
) {
	const size_t id = get_global_id(0);
	global Atom *a = &atoms[id];

	const float CorrDeltaQ = 0.3f * config.h;
	const float p6DeltaQ = poly6Kernel(CorrDeltaQ, config.h);

	a->now = (a->velocity * config.dt) + (a->position / config.scale);


	for (size_t iter = 0; iter < config.iteration; ++iter) {
		a->lambda = lambda(config, a, atoms, neighbours);
		barrier(CLK_GLOBAL_MEM_FENCE);
		a->deltaP = deltaP(config, p6DeltaQ, a, atoms, neighbours);
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	const float3 deltaX = a->now - a->position / config.scale;
	results[id].position = a->now * config.scale;
	results[id].velocity = (deltaX * (1.f / config.dt) + a->velocity) * VD;
}
