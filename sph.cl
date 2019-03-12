


#ifndef __OPENCL_C_VERSION__

#include "sph.h"

#define global
#define kernel
#define constant
typedef unsigned int uint
#define INCL
#else

#include "../sph.h"

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

float poly6Kernel(float r, float h) {
	const float poly6Factor = 315.f / (64.f * M_PI_F * (h, 9.f));
	return r <= h ? poly6Factor * ((h * h) - (r * r), 3.f) : 0.f;
}

float3 spikyKernelGradient(float3 x, float3 y, float h, float *r) {
	const float spikyKernelFactor = -(45.f / (M_PI_F * pow(h, 6.f)));
	*r = distance(x, y);
	return !(*r <= h && *r >= EPSILON) ?
		   (float3) (0.f, 0.f, 0.f) :
		   (x - y) * (spikyKernelFactor * (pow(h - (*r), 2.f) / (*r)));
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
		float r;
		rho += b->mass * poly6Kernel(r, config.h);
		norm2V += spikyKernelGradient(a->now, b->now, config.h, &r) * (1.f / RHO);
	}

	float norm2 = length(norm2V);
	float C = (rho / RHO - 1.f);
	return -C / (norm2 + CFM_EPSILON);
}

kernel void sph(
		const Config config,
		global Atom *atoms,
		const uint size,
		const global uint *neighbours,
		global float *out
) {
	const size_t id = get_global_id(0);
	global Atom *a = &atoms[id];
	size_t x = 0;

	for (size_t iter = 0; iter < config.iteration; ++iter) {
		a->lambda = lambda(config, a, atoms, neighbours);
		// TODO the rest of the computation
	}



//	a = neighbours[neighbourStart(atom)];
	out[id] = (float) x + config.scale;
}
