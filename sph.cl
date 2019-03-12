


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


float poly6(float r, float h) {
	const float poly6Factor = 315.f / (64.f * M_PI_F * (h, 9.f));
	return r <= h ? poly6Factor * ((h * h) - (r * r), 3.f) : 0.f;
}

float4 spikyKernelGradient(float4 x, float4 y, float h, float *r) {
	const float spikyKernelFactor = -(45.f / (M_PI_F * pow(h, 6.f)));
	*r = distance(x, y);
	return !(*r <= h && *r >= EPSILON) ?
	       (float4) (0.f, 0.f, 0.f, 0.f) :
	       (x - y) * (spikyKernelFactor * (pow(h - (*r), 2.f) / (*r)));
}

uint neighbourStart(const global Atom *atom) { return atom->neighbourOffset; }

uint neighbourEnd(const global Atom *atom) {
	return neighbourStart( atom) + atom->neighbourCount;
};

kernel void sph(
		const global Atom *atoms,
		const global uint *neighbours,
		const uint size,
		global float *out
) {
	const size_t id = get_global_id(0);
	const global Atom *atom = &atoms[id];
	size_t a = 0;
	for (size_t i = neighbourStart(atom); i < neighbourEnd(atom); i++) {
		a += neighbours[i];
	}
//	a = neighbours[neighbourStart(atom)];
	out[id] = (float) a;
}
