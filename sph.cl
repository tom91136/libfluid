
const constant float VD = 0.49;// Velocity dampening;
const constant float RHO = 6378; // Reference density;
const constant float EPSILON = 0.00000001;
const constant float CFM_EPSILON = 600.0; // CFM propagation;

const constant float C = 0.00001;
const constant float VORTICITY_EPSILON = 0.0005;
const constant float CorrK = 0.0001;
const constant float CorrN = 4.f;


typedef struct {
	int id;
	float mass;
	float4 now;
	float lambda;
	float4 deltaP;
	float4 omega;
	float4 velocity;
	int *neighbours;
} Atom;


float poly6(float r, float h) {
	const float poly6Factor = 315.f / (64.f * M_PI_F * (h, 9.f));
	return r <= h ? poly6Factor * ((h * h) - (r * r), 3.f) : 0.f;
}

float4 spikyKernelGradient(float4 x, float4 y, float h, float *r) {
	const float spikyKernelFactor = -(45.f / (M_PI_F * pow(h, 6.f)));
	*r = distance(x, y);
	return !(*r <= h && *r >= EPSILON) ?
		   (float4)(0.f, 0.f, 0.f, 0.f) :
		   (x - y) * (spikyKernelFactor * (pow(h - (*r), 2.f) / (*r)));
}


kernel void sph(
		global Atom* atoms,
		global int* out,
const int size
) {
const int gid = get_global_id(0);
atoms[gid].lambda = 120.f;
};