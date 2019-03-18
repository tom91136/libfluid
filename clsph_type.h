#ifndef LIBFLUID_CLSPH_TYPE
#define LIBFLUID_CLSPH_TYPE

#include "cl_types.h"

typedef struct ALIGNED_(
		sizeof(float) +
		sizeof(float) +
		sizeof(size_t) +
		sizeof(float3))
ClSphConfig {
	float scale;
	float dt;
	size_t iteration;
	float3 constForce;
} ClSphConfig;

typedef enum ALIGNED_(sizeof(int)) ClSphType {
	Fluid, Obstacle
} ClSphType;

typedef struct ALIGNED_(
		sizeof(size_t) +
		sizeof(ClSphType) +
		sizeof(float) +
		sizeof(float3) +
		sizeof(float3) + 16)
ClSphParticle {
	size_t id;
	ClSphType type;
	float mass;
	float3 position;
	float3 velocity;
} ClSphParticle;


typedef struct ALIGNED_(
		sizeof(ClSphParticle) +
		sizeof(float3) +
		sizeof(float3) +
		sizeof(float3) +
		sizeof(float) +
		sizeof(size_t) + 4)
ClSphAtom {

	ClSphParticle particle;

	float3 pStar;
	float3 deltaP;
	float3 omega;
	float lambda;
	size_t zIndex;
}  __attribute__ ((aligned)) ClSphAtom;

#endif //LIBFLUID_CLSPH_TYPE
