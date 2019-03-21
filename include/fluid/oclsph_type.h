#ifndef LIBFLUID_CLSPH_TYPE
#define LIBFLUID_CLSPH_TYPE

#include "cl_types.h"

typedef struct ClSphConfig {
	ALIGNED_(4) float dt;
	ALIGNED_(4) float scale;
	ALIGNED_(8) size_t iteration;
	ALIGNED_(16) float3 constForce;
	ALIGNED_(16) float3 min;
	ALIGNED_(16) float3 max;
} ClSphConfig;

typedef enum ALIGNED_(4) ClSphType {
	Fluid, Obstacle
} ClSphType;

typedef struct ClSphParticle {
	ALIGNED_(8) size_t id;
	ClSphType type;
	ALIGNED_(4) float mass;
	ALIGNED_(16) float3 position;
	ALIGNED_(16) float3 velocity;
//	ALIGNED_(16) float3 __padding;
} ClSphParticle;

typedef struct ClSphAtom {
	ClSphParticle particle;
	ALIGNED_(16) float3 pStar;
	ALIGNED_(16) float3 deltaP;
	ALIGNED_(16) float3 omega;
	ALIGNED_(4) float lambda;
	ALIGNED_(8) size_t zIndex;
//	ALIGNED_(4) float __padding;

} ClSphAtom;

#endif //LIBFLUID_CLSPH_TYPE
