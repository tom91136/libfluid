#ifndef LIBFLUID_OCLSPH_COLLISION
#define LIBFLUID_OCLSPH_COLLISION

#include "cl_types.h"
#include "oclsph_type.h"

inline bool inTrig(ClSphTraiangle triangle, float3 P) {
	float3 v0 = triangle.c - triangle.a;
	float3 v1 = triangle.b - triangle.a;
	float3 v2 = P - triangle.a;
// Compute dot products
	float dot00 = dot(v0, v0);
	float dot01 = dot(v0, v1);
	float dot02 = dot(v0, v2);
	float dot11 = dot(v1, v1);
	float dot12 = dot(v1, v2);
// Compute barycentric coordinates
	float invDenom = native_recip(dot00 * dot11 - dot01 * dot01);
	float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
	float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
	return (u >= 0) && (v >= 0) && (u + v < 1);
}


bool collideTriangle2(
		const global ClSphTraiangle *mesh, const uint meshN,
		float3 prev, ClSphResponse *response
) {

	float3 p = response->position;
	float3 velocity = response->velocity;
//		printf("Trig =[%d]\n", meshN);

	for (size_t i = 0; i < meshN; i++) {
		const ClSphTraiangle triangle = mesh[i];

//		printf("Trig[%d] = %f %f %f\n", i, triangle.c.x, triangle.c.y, triangle.c.z);

		const float3 n = cross((triangle.b - triangle.a), (triangle.c - triangle.a));
		const float3 nn = fast_normalize(n);

		float t = dot(nn, triangle.a) - dot(nn, p);
		float3 p0 = p + (nn * t);

		if (inTrig(triangle, p0) && fast_distance(p, p0) < 10.f) {
			float3 r = velocity - (nn * 2 * dot(velocity, nn));
			response->position = p0;
			response->velocity = r;
			return true;
		}
	}
	return false;
}


#endif //LIBFLUID_OCLSPH_COLLISION
