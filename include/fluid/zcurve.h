#ifndef LIBFLUID_ZCURVE_H
#define LIBFLUID_ZCURVE_H


#ifndef __OPENCL_C_VERSION__

#include <cstddef>

#endif


inline size_t index3d(size_t x, size_t y, size_t z,
                      size_t xMax, size_t yMax, size_t zMax
		) {
	return x * yMax * zMax + y * zMax + z;
}

inline size_t uninterleave(size_t value) {
	size_t ret = 0x0;
	ret |= (value & 0x1) >> 0;
	ret |= (value & 0x8) >> 2;
	ret |= (value & 0x40) >> 4;
	ret |= (value & 0x200) >> 6;
	ret |= (value & 0x1000) >> 8;
	ret |= (value & 0x8000) >> 10;
	ret |= (value & 0x40000) >> 12;
	ret |= (value & 0x200000) >> 14;
	ret |= (value & 0x1000000) >> 16;
	ret |= (value & 0x8000000) >> 18;
	return ret;
}

inline size_t coordAtZCurveGridIndex0(size_t index) {
	return uninterleave((index) & 0x9249249);
}

inline size_t coordAtZCurveGridIndex1(size_t index) {
	return uninterleave((index >> 1) & 0x9249249);
}

inline size_t coordAtZCurveGridIndex2(size_t index) {
	return uninterleave((index >> 2) & 0x9249249);
}

inline size_t zCurveGridIndexAtCoord(size_t x, size_t y, size_t z) {
	x = (x | (x << 16)) & 0x030000FF;
	x = (x | (x << 8)) & 0x0300F00F;
	x = (x | (x << 4)) & 0x030C30C3;
	x = (x | (x << 2)) & 0x09249249;

	y = (y | (y << 16)) & 0x030000FF;
	y = (y | (y << 8)) & 0x0300F00F;
	y = (y | (y << 4)) & 0x030C30C3;
	y = (y | (y << 2)) & 0x09249249;

	z = (z | (z << 16)) & 0x030000FF;
	z = (z | (z << 8)) & 0x0300F00F;
	z = (z | (z << 4)) & 0x030C30C3;
	z = (z | (z << 2)) & 0x09249249;
	return x | y << 1 | z << 2;
}

#endif //LIBFLUID_ZCURVE_H
