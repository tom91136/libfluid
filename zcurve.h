#ifndef LIBFLUID_ZCURVE_H
#define LIBFLUID_ZCURVE_H


#ifndef __OPENCL_C_VERSION__

#include <cstddef>


#endif





const static size_t uninterleave(size_t value) {
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

const static inline size_t coordAtZCurveGridIndex0(size_t index) {
	return uninterleave((index) & 0x9249249);
}

const static inline size_t coordAtZCurveGridIndex1(size_t index) {
	return uninterleave((index >> 1) & 0x9249249);
}

const static inline size_t coordAtZCurveGridIndex2(size_t index) {
	return uninterleave((index >> 2) & 0x9249249);
}

//const static glm::tvec3<size_t> coordAtZCurveGridIndex(size_t index) {
//	size_t i_x = index & 0x9249249;
//	size_t i_y = (index >> 1) & 0x9249249;
//	size_t i_z = (index >> 2) & 0x9249249;
//	return glm::tvec3<size_t>(uninterleave(i_x), uninterleave(i_y), uninterleave(i_z));
//}


const static inline size_t zCurveGridIndexAtCoord(size_t x, size_t y, size_t z) {
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

//const static inline void genTable(const size_t x, const size_t y, const size_t z,
//                                  const size_t count, const size_t *LUT, const size_t pMax,
//                                  void(*callback)(size_t)) {
//	size_t offsets[27] = {
//			zCurveGridIndexAtCoord(x - 1, y - 1, z - 1),
//			zCurveGridIndexAtCoord(x + 0, y - 1, z - 1),
//			zCurveGridIndexAtCoord(x + 1, y - 1, z - 1),
//			zCurveGridIndexAtCoord(x - 1, y + 0, z - 1),
//			zCurveGridIndexAtCoord(x + 0, y + 0, z - 1),
//			zCurveGridIndexAtCoord(x + 1, y + 0, z - 1),
//			zCurveGridIndexAtCoord(x - 1, y + 1, z - 1),
//			zCurveGridIndexAtCoord(x + 0, y + 1, z - 1),
//			zCurveGridIndexAtCoord(x + 1, y + 1, z - 1),
//			zCurveGridIndexAtCoord(x - 1, y - 1, z + 0),
//			zCurveGridIndexAtCoord(x + 0, y - 1, z + 0),
//			zCurveGridIndexAtCoord(x + 1, y - 1, z + 0),
//			zCurveGridIndexAtCoord(x - 1, y + 0, z + 0),
//			zCurveGridIndexAtCoord(x + 0, y + 0, z + 0),
//			zCurveGridIndexAtCoord(x + 1, y + 0, z + 0),
//			zCurveGridIndexAtCoord(x - 1, y + 1, z + 0),
//			zCurveGridIndexAtCoord(x + 0, y + 1, z + 0),
//			zCurveGridIndexAtCoord(x + 1, y + 1, z + 0),
//			zCurveGridIndexAtCoord(x - 1, y - 1, z + 1),
//			zCurveGridIndexAtCoord(x + 0, y - 1, z + 1),
//			zCurveGridIndexAtCoord(x + 1, y - 1, z + 1),
//			zCurveGridIndexAtCoord(x - 1, y + 0, z + 1),
//			zCurveGridIndexAtCoord(x + 0, y + 0, z + 1),
//			zCurveGridIndexAtCoord(x + 1, y + 0, z + 1),
//			zCurveGridIndexAtCoord(x - 1, y + 1, z + 1),
//			zCurveGridIndexAtCoord(x + 0, y + 1, z + 1),
//			zCurveGridIndexAtCoord(x + 1, y + 1, z + 1)
//	};
////	sortArray16(offsets);
//	size_t lastStart = 0;
//	size_t lastEnd = 0;
//	for (int i = 0; i < 27; ++i) {
//		size_t offset = offsets[i];
//
//		size_t start = LUT[offset];
//		size_t end = ((offset + 1) < count) ? LUT[offset + 1] : pMax;
//
//		if (lastStart == start && lastEnd == end) continue;
//
//		for (size_t j = start; j < end; ++j) callback(j);
//		lastStart = start;
//		lastEnd = end;
//	}
//}


#endif //LIBFLUID_ZCURVE_H
