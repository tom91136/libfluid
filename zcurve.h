#ifndef LIBFLUID_ZCURVE_H
#define LIBFLUID_ZCURVE_H


#ifndef __OPENCL_C_VERSION__

#include <cstddef>

#endif


const static void sortArray16(size_t d[27]) {
#define SWAP(x, y) if (d[y] < d[x]) { int tmp = d[x]; d[x] = d[y]; d[y] = tmp; }
	SWAP(1, 2);
	SWAP(0, 2);
	SWAP(0, 1);
	SWAP(4, 5);
	SWAP(3, 5);
	SWAP(3, 4);
	SWAP(0, 3);
	SWAP(1, 4);
	SWAP(2, 5);
	SWAP(2, 4);
	SWAP(1, 3);
	SWAP(2, 3);
	SWAP(7, 8);
	SWAP(6, 8);
	SWAP(6, 7);
	SWAP(9, 10);
	SWAP(11, 12);
	SWAP(9, 11);
	SWAP(10, 12);
	SWAP(10, 11);
	SWAP(6, 10);
	SWAP(6, 9);
	SWAP(7, 11);
	SWAP(8, 12);
	SWAP(8, 11);
	SWAP(7, 9);
	SWAP(8, 10);
	SWAP(8, 9);
	SWAP(0, 7);
	SWAP(0, 6);
	SWAP(1, 8);
	SWAP(2, 9);
	SWAP(2, 8);
	SWAP(1, 6);
	SWAP(2, 7);
	SWAP(2, 6);
	SWAP(3, 10);
	SWAP(4, 11);
	SWAP(5, 12);
	SWAP(5, 11);
	SWAP(4, 10);
	SWAP(5, 10);
	SWAP(3, 7);
	SWAP(3, 6);
	SWAP(4, 8);
	SWAP(5, 9);
	SWAP(5, 8);
	SWAP(4, 6);
	SWAP(5, 7);
	SWAP(5, 6);
	SWAP(14, 15);
	SWAP(13, 15);
	SWAP(13, 14);
	SWAP(16, 17);
	SWAP(18, 19);
	SWAP(16, 18);
	SWAP(17, 19);
	SWAP(17, 18);
	SWAP(13, 17);
	SWAP(13, 16);
	SWAP(14, 18);
	SWAP(15, 19);
	SWAP(15, 18);
	SWAP(14, 16);
	SWAP(15, 17);
	SWAP(15, 16);
	SWAP(21, 22);
	SWAP(20, 22);
	SWAP(20, 21);
	SWAP(23, 24);
	SWAP(25, 26);
	SWAP(23, 25);
	SWAP(24, 26);
	SWAP(24, 25);
	SWAP(20, 24);
	SWAP(20, 23);
	SWAP(21, 25);
	SWAP(22, 26);
	SWAP(22, 25);
	SWAP(21, 23);
	SWAP(22, 24);
	SWAP(22, 23);
	SWAP(13, 20);
	SWAP(14, 21);
	SWAP(15, 22);
	SWAP(15, 21);
	SWAP(14, 20);
	SWAP(15, 20);
	SWAP(16, 23);
	SWAP(17, 24);
	SWAP(17, 23);
	SWAP(18, 25);
	SWAP(19, 26);
	SWAP(19, 25);
	SWAP(18, 23);
	SWAP(19, 24);
	SWAP(19, 23);
	SWAP(16, 20);
	SWAP(17, 21);
	SWAP(17, 20);
	SWAP(18, 22);
	SWAP(19, 22);
	SWAP(18, 20);
	SWAP(19, 21);
	SWAP(19, 20);
	SWAP(0, 14);
	SWAP(0, 13);
	SWAP(1, 15);
	SWAP(2, 16);
	SWAP(2, 15);
	SWAP(1, 13);
	SWAP(2, 14);
	SWAP(2, 13);
	SWAP(3, 17);
	SWAP(4, 18);
	SWAP(5, 19);
	SWAP(5, 18);
	SWAP(4, 17);
	SWAP(5, 17);
	SWAP(3, 14);
	SWAP(3, 13);
	SWAP(4, 15);
	SWAP(5, 16);
	SWAP(5, 15);
	SWAP(4, 13);
	SWAP(5, 14);
	SWAP(5, 13);
	SWAP(6, 20);
	SWAP(7, 21);
	SWAP(8, 22);
	SWAP(8, 21);
	SWAP(7, 20);
	SWAP(8, 20);
	SWAP(9, 23);
	SWAP(10, 24);
	SWAP(10, 23);
	SWAP(11, 25);
	SWAP(12, 26);
	SWAP(12, 25);
	SWAP(11, 23);
	SWAP(12, 24);
	SWAP(12, 23);
	SWAP(9, 20);
	SWAP(10, 21);
	SWAP(10, 20);
	SWAP(11, 22);
	SWAP(12, 22);
	SWAP(11, 20);
	SWAP(12, 21);
	SWAP(12, 20);
	SWAP(6, 13);
	SWAP(7, 14);
	SWAP(8, 15);
	SWAP(8, 14);
	SWAP(7, 13);
	SWAP(8, 13);
	SWAP(9, 16);
	SWAP(10, 17);
	SWAP(10, 16);
	SWAP(11, 18);
	SWAP(12, 19);
	SWAP(12, 18);
	SWAP(11, 16);
	SWAP(12, 17);
	SWAP(12, 16);
	SWAP(9, 13);
	SWAP(10, 14);
	SWAP(10, 13);
	SWAP(11, 15);
	SWAP(12, 15);
	SWAP(11, 13);
	SWAP(12, 14);
	SWAP(12, 13);
#undef SWAP

}


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

const static inline void genTable(const size_t x, const size_t y, const size_t z,
                                  const size_t count, const size_t *LUT, const size_t pMax,
                                  void(*callback)(size_t)) {
	size_t offsets[27] = {
			zCurveGridIndexAtCoord(x - 1, y - 1, z - 1),
			zCurveGridIndexAtCoord(x + 0, y - 1, z - 1),
			zCurveGridIndexAtCoord(x + 1, y - 1, z - 1),
			zCurveGridIndexAtCoord(x - 1, y + 0, z - 1),
			zCurveGridIndexAtCoord(x + 0, y + 0, z - 1),
			zCurveGridIndexAtCoord(x + 1, y + 0, z - 1),
			zCurveGridIndexAtCoord(x - 1, y + 1, z - 1),
			zCurveGridIndexAtCoord(x + 0, y + 1, z - 1),
			zCurveGridIndexAtCoord(x + 1, y + 1, z - 1),
			zCurveGridIndexAtCoord(x - 1, y - 1, z + 0),
			zCurveGridIndexAtCoord(x + 0, y - 1, z + 0),
			zCurveGridIndexAtCoord(x + 1, y - 1, z + 0),
			zCurveGridIndexAtCoord(x - 1, y + 0, z + 0),
			zCurveGridIndexAtCoord(x + 0, y + 0, z + 0),
			zCurveGridIndexAtCoord(x + 1, y + 0, z + 0),
			zCurveGridIndexAtCoord(x - 1, y + 1, z + 0),
			zCurveGridIndexAtCoord(x + 0, y + 1, z + 0),
			zCurveGridIndexAtCoord(x + 1, y + 1, z + 0),
			zCurveGridIndexAtCoord(x - 1, y - 1, z + 1),
			zCurveGridIndexAtCoord(x + 0, y - 1, z + 1),
			zCurveGridIndexAtCoord(x + 1, y - 1, z + 1),
			zCurveGridIndexAtCoord(x - 1, y + 0, z + 1),
			zCurveGridIndexAtCoord(x + 0, y + 0, z + 1),
			zCurveGridIndexAtCoord(x + 1, y + 0, z + 1),
			zCurveGridIndexAtCoord(x - 1, y + 1, z + 1),
			zCurveGridIndexAtCoord(x + 0, y + 1, z + 1),
			zCurveGridIndexAtCoord(x + 1, y + 1, z + 1)
	};
//	sortArray16(offsets);
	size_t lastStart = 0;
	size_t lastEnd = 0;
	for (int i = 0; i < 27; ++i) {
		size_t offset = offsets[i];

		size_t start = LUT[offset];
		size_t end = ((offset + 1) < count) ? LUT[offset + 1] : pMax;

		if (lastStart == start && lastEnd == end) continue;

		for (size_t j = start; j < end; ++j) callback(j);
		lastStart = start;
		lastEnd = end;
	}
}


#endif //LIBFLUID_ZCURVE_H
