#include "cl_types.h"
#include "zcurve.h"
#include "mc.h"
#include "oclsph_type.h"

// Heavily modified version of https://github.com/smistad/GPU-Marching-Cubes
// See https://github.com/smistad/GPU-Marching-Cubes/blob/master/gpu-mc-morton.cl
/*

Copyright 2011 Erik Smistad. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY Erik Smistad ''AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Erik Smistad OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of Erik Smistad.

*/

constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

const constant int4 cubeOffsets[8] = {
		{0, 0, 0, 0},
		{1, 0, 0, 0},
		{0, 0, 1, 0},
		{1, 0, 1, 0},
		{0, 1, 0, 0},
		{1, 1, 0, 0},
		{0, 1, 1, 0},
		{1, 1, 1, 0},
};

static inline size_t zCurveGridIndexAtCoord4(int4 v) {
	return zCurveGridIndexAtCoord(v.x, v.y, v.z);
}


#define DEFINE_CONSTRUCT_HP_LEVEL(name, inType, outType) \
kernel void name(global inType *readHistoPyramid, global outType *writeHistoPyramid) { \
    const size_t x = get_global_id(0); \
    const size_t y = get_global_id(1); \
    const size_t z = get_global_id(2); \
    const size_t writePos = zCurveGridIndexAtCoord(x, y, z); \
    const size_t readPos = zCurveGridIndexAtCoord(x * 2, y * 2, z * 2); \
    writeHistoPyramid[writePos] = \
                readHistoPyramid[readPos] + \
                readHistoPyramid[readPos + 1] + \
                readHistoPyramid[readPos + 2] + \
                readHistoPyramid[readPos + 3] + \
                readHistoPyramid[readPos + 4] + \
                readHistoPyramid[readPos + 5] + \
                readHistoPyramid[readPos + 6] + \
                readHistoPyramid[readPos + 7]; \
} \


DEFINE_CONSTRUCT_HP_LEVEL(constructHPLevelCharChar, uchar, uchar)

DEFINE_CONSTRUCT_HP_LEVEL(constructHPLevelCharShort, uchar, ushort)

DEFINE_CONSTRUCT_HP_LEVEL(constructHPLevelShortShort, ushort, ushort)

DEFINE_CONSTRUCT_HP_LEVEL(constructHPLevelShortInt, ushort, int)

DEFINE_CONSTRUCT_HP_LEVEL(constructHPLevelIntInt, int, int)


#define DEFINE_SCAN_HP_LEVEL(name, inType) \
int4 name(int target, global inType *hp, int4 current) { \
    int8 neighbors = {\
            hp[zCurveGridIndexAtCoord4(current)], \
            hp[zCurveGridIndexAtCoord4(current + cubeOffsets[1])], \
            hp[zCurveGridIndexAtCoord4(current + cubeOffsets[2])], \
            hp[zCurveGridIndexAtCoord4(current + cubeOffsets[3])], \
            hp[zCurveGridIndexAtCoord4(current + cubeOffsets[4])], \
            hp[zCurveGridIndexAtCoord4(current + cubeOffsets[5])], \
            hp[zCurveGridIndexAtCoord4(current + cubeOffsets[6])], \
            hp[zCurveGridIndexAtCoord4(current + cubeOffsets[7])], \
    }; \
    int acc = current.s3 + neighbors.s0; \
    int8 cmp; \
    cmp.s0 = acc <= target; \
    acc += neighbors.s1; \
    cmp.s1 = acc <= target; \
    acc += neighbors.s2; \
    cmp.s2 = acc <= target; \
    acc += neighbors.s3; \
    cmp.s3 = acc <= target; \
    acc += neighbors.s4; \
    cmp.s4 = acc <= target; \
    acc += neighbors.s5; \
    cmp.s5 = acc <= target; \
    acc += neighbors.s6; \
    cmp.s6 = acc <= target; \
    cmp.s7 = 0; \
    current += cubeOffsets[(cmp.s0 + cmp.s1 + cmp.s2 + cmp.s3 + cmp.s4 + cmp.s5 + cmp.s6 + cmp.s7)]; \
    current.s0 = current.s0 * 2; \
    current.s1 = current.s1 * 2; \
    current.s2 = current.s2 * 2; \
    current.s3 = current.s3 + \
                 cmp.s0 * neighbors.s0 + \
                 cmp.s1 * neighbors.s1 + \
                 cmp.s2 * neighbors.s2 + \
                 cmp.s3 * neighbors.s3 + \
                 cmp.s4 * neighbors.s4 + \
                 cmp.s5 * neighbors.s5 + \
                 cmp.s6 * neighbors.s6 + \
                 cmp.s7 * neighbors.s7; \
    return current; \
} \


DEFINE_SCAN_HP_LEVEL(scanHPLevelShort, ushort)

DEFINE_SCAN_HP_LEVEL(scanHPLevelChar, uchar)

DEFINE_SCAN_HP_LEVEL(scanHPLevel, int)


kernel void traverseHP(
		read_only image3d_t rawData,
		read_only image3d_t cubeIndexes,
		global uchar *hp0, // Largest HP
		global uchar *hp1,
		global ushort *hp2,
		global ushort *hp3,
		global ushort *hp4,
		global int *hp5,
#if SIZE > 64
		global int * hp6,
#endif
#if SIZE > 128
		global int * hp7,
#endif
#if SIZE > 256
		global int * hp8,
#endif
#if SIZE > 512
		global int * hp9,
#endif
		global float *VBOBuffer,
		__private int isolevel,
		__private int sum
) {

	int target = get_global_id(0);
	if (target >= sum)
		target = 0;

	int4 cubePosition = {0, 0, 0, 0}; // x,y,z,sum
#if SIZE > 512
	cubePosition = scanHPLevel(target, hp9, cubePosition);
#endif
#if SIZE > 256
	cubePosition = scanHPLevel(target, hp8, cubePosition);
#endif
#if SIZE > 128
	cubePosition = scanHPLevel(target, hp7, cubePosition);
#endif
#if SIZE > 64
	cubePosition = scanHPLevel(target, hp6, cubePosition);
#endif
	cubePosition = scanHPLevel(target, hp5, cubePosition);
	cubePosition = scanHPLevelShort(target, hp4, cubePosition);
	cubePosition = scanHPLevelShort(target, hp3, cubePosition);
	cubePosition = scanHPLevelShort(target, hp2, cubePosition);
	cubePosition = scanHPLevelChar(target, hp1, cubePosition);
	cubePosition = scanHPLevelChar(target, hp0, cubePosition);
	cubePosition.
			x = cubePosition.x / 2;
	cubePosition.
			y = cubePosition.y / 2;
	cubePosition.
			z = cubePosition.z / 2;

	char vertexNr = 0;

// max 5 triangles
	uchar cubeindex = read_imageui(cubeIndexes, sampler, cubePosition).x;
	for (
			int i = (target - cubePosition.s3) * 3;
			i < (target - cubePosition.s3 + 1) * 3; i++) { // for each vertex in triangle
		const uchar edge = triTable[cubeindex * 16 + i];
		const int3 point0 = (int3) (cubePosition.x + offsets3[edge * 6], cubePosition.y +
		                                                                 offsets3[edge * 6 + 1],
				cubePosition.z + offsets3[edge * 6 + 2]);
		const int3 point1 = (int3) (cubePosition.x + offsets3[edge * 6 + 3], cubePosition.y +
		                                                                     offsets3[edge * 6 + 4],
				cubePosition.z + offsets3[edge * 6 + 5]);

// Store vertex in VBO

		const float3 forwardDifference0 = (float3) (
				(float) (-read_imagei(rawData, sampler,
				                      (int4) (point0.x + 1, point0.y, point0.z, 0)).x +
				         read_imagei(rawData, sampler,
				                     (int4) (point0.x - 1, point0.y, point0.z, 0)).x),
						(float) (-read_imagei(rawData, sampler,
						                      (int4) (point0.x, point0.y + 1, point0.z, 0)).x +
						         read_imagei(rawData, sampler,
						                     (int4) (point0.x, point0.y - 1, point0.z, 0)).x),
						(float) (-read_imagei(rawData, sampler,
						                      (int4) (point0.x, point0.y, point0.z + 1, 0)).x +
						         read_imagei(rawData, sampler,
						                     (int4) (point0.x, point0.y, point0.z - 1, 0)).x)
		);
		const float3 forwardDifference1 = (float3) (
				(float) (-read_imagei(rawData, sampler,
				                      (int4) (point1.x + 1, point1.y, point1.z, 0)).x +
				         read_imagei(rawData, sampler,
				                     (int4) (point1.x - 1, point1.y, point1.z, 0)).x),
						(float) (-read_imagei(rawData, sampler,
						                      (int4) (point1.x, point1.y + 1, point1.z, 0)).x +
						         read_imagei(rawData, sampler,
						                     (int4) (point1.x, point1.y - 1, point1.z, 0)).x),
						(float) (-read_imagei(rawData, sampler,
						                      (int4) (point1.x, point1.y, point1.z + 1, 0)).x +
						         read_imagei(rawData, sampler,
						                     (int4) (point1.x, point1.y, point1.z - 1, 0)).x)
		);

		const int value0 = read_imagei(rawData, sampler,
		                               (int4) (point0.x, point0.y, point0.z, 0)).x;
		const float diff = native_divide(
				(float) (isolevel - value0),
				(float) (read_imagei(rawData, sampler, (int4) (point1.x, point1.y, point1.z, 0)).x -
				         value0));

		const float3 vertex = mix((float3) (point0.x, point0.y, point0.z),
		                          (float3) (point1.x, point1.y, point1.z), diff);

		const float3 normal = mix(forwardDifference0, forwardDifference1, diff);


		vstore3(vertex, target * 6 + vertexNr * 2, VBOBuffer);
		vstore3(normal, target * 6 + vertexNr * 2 + 1, VBOBuffer);


		++vertexNr;
	}
}

//kernel void compute_field(H h,
//                          const ClSphConfig config,
//                          global ClSphAtom *atoms, uint atomN,
//                          const global uint *gridTable, uint gridTableN
//) {
//
//}

kernel void classifyCubes(
		global uchar *histoPyramid,
		global uchar *cubeIndexes,
		read_only image3d_t rawData,
		__private int isolevel
) {
	int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};

// Find cube class nr
	const uchar first = read_imagei(rawData, sampler, pos).x;
	const uchar cubeindex =
			((first > isolevel)) |
			((read_imagei(rawData, sampler, pos + cubeOffsets[1]).x > isolevel) << 1) |
			((read_imagei(rawData, sampler, pos + cubeOffsets[3]).x > isolevel) << 2) |
			((read_imagei(rawData, sampler, pos + cubeOffsets[2]).x > isolevel) << 3) |
			((read_imagei(rawData, sampler, pos + cubeOffsets[4]).x > isolevel) << 4) |
			((read_imagei(rawData, sampler, pos + cubeOffsets[5]).x > isolevel) << 5) |
			((read_imagei(rawData, sampler, pos + cubeOffsets[7]).x > isolevel) << 6) |
			((read_imagei(rawData, sampler, pos + cubeOffsets[6]).x > isolevel) << 7);

// Store number of triangles and index
	size_t writePos = zCurveGridIndexAtCoord(pos.x, pos.y, pos.z);
	histoPyramid[writePos] = nrOfTriangles[cubeindex];
	cubeIndexes[pos.x + pos.y * get_global_size(0) +
	            pos.z * get_global_size(0) * get_global_size(1)] = cubeindex;
}