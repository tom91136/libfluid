#ifndef LIBFLUID_SURFACE_H
#define LIBFLUID_SURFACE_H

#define GLM_ENABLE_EXPERIMENTAL

#include <ostream>
#include <functional>
#include <vector>
#include <numeric>
#include <map>
#include <iostream>
#include <algorithm>
#include "glm/ext.hpp"
#include "glm/glm.hpp"
#include <cstdlib>
#include <memory>
#include <array>


namespace surface {

	const static std::array<glm::tvec3<size_t>, 8> CUBE_OFFSETS = {
			glm::tvec3<size_t>(0, 0, 0),
			glm::tvec3<size_t>(1, 0, 0),
			glm::tvec3<size_t>(1, 1, 0),
			glm::tvec3<size_t>(0, 1, 0),
			glm::tvec3<size_t>(0, 0, 1),
			glm::tvec3<size_t>(1, 0, 1),
			glm::tvec3<size_t>(1, 1, 1),
			glm::tvec3<size_t>(0, 1, 1)
	};

	//<editor-fold>
	const static std::array<int, 256> edgeTable = {
			0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
			0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
			0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
			0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
			0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
			0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
			0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
			0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
			0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
			0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
			0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
			0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
			0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
			0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
			0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
			0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
			0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
			0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
			0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
			0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
			0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
			0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
			0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
			0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
			0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
			0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
			0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
			0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
			0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
			0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
			0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
			0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
	};
	//</editor-fold>

	//<editor-fold>
	const static std::array<std::array<int, 16>, 256> triTable =
			{{
					 {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
					 {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
					 {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
					 {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
					 {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
					 {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
					 {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
					 {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
					 {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
					 {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
					 {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
					 {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
					 {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
					 {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
					 {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
					 {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
					 {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
					 {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
					 {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
					 {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
					 {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
					 {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
					 {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
					 {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
					 {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
					 {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
					 {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
					 {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
					 {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
					 {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
					 {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
					 {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
					 {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
					 {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
					 {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
					 {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
					 {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
					 {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
					 {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
					 {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
					 {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
					 {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
					 {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
					 {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
					 {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
					 {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
					 {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
					 {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
					 {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
					 {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
					 {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
					 {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
					 {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
					 {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
					 {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
					 {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
					 {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
					 {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
					 {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
					 {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
					 {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
					 {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
					 {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
					 {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
					 {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
					 {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
					 {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
					 {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
					 {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
					 {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
					 {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
					 {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
					 {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
					 {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
					 {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
					 {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
					 {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
					 {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
					 {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
					 {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
					 {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
					 {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
					 {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
					 {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
					 {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
					 {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
					 {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
					 {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
					 {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
					 {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
					 {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
					 {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
					 {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
					 {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
					 {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
					 {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
					 {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
					 {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
					 {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
					 {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
					 {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
					 {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
					 {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
					 {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
					 {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
					 {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
					 {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
					 {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
					 {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
					 {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
					 {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
					 {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
					 {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
					 {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
					 {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
					 {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
					 {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
					 {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
					 {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
					 {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
					 {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
					 {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
					 {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
					 {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
					 {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
					 {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
					 {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
					 {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
					 {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
					 {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
					 {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
					 {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
					 {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
					 {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
					 {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
					 {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
					 {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
					 {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
					 {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
					 {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
					 {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
					 {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
					 {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
					 {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
					 {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
					 {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
					 {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
					 {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
					 {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
					 {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
					 {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
					 {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
					 {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
					 {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
					 {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
					 {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
					 {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
					 {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
					 {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
					 {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
					 {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
					 {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
					 {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
					 {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
					 {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
					 {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
					 {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
					 {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
					 {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
					 {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
					 {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
					 {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
					 {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
					 {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
					 {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
					 {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
					 {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
					 {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
					 {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
					 {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
					 {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
					 {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
					 {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
					 {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
					 {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
					 {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
					 {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
					 {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
					 {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
					 {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
			 }};
	//</editor-fold>



	using glm::tvec3;

	template<typename N>
	struct Triangle {
		tvec3<N> v0;
		tvec3<N> v1;
		tvec3<N> v2;
		tvec3<N> normal;

		explicit Triangle(const tvec3<N> &v0, const tvec3<N> &v1, const tvec3<N> &v2) :
				v0(v0), v1(v1), v2(v2) {}

		friend std::ostream &operator<<(std::ostream &os, const Triangle &triangle) {
			os << "T("
			   << glm::to_string(triangle.v0) << ","
			   << glm::to_string(triangle.v1) << ","
			   << glm::to_string(triangle.v2)
			   << ")";
			return os;
		}

	};

	template<typename N>
	struct Cell {
		std::array<tvec3<N>, 8> p;

		explicit Cell(const std::array<tvec3<N>, 8> &p) : p(p) {}
	};


	//https://github.com/opengl-tutorials/ogl/blob/master/common/vboindexer.cpp
	template<typename N>
	struct PackedVertex {
		tvec3<N> position;

		bool operator<(const PackedVertex that) const {
			return memcmp((void *) this, (void *) &that, sizeof(PackedVertex)) > 0;
		};
	};

	template<typename N>
	bool getSimilarVertexIndex_fast(
			PackedVertex<N> &packed,
			std::map<PackedVertex<N>, unsigned short> &VertexToOutIndex,
			unsigned short &result
	) {
		typename std::map<PackedVertex<N>, unsigned short>::iterator it = VertexToOutIndex.find(
				packed);
		if (it == VertexToOutIndex.end()) {
			return false;
		} else {
			result = it->second;
			return true;
		}
	}

	template<typename N>
	void indexVBO(
			std::vector<tvec3<N>> &in_vertices,
			std::vector<unsigned short> &out_indices,
			std::vector<tvec3<N>> &out_vertices
	) {
		std::map<PackedVertex<N>, unsigned short> VertexToOutIndex;
		// For each input vertex
		for (size_t i = 0; i < in_vertices.size(); i++) {
			PackedVertex<N> packed = {in_vertices[i]};
			// Try to find a similar vertex in out_XXXX
			unsigned short index;
			bool found = getSimilarVertexIndex_fast(packed, VertexToOutIndex, index);
			if (found) { // A similar vertex is already in the VBO, use it instead !
				out_indices.push_back(index);
			} else { // If not, it needs to be added in the output data.
				out_vertices.push_back(in_vertices[i]);
				unsigned short newindex = (unsigned short) out_vertices.size() - 1;
				out_indices.push_back(newindex);
				VertexToOutIndex[packed] = newindex;
			}
		}
	}


	template<typename N>
	struct GridCell {
		std::array<tvec3<N>, 8> p;
		std::array<N, 8> val;

		explicit GridCell(const std::array<tvec3<N>, 8> &p, const std::array<N, 8> &val) :
				p(p), val(val) {}
	};

	template<typename N>
	tvec3<N> lerp(N isolevel, tvec3<N> p1, tvec3<N> p2, N v1, N v2) {
		if (std::abs(isolevel - v1) < 0.00001) return p1;
		if (std::abs(isolevel - v2) < 0.00001) return p2;
		if (std::abs(v1 - v2) < 0.00001) return p1;
		return p1 + (p2 - p1) * ((isolevel - v1) / (v2 - v1));
	}

	//http://paulbourke.net/geometry/polygonise/
	template<typename N>
	void polygonise(const GridCell<N> &g, const N isolevel, std::vector<Triangle<N>> &sink) {

		size_t ci = 0;
		if (g.val[0] < isolevel) ci |= 1;
		if (g.val[1] < isolevel) ci |= 2;
		if (g.val[2] < isolevel) ci |= 4;
		if (g.val[3] < isolevel) ci |= 8;
		if (g.val[4] < isolevel) ci |= 16;
		if (g.val[5] < isolevel) ci |= 32;
		if (g.val[6] < isolevel) ci |= 64;
		if (g.val[7] < isolevel) ci |= 128;

		/* Cube is entirely in/out of the surface */
		if (edgeTable[ci] == 0) std::vector<Triangle<N>>();

		std::array<tvec3<N>, 12> vs;
		/* Find the vertices where the surface intersects the cube */
		if (edgeTable[ci] & 1 << 0) vs[0] = lerp(isolevel, g.p[0], g.p[1], g.val[0], g.val[1]);
		if (edgeTable[ci] & 1 << 1) vs[1] = lerp(isolevel, g.p[1], g.p[2], g.val[1], g.val[2]);
		if (edgeTable[ci] & 1 << 2) vs[2] = lerp(isolevel, g.p[2], g.p[3], g.val[2], g.val[3]);
		if (edgeTable[ci] & 1 << 3) vs[3] = lerp(isolevel, g.p[3], g.p[0], g.val[3], g.val[0]);
		if (edgeTable[ci] & 1 << 4) vs[4] = lerp(isolevel, g.p[4], g.p[5], g.val[4], g.val[5]);
		if (edgeTable[ci] & 1 << 5) vs[5] = lerp(isolevel, g.p[5], g.p[6], g.val[5], g.val[6]);
		if (edgeTable[ci] & 1 << 6) vs[6] = lerp(isolevel, g.p[6], g.p[7], g.val[6], g.val[7]);
		if (edgeTable[ci] & 1 << 7) vs[7] = lerp(isolevel, g.p[7], g.p[4], g.val[7], g.val[4]);
		if (edgeTable[ci] & 1 << 8) vs[8] = lerp(isolevel, g.p[0], g.p[4], g.val[0], g.val[4]);
		if (edgeTable[ci] & 1 << 9) vs[9] = lerp(isolevel, g.p[1], g.p[5], g.val[1], g.val[5]);
		if (edgeTable[ci] & 1 << 10) vs[10] = lerp(isolevel, g.p[2], g.p[6], g.val[2], g.val[6]);
		if (edgeTable[ci] & 1 << 11) vs[11] = lerp(isolevel, g.p[3], g.p[7], g.val[3], g.val[7]);

		for (size_t i = 0; triTable[ci][i] != -1; i += 3)
			sink.emplace_back(
					vs[triTable[ci][i]],
					vs[triTable[ci][i + 1]],
					vs[triTable[ci][i + 2]]);
	}


	template<typename T>
	class Lattice {

		const size_t d1, d2, d3;
		std::vector<T> data;

	public:
		explicit Lattice(
				const size_t d1 = 0,
				const size_t d2 = 0,
				const size_t d3 = 0, T const &t = T()) :
				d1(d1), d2(d2), d3(d3), data(d1 * d2 * d3, t) {}


		T &operator()(size_t i, size_t j, size_t k) {
			return data[i * d2 * d3 + j * d3 + k];
		}

		T const &operator()(size_t i, size_t j, size_t k) const {
			return data[i * d2 * d3 + j * d3 + k];
		}

		size_t xSize() const { return d1; }

		size_t ySize() const { return d2; }

		size_t zSize() const { return d3; }

		size_t size() const { return data.size(); }

		decltype(auto) begin() { return data.begin(); }

		decltype(auto) end() { return data.end(); }

	};


	template<typename N>
	using MCLattice = Lattice<tvec3<N> >;

	template<typename N>
	MCLattice<N> createLattice(
			const size_t xSize,
			const size_t ySize,
			const size_t zSize,
			const N offset, const N factor) {

		MCLattice<N> lattice =
				MCLattice<N>(xSize, ySize, zSize, tvec3<N>(0));

		for (size_t x = 0; x < xSize; ++x) {
			for (size_t y = 0; y < ySize; ++y) {
				for (size_t z = 0; z < zSize; ++z) {
					auto vec = tvec3<N>(
							static_cast<N>(x) * factor + offset,
							static_cast<N>(y) * factor + offset,
							static_cast<N>(z) * factor + offset);
					lattice(x, y, z) = vec;
				}
			}
		}
		return lattice;
	}


	template<typename N>
	inline void marchSingle(const N isolevel,
	                        const std::array<N, 8> &ns, const std::array<tvec3<N>, 8> &vs,
	                        std::vector<Triangle<N>> &triangles) {


		size_t ci = 0;
		if (ns[0] < isolevel) ci |= 1;
		if (ns[1] < isolevel) ci |= 2;
		if (ns[2] < isolevel) ci |= 4;
		if (ns[3] < isolevel) ci |= 8;
		if (ns[4] < isolevel) ci |= 16;
		if (ns[5] < isolevel) ci |= 32;
		if (ns[6] < isolevel) ci |= 64;
		if (ns[7] < isolevel) ci |= 128;

		/* Cube is entirely in/out of the surface */
		if (edgeTable[ci] == 0) return;

		std::array<tvec3<N>, 12> ts;
		/* Find the vertices where the surface intersects the cube */
		if (edgeTable[ci] & 1 << 0) ts[0] = lerp(isolevel, vs[0], vs[1], ns[0], ns[1]);
		if (edgeTable[ci] & 1 << 1) ts[1] = lerp(isolevel, vs[1], vs[2], ns[1], ns[2]);
		if (edgeTable[ci] & 1 << 2) ts[2] = lerp(isolevel, vs[2], vs[3], ns[2], ns[3]);
		if (edgeTable[ci] & 1 << 3) ts[3] = lerp(isolevel, vs[3], vs[0], ns[3], ns[0]);
		if (edgeTable[ci] & 1 << 4) ts[4] = lerp(isolevel, vs[4], vs[5], ns[4], ns[5]);
		if (edgeTable[ci] & 1 << 5) ts[5] = lerp(isolevel, vs[5], vs[6], ns[5], ns[6]);
		if (edgeTable[ci] & 1 << 6) ts[6] = lerp(isolevel, vs[6], vs[7], ns[6], ns[7]);
		if (edgeTable[ci] & 1 << 7) ts[7] = lerp(isolevel, vs[7], vs[4], ns[7], ns[4]);
		if (edgeTable[ci] & 1 << 8) ts[8] = lerp(isolevel, vs[0], vs[4], ns[0], ns[4]);
		if (edgeTable[ci] & 1 << 9) ts[9] = lerp(isolevel, vs[1], vs[5], ns[1], ns[5]);
		if (edgeTable[ci] & 1 << 10) ts[10] = lerp(isolevel, vs[2], vs[6], ns[2], ns[6]);
		if (edgeTable[ci] & 1 << 11) ts[11] = lerp(isolevel, vs[3], vs[7], ns[3], ns[7]);

		for (size_t i = 0; triTable[ci][i] != -1; i += 3) {
			triangles.emplace_back(
					ts[triTable[ci][i]],
					ts[triTable[ci][i + 1]],
					ts[triTable[ci][i + 2]]);
		}

	}


	template<typename N>
	std::vector<Triangle<N>> parameterise(const N isolevel,
	                                      MCLattice<N> lattice,
	                                      const std::function<N(tvec3<N> &)> &f) {

		Lattice<N> field = Lattice<N>(lattice.xSize(), lattice.ySize(), lattice.zSize(), 0);

#pragma omp parallel for
		for (int x = 0; x < lattice.xSize(); ++x) {
			for (int y = 0; y < lattice.ySize(); ++y) {
				for (int z = 0; z < lattice.zSize(); ++z) {
					field(x, y, z) = f(lattice(x, y, z));
				}
			}
		}

		const static std::array<std::tuple<size_t, size_t, size_t>, 8> &verticies = {
				std::make_tuple(0, 0, 0),
				std::make_tuple(1, 0, 0),
				std::make_tuple(1, 1, 0),
				std::make_tuple(0, 1, 0),
				std::make_tuple(0, 0, 1),
				std::make_tuple(1, 0, 1),
				std::make_tuple(1, 1, 1),
				std::make_tuple(0, 1, 1),
		};


		std::vector<Triangle<N>> triangles;
		std::vector<tvec3<N>> points;


#ifndef _MSC_VER
#pragma omp declare reduction (merge : std::vector<Triangle<N>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for collapse(3) reduction(merge: triangles)
#endif
		for (size_t x = 0; x < lattice.xSize() - 1; ++x) {
			for (size_t y = 0; y < lattice.ySize() - 1; ++y) {
				for (size_t z = 0; z < lattice.zSize() - 1; ++z) {

					std::array<N, 8> ns{};
					std::array<tvec3<N>, 8> vs{};

					for (size_t j = 0; j < 8; ++j) {

						size_t ox = x + std::get<0>(verticies[j]);
						size_t oy = y + std::get<1>(verticies[j]);
						size_t oz = z + std::get<2>(verticies[j]);

						vs[j] = lattice(ox, oy, oz);
						ns[j] = field(ox, oy, oz);

					}

					size_t ci = 0;
					if (ns[0] < isolevel) ci |= 1;
					if (ns[1] < isolevel) ci |= 2;
					if (ns[2] < isolevel) ci |= 4;
					if (ns[3] < isolevel) ci |= 8;
					if (ns[4] < isolevel) ci |= 16;
					if (ns[5] < isolevel) ci |= 32;
					if (ns[6] < isolevel) ci |= 64;
					if (ns[7] < isolevel) ci |= 128;

					/* Cube is entirely in/out of the surface */
					if (edgeTable[ci] == 0) std::vector<Triangle<N>>();

					std::array<tvec3<N>, 12> xs;
					/* Find the vertices where the surface intersects the cube */
					if (edgeTable[ci] & 1 << 0) xs[0] = lerp(isolevel, vs[0], vs[1], ns[0], ns[1]);
					if (edgeTable[ci] & 1 << 1) xs[1] = lerp(isolevel, vs[1], vs[2], ns[1], ns[2]);
					if (edgeTable[ci] & 1 << 2) xs[2] = lerp(isolevel, vs[2], vs[3], ns[2], ns[3]);
					if (edgeTable[ci] & 1 << 3) xs[3] = lerp(isolevel, vs[3], vs[0], ns[3], ns[0]);
					if (edgeTable[ci] & 1 << 4) xs[4] = lerp(isolevel, vs[4], vs[5], ns[4], ns[5]);
					if (edgeTable[ci] & 1 << 5) xs[5] = lerp(isolevel, vs[5], vs[6], ns[5], ns[6]);
					if (edgeTable[ci] & 1 << 6) xs[6] = lerp(isolevel, vs[6], vs[7], ns[6], ns[7]);
					if (edgeTable[ci] & 1 << 7) xs[7] = lerp(isolevel, vs[7], vs[4], ns[7], ns[4]);
					if (edgeTable[ci] & 1 << 8) xs[8] = lerp(isolevel, vs[0], vs[4], ns[0], ns[4]);
					if (edgeTable[ci] & 1 << 9) xs[9] = lerp(isolevel, vs[1], vs[5], ns[1], ns[5]);
					if (edgeTable[ci] & 1 << 10)
						xs[10] = lerp(isolevel, vs[2], vs[6], ns[2], ns[6]);
					if (edgeTable[ci] & 1 << 11)
						xs[11] = lerp(isolevel, vs[3], vs[7], ns[3], ns[7]);

					for (size_t i = 0; triTable[ci][i] != -1; i += 3) {
//						points.emplace_back(xs[triTable[ci][i + 0]]);
//						points.emplace_back(xs[triTable[ci][i + 1]]);
//						points.emplace_back(xs[triTable[ci][i + 2]]);

						triangles.emplace_back(
								xs[triTable[ci][i]],
								xs[triTable[ci][i + 1]],
								xs[triTable[ci][i + 2]]);
					}


				}
			}
		}

//		using hrc = std::chrono::high_resolution_clock;
//		hrc::time_point mmt1 = hrc::now();
//
//		std::vector<unsigned short> out_indices;
//		std::vector<tvec3<N>> out_vertices;
//		indexVBO(points, out_indices, out_vertices);
//
//		hrc::time_point mmt2 = hrc::now();
//		auto solve = std::chrono::duration_cast<std::chrono::nanoseconds>(mmt2 - mmt1).count();
//
//		std::cout << triangles.size() * 3 << " " << out_indices.size() << " " << out_vertices.size()
//				  << " " << (solve / 1000000.0) << "ms" << std::endl;


		return triangles;
	}


}


#endif