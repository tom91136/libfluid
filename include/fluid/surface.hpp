#ifndef LIBFLUID_SURFACE_H
#define LIBFLUID_SURFACE_H

// #define GLM_ENABLE_EXPERIMENTAL

#include <ostream>
#include <functional>
#include <vector>
#include <numeric>
#include <map>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <array>

#include "geometry.hpp"
#include "mctable.hpp"

namespace surface {

	using mctable::EdgeTable;
	using mctable::TriTable;
	using geometry::MeshTriangle;
	using glm::tvec3;

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

	template<typename N>
	struct GridCell {
		std::array<tvec3<N>, 8> p;
		std::array<N, 8> val;

		explicit GridCell(const std::array<tvec3<N>, 8> &p, const std::array<N, 8> &val) :
				p(p), val(val) {}
	};

	template<typename N>
	inline tvec3<N> lerp(N isolevel, tvec3<N> p1, tvec3<N> p2, N v1, N v2) {
		if (std::abs(isolevel - v1) < 0.00001) return p1;
		if (std::abs(isolevel - v2) < 0.00001) return p2;
		if (std::abs(v1 - v2) < 0.00001) return p1;
		return p1 + (p2 - p1) * ((isolevel - v1) / (v2 - v1));
	}

//	template<typename N>
//	inline tvec3<N> lerp(tvec3<N> p1, tvec3<N> p2, N v1, N v2) {
//		return lerp()
//		return glm::lerp(p1, p2, v1 / (v2 - v1));
//	}

	//http://paulbourke.net/geometry/polygonise/
	template<typename N>
	void polygonise(const GridCell<N> &g, const N isolevel, std::vector<MeshTriangle<N>> &sink) {

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
		if (EdgeTable[ci] == 0) std::vector<MeshTriangle<N>>();

		std::array<tvec3<N>, 12> vs;
		/* Find the vertices where the surface intersects the cube */
		if (EdgeTable[ci] & 1 << 0) vs[0] = lerp(isolevel, g.p[0], g.p[1], g.val[0], g.val[1]);
		if (EdgeTable[ci] & 1 << 1) vs[1] = lerp(isolevel, g.p[1], g.p[2], g.val[1], g.val[2]);
		if (EdgeTable[ci] & 1 << 2) vs[2] = lerp(isolevel, g.p[2], g.p[3], g.val[2], g.val[3]);
		if (EdgeTable[ci] & 1 << 3) vs[3] = lerp(isolevel, g.p[3], g.p[0], g.val[3], g.val[0]);
		if (EdgeTable[ci] & 1 << 4) vs[4] = lerp(isolevel, g.p[4], g.p[5], g.val[4], g.val[5]);
		if (EdgeTable[ci] & 1 << 5) vs[5] = lerp(isolevel, g.p[5], g.p[6], g.val[5], g.val[6]);
		if (EdgeTable[ci] & 1 << 6) vs[6] = lerp(isolevel, g.p[6], g.p[7], g.val[6], g.val[7]);
		if (EdgeTable[ci] & 1 << 7) vs[7] = lerp(isolevel, g.p[7], g.p[4], g.val[7], g.val[4]);
		if (EdgeTable[ci] & 1 << 8) vs[8] = lerp(isolevel, g.p[0], g.p[4], g.val[0], g.val[4]);
		if (EdgeTable[ci] & 1 << 9) vs[9] = lerp(isolevel, g.p[1], g.p[5], g.val[1], g.val[5]);
		if (EdgeTable[ci] & 1 << 10) vs[10] = lerp(isolevel, g.p[2], g.p[6], g.val[2], g.val[6]);
		if (EdgeTable[ci] & 1 << 11) vs[11] = lerp(isolevel, g.p[3], g.p[7], g.val[3], g.val[7]);

		for (size_t i = 0; TriTable[ci][i] != -1; i += 3)
			sink.emplace_back(
					vs[TriTable[ci][i]],
					vs[TriTable[ci][i + 1]],
					vs[TriTable[ci][i + 2]]);
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

		inline size_t xSize() const { return d1; }

		inline size_t ySize() const { return d2; }

		inline size_t zSize() const { return d3; }

		inline size_t size() const { return data.size(); }

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
							const std::array<N, 8> &vertices,
							const std::array<tvec3<N>, 8> &normals,
							const std::array<tvec3<N>, 8> &pos,
							std::vector<MeshTriangle<N>> &triangles) {


		size_t ci = 0;
		ci = (vertices[0] < isolevel) ? ci | (1 << 0) : ci;
		ci = (vertices[1] < isolevel) ? ci | (1 << 1) : ci;
		ci = (vertices[2] < isolevel) ? ci | (1 << 2) : ci;
		ci = (vertices[3] < isolevel) ? ci | (1 << 3) : ci;
		ci = (vertices[4] < isolevel) ? ci | (1 << 4) : ci;
		ci = (vertices[5] < isolevel) ? ci | (1 << 5) : ci;
		ci = (vertices[6] < isolevel) ? ci | (1 << 6) : ci;
		ci = (vertices[7] < isolevel) ? ci | (1 << 7) : ci;

		/* Cube is entirely in/out of the surface */
		if (EdgeTable[ci] == 0) return;

		std::array<tvec3<N>, 12> ts;
		std::array<tvec3<N>, 12> ns;

		/* Find the vertices where the surface intersects the cube */
		if (EdgeTable[ci] & 1 << 0) {
			ts[0] = lerp(isolevel, pos[0], pos[1], vertices[0], vertices[1]);
			ns[0] = lerp(isolevel, normals[0], normals[1], vertices[0], vertices[1]);
		}
		if (EdgeTable[ci] & 1 << 1) {
			ts[1] = lerp(isolevel, pos[1], pos[2], vertices[1], vertices[2]);
			ns[1] = lerp(isolevel, normals[1], normals[2], vertices[1], vertices[2]);
		}
		if (EdgeTable[ci] & 1 << 2) {
			ts[2] = lerp(isolevel, pos[2], pos[3], vertices[2], vertices[3]);
			ns[2] = lerp(isolevel, normals[2], normals[3], vertices[2], vertices[3]);
		}
		if (EdgeTable[ci] & 1 << 3) {
			ts[3] = lerp(isolevel, pos[3], pos[0], vertices[3], vertices[0]);
			ns[3] = lerp(isolevel, normals[3], normals[0], vertices[3], vertices[0]);
		}
		if (EdgeTable[ci] & 1 << 4) {
			ts[4] = lerp(isolevel, pos[4], pos[5], vertices[4], vertices[5]);
			ns[4] = lerp(isolevel, normals[4], normals[5], vertices[4], vertices[5]);
		}
		if (EdgeTable[ci] & 1 << 5) {
			ts[5] = lerp(isolevel, pos[5], pos[6], vertices[5], vertices[6]);
			ns[5] = lerp(isolevel, normals[5], normals[6], vertices[5], vertices[6]);
		}
		if (EdgeTable[ci] & 1 << 6) {
			ts[6] = lerp(isolevel, pos[6], pos[7], vertices[6], vertices[7]);
			ns[6] = lerp(isolevel, normals[6], normals[7], vertices[6], vertices[7]);
		}
		if (EdgeTable[ci] & 1 << 7) {
			ts[7] = lerp(isolevel, pos[7], pos[4], vertices[7], vertices[4]);
			ns[7] = lerp(isolevel, normals[7], normals[4], vertices[7], vertices[4]);
		}
		if (EdgeTable[ci] & 1 << 8) {
			ts[8] = lerp(isolevel, pos[0], pos[4], vertices[0], vertices[4]);
			ns[8] = lerp(isolevel, normals[0], normals[4], vertices[0], vertices[4]);
		}
		if (EdgeTable[ci] & 1 << 9) {
			ts[9] = lerp(isolevel, pos[1], pos[5], vertices[1], vertices[5]);
			ns[9] = lerp(isolevel, normals[1], normals[5], vertices[1], vertices[5]);
		}
		if (EdgeTable[ci] & 1 << 10) {
			ts[10] = lerp(isolevel, pos[2], pos[6], vertices[2], vertices[6]);
			ns[10] = lerp(isolevel, normals[2], normals[6], vertices[2], vertices[6]);
		}
		if (EdgeTable[ci] & 1 << 11) {
			ts[11] = lerp(isolevel, pos[3], pos[7], vertices[3], vertices[7]);
			ns[11] = lerp(isolevel, normals[3], normals[7], vertices[3], vertices[7]);
		}

		for (size_t i = 0; TriTable[ci][i] != -1; i += 3) {
			int x = TriTable[ci][i + 0];
			int y = TriTable[ci][i + 1];
			int z = TriTable[ci][i + 2];
			triangles.emplace_back(ts[x], ts[y], ts[z], ns[x], ns[y], ns[z]);
		}

	}


	template<typename N>
	std::vector<MeshTriangle<N>> parameterise(const N isolevel,
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


		std::vector<MeshTriangle<N>> triangles;
		std::vector<tvec3<N>> points;


#ifndef _MSC_VER
#pragma omp declare reduction (merge : std::vector<MeshTriangle<N>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
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
					if (EdgeTable[ci] == 0) std::vector<MeshTriangle<N>>();

					std::array<tvec3<N>, 12> xs;
					/* Find the vertices where the surface intersects the cube */
					if (EdgeTable[ci] & 1 << 0) xs[0] = lerp(isolevel, vs[0], vs[1], ns[0], ns[1]);
					if (EdgeTable[ci] & 1 << 1) xs[1] = lerp(isolevel, vs[1], vs[2], ns[1], ns[2]);
					if (EdgeTable[ci] & 1 << 2) xs[2] = lerp(isolevel, vs[2], vs[3], ns[2], ns[3]);
					if (EdgeTable[ci] & 1 << 3) xs[3] = lerp(isolevel, vs[3], vs[0], ns[3], ns[0]);
					if (EdgeTable[ci] & 1 << 4) xs[4] = lerp(isolevel, vs[4], vs[5], ns[4], ns[5]);
					if (EdgeTable[ci] & 1 << 5) xs[5] = lerp(isolevel, vs[5], vs[6], ns[5], ns[6]);
					if (EdgeTable[ci] & 1 << 6) xs[6] = lerp(isolevel, vs[6], vs[7], ns[6], ns[7]);
					if (EdgeTable[ci] & 1 << 7) xs[7] = lerp(isolevel, vs[7], vs[4], ns[7], ns[4]);
					if (EdgeTable[ci] & 1 << 8) xs[8] = lerp(isolevel, vs[0], vs[4], ns[0], ns[4]);
					if (EdgeTable[ci] & 1 << 9) xs[9] = lerp(isolevel, vs[1], vs[5], ns[1], ns[5]);
					if (EdgeTable[ci] & 1 << 10)
						xs[10] = lerp(isolevel, vs[2], vs[6], ns[2], ns[6]);
					if (EdgeTable[ci] & 1 << 11)
						xs[11] = lerp(isolevel, vs[3], vs[7], ns[3], ns[7]);

					for (size_t i = 0; TriTable[ci][i] != -1; i += 3) {
//						points.emplace_back(xs[triTable[ci][i + 0]]);
//						points.emplace_back(xs[triTable[ci][i + 1]]);
//						points.emplace_back(xs[triTable[ci][i + 2]]);

						triangles.emplace_back(
								xs[TriTable[ci][i]],
								xs[TriTable[ci][i + 1]],
								xs[TriTable[ci][i + 2]]);
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