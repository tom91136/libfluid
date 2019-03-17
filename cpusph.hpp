#ifndef LIBFLUID_CPUSPH_HPP
#define LIBFLUID_CPUSPH_HPP

#define _GLIBCXX_PARALLEL

#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_SIMD_AVX2
#define GLM_ENABLE_EXPERIMENTAL

#include <ostream>
#include <functional>
#include <vector>
#include <numeric>
#include <chrono>
#include <unordered_map>
#include <iostream>
#include <algorithm>
#include <limits>
#include "glm/ext.hpp"
#include "glm/glm.hpp"
#include "Octree.hpp"
#include "nanoflann.hpp"
#include <cstdlib>
#include <memory>
#include "clsph.hpp"
#include "fluid.hpp"
#include "ska_sort.hpp"


namespace cpusph {

	using fluid::Particle;
	using fluid::Atom;
	using fluid::Response;
	using fluid::Ray;


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

	const static glm::tvec3<size_t> get_cell_coords_z_curve(size_t index) {
		size_t mask = 0x9249249;
		size_t i_x = index & mask;
		size_t i_y = (index >> 1) & mask;
		size_t i_z = (index >> 2) & mask;
		return glm::tvec3<size_t>(uninterleave(i_x), uninterleave(i_y), uninterleave(i_z));
	}

	const static size_t get_grid_index_z_curve(size_t in_x, size_t in_y, size_t in_z) {
		size_t x = in_x;
		size_t y = in_y;
		size_t z = in_z;

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

		return x | (y << 1) | (z << 2);
	}


	template<typename T, typename N>
	const static void allPairNN(std::vector<Atom<T, N>> &atoms) {


		const float H2 = 0.1 /2;

		float min_x, max_x, min_y, max_y, min_z, max_z;
		float grid_cell_side_length = (H2 * 2);
		min_x = min_y = min_z = std::numeric_limits<cl_int>::max();
		max_x = max_y = max_z = std::numeric_limits<cl_int>::min();

		for (int i = 0; i < atoms.size(); ++i) {
			Atom<T, N> &p = atoms[i];
			auto pos = p.now;
			if (pos.x < min_x) min_x = pos.x;
			if (pos.y < min_y) min_y = pos.y;
			if (pos.z < min_z) min_z = pos.z;

			if (pos.x > max_x) max_x = pos.x;
			if (pos.y > max_y) max_y = pos.y;
			if (pos.z > max_z) max_z = pos.z;
		}

		min_x -= grid_cell_side_length * 2;
		min_y -= grid_cell_side_length * 2;
		min_z -= grid_cell_side_length * 2;

		max_x += grid_cell_side_length * 2;
		max_y += grid_cell_side_length * 2;
		max_z += grid_cell_side_length * 2;

		glm::tvec3<N> minP(min_x, min_y, min_z);
		glm::tvec3<N> maxP(max_x, max_y, max_z);

		glm::tvec3<size_t> sizes(
				static_cast<size_t>((max_x - min_x) / grid_cell_side_length),
				static_cast<size_t>((max_y - min_y) / grid_cell_side_length),
				static_cast<size_t>((max_z - min_z) / grid_cell_side_length)
		);

		size_t count = get_grid_index_z_curve(
				sizes.x, sizes.y, sizes.z);

#pragma omp parallel for
		for (int i = 0; i < atoms.size(); ++i) {
			Atom<T, N> &p = atoms[i];


			size_t grid_index = get_grid_index_z_curve(
					static_cast<size_t>((p.now.x - min_x) / (grid_cell_side_length)),
					static_cast<size_t>((p.now.y - min_y) / (grid_cell_side_length)),
					static_cast<size_t>((p.now.z - min_z) / (grid_cell_side_length)));


			p.zIndex = grid_index; // get_grid_index_z_curve(p.now.x, p.now.y, p.now.z);
		}


//		for (const Atom<T, N> &a : atoms) {
//			std::cout << "CPU1 >> " << " t=" << a.particle->t << " ZIdx=" << a.zIndex << std::endl;
//		}


//		ska_sort(atoms.begin(), atoms.end(), [](const Atom<T, N> &a) ->  { return   a.zIndex; });
		std::sort(atoms.begin(), atoms.end(), [](const Atom<T, N> &lhs, const Atom<T, N> &rhs) {
			return lhs.zIndex < rhs.zIndex;
		});

//		std::cout << " >> " << atoms.size() << std::endl;

//		for (const Atom<T, N> &a : atoms) {
//			std::cout << "CPU >> " << " t=" << a.particle->t << " ZIdx=" << a.zIndex << std::endl;
//		}

		std::vector<size_t> ses(count);
		size_t current_index = 0;
		for (size_t i = 0; i < count; ++i) {
			ses[i] = current_index;
//			std::cout << "\t-> " << i << "  =" << current_index << std::endl;
			while (current_index != atoms.size() && atoms[current_index].zIndex == i) {
				current_index++;
			}
		}


#pragma omp parallel for
		for (size_t xx = 0; xx < atoms.size(); ++xx) {
			Atom<T, N> &p = atoms[xx];


			auto cell_coords = get_cell_coords_z_curve(p.zIndex);

			for (size_t z = cell_coords.z - 1; z <= cell_coords.z + 1; ++z) {
				for (size_t y = cell_coords.y - 1; y <= cell_coords.y + 1; ++y) {
					for (size_t x = cell_coords.x - 1; x <= cell_coords.x + 1; ++x) {
						size_t grid_index = get_grid_index_z_curve(x, y, z);


						size_t start = ses[grid_index];
						size_t end = (count > (grid_index + 1))
						             ? ses[grid_index + 1]
						             : atoms.size();


//						std::cout << "\t\t<ZI> " << grid_index << "  [" << start << "->" << end
//						          << "] @  "
//						          << glm::to_string(glm::tvec3<size_t>(x, y, z)) << std::endl;

						for (size_t ni = start; ni < end; ++ni) {
							p.neighbours->emplace_back(&atoms[ni]);
						}
					}
				}
			}

//			std::cout << "\t[ZI] " << p.particle->t << " N=" << p.neighbours->size() << " @ "
//			          << glm::to_string(cell_coords) << std::endl;

			p.p6ks->reserve(p.neighbours->size());
			p.skgs->reserve(p.neighbours->size());


		}


		std::cout << "\t**ZI "
		          << " side length=" << grid_cell_side_length
		          << " [CI]=" << current_index
		          << " [CL]=" << ses.size()
		          << " count=" << count << "->"
		          << glm::to_string(get_cell_coords_z_curve(count))
		          << " X*Y*Z=" << glm::to_string(sizes)
		          << " min=" << glm::to_string(minP)
		          << " max=" << glm::to_string(maxP)
		          << std::endl;
		std::cout << " Atom size >>> " << atoms.size() << std::endl;

	}


//	template<typename T, typename N>
//	class GridNN {
//
//
//		glm::tvec3<N> pMin;
//		glm::tvec3<N> pMax;
//		glm::tvec3<int> cellMax;
//		std::vector<glm::tvec3<N>> offsets = std::vector<glm::tvec3<N>>();
//
//		const std::vector<Atom<T, N>> &xs;
//
//	public:
//
//
//		static constexpr auto CELL_SIZE = 0.1;
//
//		struct AtomCompare {
//			bool operator()(const glm::tvec3<int> &lhs, const glm::tvec3<int> &rhs) const {
//				return memcmp((void *) &lhs, (void *) &rhs, sizeof(glm::tvec3<int>)) > 0;
//			}
//		};
//
//
//		typedef std::unordered_multimap<size_t, size_t> SMap;
//		SMap M;
//
//
//		size_t hash(glm::tvec3<N> a) {
//			size_t res = 17;
//			res = res * 31 + std::hash<int>()(round(a.x / CELL_SIZE));
//			res = res * 31 + std::hash<int>()(round(a.y / CELL_SIZE));
//			res = res * 31 + std::hash<int>()(round(a.z / CELL_SIZE));
//			return res;
//		}
//
//
//		GridNN(std::vector<Atom<T, N>> &xs) : xs(xs) {
//
//
//			float min_x, max_x, min_y, max_y, min_z, max_z;
//			float grid_cell_side_length = (0.1f * 2);
//			min_x = min_y = min_z = std::numeric_limits<cl_int>::max();
//			max_x = max_y = max_z = std::numeric_limits<cl_int>::min();
//
//
//			for (int i = 0; i < xs.size(); ++i) {
//				Atom<T, N> &p = xs[i];
//				auto pos = p.now;
//				if (pos.x < min_x) min_x = pos.x;
//				if (pos.y < min_y) min_y = pos.y;
//				if (pos.z < min_z) min_z = pos.z;
//
//				if (pos.x > max_x) max_x = pos.x;
//				if (pos.y > max_y) max_y = pos.y;
//				if (pos.z > max_z) max_z = pos.z;
//			}
//
//			min_x -= grid_cell_side_length * 2;
//			min_y -= grid_cell_side_length * 2;
//			min_z -= grid_cell_side_length * 2;
//
//			max_x += grid_cell_side_length * 2;
//			max_y += grid_cell_side_length * 2;
//			max_z += grid_cell_side_length * 2;
//
//			glm::tvec3<N> minP(min_x, min_y, min_z);
//			glm::tvec3<N> maxP(max_x, max_y, max_z);
//
//			glm::tvec3<size_t> sizes(
//					static_cast<size_t>((max_x - min_x) / grid_cell_side_length),
//					static_cast<size_t>((max_y - min_y) / grid_cell_side_length),
//					static_cast<size_t>((max_z - min_z) / grid_cell_side_length)
//			);
//
//			size_t count = get_grid_index_z_curve(
//					sizes.x, sizes.y, sizes.z);
//
////#pragma parallel for
//			for (int i = 0; i < xs.size(); ++i) {
//				Atom<T, N> &p = xs[i];
//				p.zIndex = get_grid_index_z_curve(p.now.x, p.now.y, p.now.z);
//			}
//
//			ska_sort(xs.begin(), xs.end(), [](const Atom<T, N> &a) { return a.zIndex; });
////			std::sort(xs.begin(), xs.end(), [](const Atom<T, N> &lhs, const Atom<T, N> &rhs) {
////				return lhs.zIndex > rhs.zIndex;
////			});
//
//
//
//
//
//			std::vector<size_t> ses(count);
//			size_t current_index = 0;
//			for (size_t i = 0; i < count; ++i) {
//				ses[i] = current_index;
//				while (current_index != xs.size() && xs[current_index].zIndex == i) {
//					current_index++;
//				}
//			}
//
//
//
//
//
////			std::sort(xs.begin(), xs.end(), [](const Atom<T, N> &lhs, const Atom<T, N> &rhs) {
////				return lhs.zIndex > rhs.zIndex;
////			});
//
//
//			std::cout << "\t**ZI "
//			          << " side length=" << grid_cell_side_length
//			          << " [CI]=" << current_index
//			          << " [CL]=" << ses.size()
//			          << " count=" << count << "->"
//			          << glm::to_string(get_cell_coords_z_curve(count))
//			          << " X*Y*Z=" << glm::to_string(sizes)
//			          << " min=" << glm::to_string(minP)
//			          << " max=" << glm::to_string(maxP)
//			          << std::endl;
//
//
//
//
//			// bound size
//			// offset bound size
//			// for all particle
//			//     write grid_index
//			// build grid table
//			// radix sort grid
//			// do sph
//
//
//
//
//
////
////			N H = 0.1f;
////			N H2 = H * 2;
////			std::vector<N> off = {-H, 0.f, H};
////			for (size_t x = 0; x < off.size(); ++x)
////				for (size_t y = 0; y < off.size(); ++y)
////					for (size_t z = 0; z < off.size(); ++z)
////						offsets.push_back(tvec3<N>(off[x], off[y], off[z]));
////
////			for (size_t i = 0; i < xs.size(); ++i) {
////				const Atom<T, N> &p = xs[i];
////
////				for (const auto ov : offsets) {
////					M.insert({hash(p.now + ov) % xs.size(), p.particle->t});
////
////				}
////
////
//////				M.insert({hash(p.now), p.particle->t});
////			}
//
//
//		}
//
//
//		size_t length() {
//			return M.size();
//		}
//
//
//		void find(const Atom<T, N> &p, N radius, std::vector<size_t> &acc) {
//			auto r2 = radius * radius;
////			for (const auto ov : offsets) {
//			auto ret = M.equal_range(hash(p.now));
//			for (auto it = ret.first; it != ret.second; ++it) {
//				if (glm::distance2(xs[it->second].now, p.now) <= r2) {
//					acc.push_back(it->second);
//				}
//			}
////			}
//		}
//
//
//	};

	template<typename T, typename N>
	struct AtomCloud {
		const std::vector<Atom<T, N>> &pts;
		const N scale;

		AtomCloud(const std::vector<Atom<T, N>> &pts, const N scale) : pts(pts), scale(scale) {}

		inline size_t kdtree_get_point_count() const { return pts.size(); }

		inline N kdtree_get_pt(const size_t idx, const size_t dim) const {

//			if (dim == 0) return pts[idx].now.x;
//			else if (dim == 1) return pts[idx].now.y;
//			else return pts[idx].now.z;

			return pts[idx].now[dim] * scale;
		}

		template<class BBOX>
		bool kdtree_get_bbox(BBOX &) const { return false; }
	};


	template<typename T, typename N>
	class SphSolver {

	public:
		clsph::CLOps<T, N> clo = clsph::CLOps<T, N>();

		explicit SphSolver(N h = 0.1, N scale = 1) : h(h), scale(scale) {
			clsph::enumeratePlatformToCout();
		}

	public:

		const N h;
		const N scale;

		static constexpr N VD = 0.49;// Velocity dampening;
		static constexpr N RHO = 6378; // Reference density;
		static constexpr N EPSILON = 0.00000001;
		static constexpr N CFM_EPSILON = 600.0; // CFM propagation;

		static constexpr N C = 0.00001;
		static constexpr N VORTICITY_EPSILON = 0.0005;
		static constexpr N CorrK = 0.0001;
		static constexpr N CorrN = 4.f;

	private:


		const N poly6Factor = 315.f / (64.f * glm::pi<N>() * std::pow(h, 9.f));
		const N spikyKernelFactor = -(45.f / (glm::pi<N>() * std::pow(h, 6.f)));
		const N h2 = h * 2;
		const N hp2 = h * h;
		const N CorrDeltaQ = 0.3f * h;
		const N p6DeltaQ = poly6Kernel(CorrDeltaQ);

		const N poly6Kernel(N r) {
			return r <= h ? poly6Factor * std::pow(hp2 - r * r, 3.f) : 0.f;
		}

		// 4.9.2 Spiky Kernel(Desbrun and Gascuel (1996))
		const tvec3<N> spikyKernelGradient(tvec3<N> x, tvec3<N> y, N &r) {
			r = glm::distance(x, y);
			return !(r <= h && r >= EPSILON) ?
			       tvec3<N>(0) :
			       (x - y) * (spikyKernelFactor * (std::pow(h - r, 2.f) / r));
		}

	public:

		void advance(N dt, size_t iteration,
		             std::vector<Particle<T, N>> &xs,
		             const std::function<tvec3<N>(const Particle<T, N> &)> &constForce,
		             const std::vector<std::function<const Response<N>(Ray<N> &)> > &colliders
		) {

			using namespace std::chrono;
			using hrc = high_resolution_clock;


			std::vector<Atom<T, N>> atoms;

			std::transform(xs.begin(), xs.end(), std::back_inserter(atoms),
			               [constForce, dt, this](Particle<T, N> &p) {
				               auto a = Atom<T, N>(&p);
				               a.velocity = constForce(p) * dt + p.velocity;
				               a.mass = p.mass;
				               a.now = (a.velocity * dt) + (p.position / scale);
				               return a;
			               });

			hrc::time_point gnn = hrc::now();

			allPairNN(atoms);

//			GridNN<T, N> rtn = GridNN<T, N>(atoms);
//			std::cout << "\tGNN done " << std::endl;
			hrc::time_point gnne = hrc::now();
			auto nng = duration_cast<nanoseconds>(gnne - gnn).count();
			std::cout << "\tZIDX: " << (nng / 1000000.0) << "ms   " << std::endl;
//

			hrc::time_point nns = hrc::now();

			const N truncation = 30.f;
#ifndef FLANN


			std::vector<tvec3<N>> pts;
			std::transform(atoms.begin(), atoms.end(), std::back_inserter(pts),
						   [truncation](const Atom<T, N> &a) { return a.now * truncation; });

			unibn::Octree<tvec3<N>> octree;
			octree.initialize(pts);
#elseif A

			using namespace nanoflann;
//			std::vector<tvec3<N>> pts;
//			std::transform(atoms.begin(), atoms.end(), std::back_inserter(pts),
//						   [thingy](const Atom<T, N> &a) { return a.now * thingy; });

			const AtomCloud<T, N> cloud = AtomCloud<T, N>(atoms, truncation);
			typedef KDTreeSingleIndexAdaptor<
					L2_Simple_Adaptor<N, AtomCloud<T, N> >,
					AtomCloud<T, N>, 3> kd_tree_t;

			kd_tree_t index(3, cloud, KDTreeSingleIndexAdaptorParams(10));
			index.buildIndex();

			nanoflann::SearchParams params;
			params.sorted = false;
#endif

//			// NN search
#pragma omp parallel for
			for (int i = 0; i < atoms.size(); ++i) {
				Atom<T, N> &a = atoms[i];


#ifndef FLANN
				std::vector<uint32_t> neighbours;
				octree.template radiusNeighbors<unibn::L2Distance<tvec3<N>>>(
						a.now * truncation, (float) (h2 * truncation), neighbours);
				a.neighbours->reserve(neighbours.size());
				a.p6ks->reserve(neighbours.size());
				a.skgs->reserve(neighbours.size());
				for (uint32_t &idx : neighbours)
					a.neighbours->emplace_back(&atoms[idx]);


#elseif A

//				std::vector<size_t> n2;
//				rtn.find(a, h2, n2);
//				a.neighbours->reserve(n2.size());
//				a.p6ks->reserve(n2.size());
//				a.skgs->reserve(n2.size());
//				for (const size_t t : n2)
//					a.neighbours->emplace_back(&atoms[t]);

//				octree.template radiusNeighbors<unibn::L2Distance<tvec3<N>>>(
//						a.now * truncation, (float) (h2 * truncation), n2);


//				std::cout << ">>" << a.particle->t << std::endl;
//				std::cout << "GNN[";
//				for (size_t &idx : n2)
//					std::cout << idx << ' ';
//				std::cout << "]" << std::endl;


//				std::cout << "\tNNC: " <<  neighbours.size() << std::endl;


				std::vector<std::pair<size_t, N> > drain;

				auto origin = a.now * truncation;

				index.radiusSearch(&origin[0], h2 * truncation, drain, params);
				a.neighbours->reserve(drain.size());
				a.p6ks->reserve(drain.size());
				a.skgs->reserve(drain.size());
				for (const std::pair<size_t, N> &p : drain)
					a.neighbours->emplace_back(&atoms[p.first]);

//				std::cout << "OCT[";
//				for (const std::pair<size_t, N> &p : drain)
//					std::cout << p.first << ' ';
//				std::cout << "]" << std::endl;

#endif

			}


			hrc::time_point nne = hrc::now();
			auto nn = duration_cast<nanoseconds>(nne - nns).count();

			std::cout << "\tNN: " << (nn / 1000000.0) << "ms" << std::endl;

			hrc::time_point kerns = hrc::now();
//
//
//			auto rs = clo.run(atoms, iteration, scale);
//
//
//			for (int j = 0; j < atoms.size(); ++j) {
//				Atom<T, N> &a = atoms[j];
//				a.particle->velocity = clutil::glmT(rs[j].velocity);
//				a.particle->position = clutil::glmT(rs[j].position);
//			}


#define USE_CPU


#ifdef USE_CPU

			for (size_t j = 0; j < iteration; ++j) {

				// solve for lambda
#pragma omp parallel for
				for (int i = 0; i < atoms.size(); ++i) {
					Atom<T, N> &a = atoms[i];
					// Rho : density of a particle
					N rho = 0.f;
					auto norm2V = tvec3<N>(0);
					int nss = 0;
					for (size_t l = 0; l < a.neighbours->size(); ++l) {
						Atom<T, N> *b = (*a.neighbours)[l];
						N r;
						tvec3<N> skg = spikyKernelGradient(a.now, b->now, r);
						N p6k = poly6Kernel(r);
						rho += b->mass * p6k;
						norm2V += skg * (1.f / RHO);
						(*a.skgs)[l] = skg;
						(*a.p6ks)[l] = p6k;
//						std::cout << "[" << a.particle->t << "]["<< b->particle->t << "]NS:" << nss  <<std::endl;

						nss += b->particle->t;
					}
					auto norm2 = glm::length2(norm2V);
					N C = (rho / RHO - 1.f);
					a.lambda = -C / (norm2 + CFM_EPSILON);
//					std::cout << "["<< a.particle->t << "]NS:" << nss  << " N2:" << rho <<std::endl;

				}

				// solve for delta p
#pragma omp parallel for
				for (int i = 0; i < atoms.size(); ++i) {
					Atom<T, N> &a = atoms[i];
					a.deltaP = tvec3<N>(0);

					for (size_t l = 0; l < a.neighbours->size(); ++l) {
						Atom<T, N> *b = (*a.neighbours)[l];
						N corr = -CorrK *
						         std::pow((*a.p6ks)[l] / p6DeltaQ, CorrN);
						N factor = (a.lambda + b->lambda + corr) / RHO;
						a.deltaP = (*a.skgs)[l] * factor + a.deltaP;
					}

					auto current = Response<N>((a.now + a.deltaP) * scale, a.velocity);
					for (const auto &f : colliders) {
						Ray<N> ray = Ray<N>(a.particle->position,
						                    current.getPosition(),
						                    current.getVelocity());
						current = f(ray);
					}

					a.now = current.getPosition() / scale;
					a.velocity = current.getVelocity();
				}
			}


			// finalise
			for (Atom<T, N> &a : atoms) {


				auto deltaX = a.now - a.particle->position / scale;
				a.particle->position = a.now * scale;
				a.particle->mass = a.mass;
				a.particle->velocity = (deltaX * (1.f / dt) + a.velocity) * VD;
			}

//			for (Atom<T, N> &a : atoms) {
//				std::cout << "CPU >> "
//						  << " p=" << glm::to_string(a.particle->position)
//						  << " v=" << glm::to_string(a.particle->velocity)
//						<< " lam=" << a.lambda
//						<< " deltaP=" << glm::to_string(a.deltaP)
//						  << std::endl;
//			}

#endif
			hrc::time_point kerne = hrc::now();


			auto kern = duration_cast<nanoseconds>(kerne - kerns).count();
			std::cout << "\tKern: " << (kern / 1000000.0) << "ms" << std::endl;


		}


	};



//namespace unibn {
//	namespace traits {
//		template<typename T>
//		struct access<fluid::Atom<T>, 0> {
//			static float get(const fluid::Atom<T> &p) {
//				return static_cast<float>(p.now.x);
//			}
//		};
//
//		template<typename T>
//		struct access<fluid::Atom<T>, 1> {
//			static float get(const fluid::Atom<T> &p) {
//				return static_cast<float>(p.now.y);
//			}
//		};
//
//		template<typename T>
//		struct access<fluid::Atom<T>, 2> {
//			static float get(const fluid::Atom<T> &p) {
//				return static_cast<float>(p.now.z);
//			}
//		};
//	}
//}

}

#endif //LIBFLUID_CPUSPH_HPP
