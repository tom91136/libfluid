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
#include "zcurve.h"


namespace cpusph {

	using fluid::Particle;
	using fluid::Atom;
	using fluid::Response;
	using fluid::Ray;


//	const static size_t uninterleave(size_t value) {
//		size_t ret = 0x0;
//		ret |= (value & 0x1) >> 0;
//		ret |= (value & 0x8) >> 2;
//		ret |= (value & 0x40) >> 4;
//		ret |= (value & 0x200) >> 6;
//		ret |= (value & 0x1000) >> 8;
//		ret |= (value & 0x8000) >> 10;
//		ret |= (value & 0x40000) >> 12;
//		ret |= (value & 0x200000) >> 14;
//		ret |= (value & 0x1000000) >> 16;
//		ret |= (value & 0x8000000) >> 18;
//		return ret;
//	}
//
	const static glm::tvec3<size_t> coordAtZCurveGridIndex(size_t index) {
		return glm::tvec3<size_t>(
				coordAtZCurveGridIndex0(index),
				coordAtZCurveGridIndex1(index),
				coordAtZCurveGridIndex2(index));
	}
//
//	const static size_t zCurveGridIndexAtCoord(size_t x, size_t y, size_t z) {
//		x = (x | (x << 16)) & 0x030000FF;
//		x = (x | (x << 8)) & 0x0300F00F;
//		x = (x | (x << 4)) & 0x030C30C3;
//		x = (x | (x << 2)) & 0x09249249;
//		y = (y | (y << 16)) & 0x030000FF;
//		y = (y | (y << 8)) & 0x0300F00F;
//		y = (y | (y << 4)) & 0x030C30C3;
//		y = (y | (y << 2)) & 0x09249249;
//		z = (z | (z << 16)) & 0x030000FF;
//		z = (z | (z << 8)) & 0x0300F00F;
//		z = (z | (z << 4)) & 0x030C30C3;
//		z = (z | (z << 2)) & 0x09249249;
//		return x | (y << 1) | (z << 2);
//	}


	template<typename T, typename N>
	const static void allPairNN(std::vector<Atom<T, N>> &atoms) {

		const N H2 = 0.1 / 2;

		glm::tvec3<N> min(std::numeric_limits<N>::max());
		glm::tvec3<N> max(std::numeric_limits<N>::min());

		N sideLength = (H2 * 2);

		for (int i = 0; i < atoms.size(); ++i) {
			Atom<T, N> &p = atoms[i];
			auto pos = p.now;
			if (pos.x < min.x) min.x = pos.x;
			if (pos.y < min.y) min.y = pos.y;
			if (pos.z < min.z) min.z = pos.z;

			if (pos.x > max.x) max.x = pos.x;
			if (pos.y > max.y) max.y = pos.y;
			if (pos.z > max.z) max.z = pos.z;
		}

		N padding = sideLength * 2;
		min -= padding;
		max += padding;
		glm::tvec3<size_t> sizes((max - min) / sideLength);

		size_t count = zCurveGridIndexAtCoord(sizes.x, sizes.y, sizes.z);

#pragma omp parallel for
		for (int i = 0; i < atoms.size(); ++i) {
			Atom<T, N> &p = atoms[i];
			auto scaled = (p.now - min) / sideLength;
			p.zIndex = zCurveGridIndexAtCoord(
					static_cast<size_t>(scaled.x ),
					static_cast<size_t>(scaled.y ),
					static_cast<size_t>(scaled.z ));
		}

//		ska_sort(atoms.begin(), atoms.end(), [](const Atom<T, N> &a) ->  { return   a.zIndex; });
		std::sort(atoms.begin(), atoms.end(), [](const Atom<T, N> &lhs, const Atom<T, N> &rhs) {
			return lhs.zIndex < rhs.zIndex;
		});

		std::vector<size_t> ses(count);
		size_t current_index = 0;
		for (size_t i = 0; i < count; ++i) {
			ses[i] = current_index;
//			std::cout << "\t-> " << i << "  =" << current_index << std::endl;
			while (current_index != atoms.size() && atoms[current_index].zIndex == i) {
				current_index++;
			}
		}


		for (size_t xx = 0; xx < atoms.size(); ++xx) {
			Atom<T, N> &p = atoms[xx];


			auto cell_coords = coordAtZCurveGridIndex(p.zIndex);


			for (size_t z = cell_coords.z - 1; z <= cell_coords.z + 1; ++z) {
				for (size_t y = cell_coords.y - 1; y <= cell_coords.y + 1; ++y) {
					for (size_t x = cell_coords.x - 1; x <= cell_coords.x + 1; ++x) {
						size_t offset = zCurveGridIndexAtCoord(x, y, z);


						size_t start = ses[offset];
						size_t end = ((offset + 1) < count) ? ses[offset + 1] : atoms.size();



//						std::cout << "\t\t" << "B[" << start << "->" << end << "]" << " @ "
//						          << glm::to_string(glm::tvec3<size_t>(x, y, z)) << " <- " << offset
//						          << " o="
//						          << glm::to_string(glm::tvec3<size_t>(x, y, z) - cell_coords)
//						          << std::endl;
						for (size_t ni = start; ni < end; ++ni) {
							p.neighbours->emplace_back(&atoms[ni]);
						}
					}
				}
			}


//			std::cout << "\t[ZI] " << p.particle->t << " N=" << p.neighbours->size()
//			          << glm::to_string(cell_coords) << std::endl;


			p.p6ks->reserve(p.neighbours->size());
			p.skgs->reserve(p.neighbours->size());


		}


		std::cout << "\t***ZI "
		          << " side length=" << sideLength
		          << " [CI]=" << current_index
		          << " [CL]=" << ses.size()
		          << " count=" << count << "->"
		          << glm::to_string(coordAtZCurveGridIndex(count))
		          << " X*Y*Z=" << glm::to_string(sizes)
		          << " min=" << glm::to_string(min)
		          << " max=" << glm::to_string(max)
		          << std::endl;
		std::cout << " Atom size >>> " << atoms.size() << std::endl;

	}

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

			hrc::time_point kerns = hrc::now();

			clo.run(xs, iteration, scale, tvec3<N>(0, 9.8, 0), dt);

//
//
//			std::vector<Atom<T, N>> atoms;
//
//			std::transform(xs.begin(), xs.end(), std::back_inserter(atoms),
//			               [constForce, dt, this](Particle<T, N> &p) {
//				               auto a = Atom<T, N>(&p);
//				               a.velocity = constForce(p) * dt + p.velocity;
//				               a.mass = p.mass;
//				               a.now = (a.velocity * dt) + (p.position / scale);
//				               return a;
//			               });
//
//			hrc::time_point gnn = hrc::now();
//
//			allPairNN(atoms);
//
////			GridNN<T, N> rtn = GridNN<T, N>(atoms);
////			std::cout << "\tGNN done " << std::endl;
//			hrc::time_point gnne = hrc::now();
//			auto nng = duration_cast<nanoseconds>(gnne - gnn).count();
//			std::cout << "\tZIDX: " << (nng / 1000000.0) << "ms   " << std::endl;
////
//
//			hrc::time_point nns = hrc::now();
//
//			const N truncation = 30.f;
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
//#pragma omp parallel for
//			for (int i = 0; i < atoms.size(); ++i) {
//				Atom<T, N> &a = atoms[i];


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

//			}


//			hrc::time_point nne = hrc::now();
//			auto nn = duration_cast<nanoseconds>(nne - nns).count();
//
//			std::cout << "\tNN: " << (nn / 1000000.0) << "ms" << std::endl;

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


//#define USE_CPU


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
