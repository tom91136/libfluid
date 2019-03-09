#ifndef LIBFLUID_FLUID_H
#define LIBFLUID_FLUID_H

#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_SIMD_AVX2
#define GLM_ENABLE_EXPERIMENTAL


#include <ostream>
#include <functional>
#include <vector>
#include <numeric>
#include <chrono>
#include <iostream>
#include <algorithm>
#include "glm/ext.hpp"
#include "glm/glm.hpp"
#include "Octree.hpp"
#include "nanoflann.hpp"
#include <cstdlib>
#include <memory>


namespace fluid {


	using glm::tvec3;


	enum Type {
		Fluid = 5, Obstacle = 6
	};


	template<typename T, typename N>
	struct Particle {
		T t;
		Type type;
		N mass;
		tvec3<N> position;
		tvec3<N> velocity;
//		std::unique_ptr<std::vector<uint32_t >> neighbours;

		explicit Particle(T t, Type type, N mass, const tvec3<N> &position,
		                  const tvec3<N> &velocity) :
				t(t), type(type), mass(mass), position(position), velocity(velocity) {
//			neighbours = std::make_unique<std::vector<uint32_t >>();

		}

		friend std::ostream &operator<<(std::ostream &os, const Particle &particle) {
			os << "t: " << particle.t
			   << ", mass: " << particle.mass
			   << ", pos: " << glm::to_string(particle.position)
			   << ", vel: " << glm::to_string(particle.velocity);
			return os;
		}

		bool operator==(const Particle &rhs) const {
			return t == rhs.t &&
			       mass == rhs.mass &&
			       position == rhs.position &&
			       velocity == rhs.velocity;
		}

		bool operator!=(const Particle &rhs) const {
			return !(rhs == *this);
		}
	};

	template<typename N>
	struct Ray {

		Ray(const tvec3<N> &prev, const tvec3<N> &origin, const tvec3<N> &velocity) :
				prev(prev), origin(origin), velocity(velocity) {}

		const tvec3<N> &getPrev() const { return prev; }

		const tvec3<N> &getOrigin() const { return origin; }

		const tvec3<N> &getVelocity() const { return velocity; }

	private:
		tvec3<N> prev, origin, velocity;
	};

	template<typename N>
	struct Response {

		Response(const tvec3<N> &position, const tvec3<N> &velocity) :
				position(position), velocity(velocity) {}

		const tvec3<N> &getPosition() const { return position; }

		const tvec3<N> &getVelocity() const { return velocity; }

	private:
		tvec3<N> position, velocity;
	};


	template<typename T, typename N>
	struct Atom {

		Particle<T, N> *particle;
		std::unique_ptr<std::vector<Atom<T, N> *>> neighbours;
		std::unique_ptr<std::vector<N>> p6ks;
		std::unique_ptr<std::vector<tvec3<N>>> skgs;
		N mass;
		tvec3<N> now;
		N lambda;
		tvec3<N> deltaP;
		tvec3<N> omega;
		tvec3<N> velocity;

		explicit Atom(Particle<T, N> *particle) : particle(particle) {
			lambda = 0;
			now = tvec3<N>(0);
			deltaP = tvec3<N>(0);
			omega = tvec3<N>(0);
			velocity = tvec3<N>(0);
			neighbours = std::make_unique<std::vector<Atom<T, N> *>>();
			p6ks = std::make_unique<std::vector<N>>();
			skgs = std::make_unique<std::vector<tvec3<N>>>();

		}
	};

	template<typename T, typename N>
	struct AtomCloud {
		const std::vector<Atom<T, N>> &pts;
		const N scale;

		AtomCloud(const std::vector<Atom<T, N>> &pts,	const N scale) : pts(pts), scale(scale) {}

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
		explicit SphSolver(N h = 0.1, N scale = 1) : h(h), scale(scale) {}

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


			hrc::time_point nns = hrc::now();

			const N truncation = 30.f;
#ifndef FLANN


			std::vector<tvec3<N>> pts;
			std::transform(atoms.begin(), atoms.end(), std::back_inserter(pts),
			               [truncation](const Atom<T, N> &a) { return a.now * truncation; });

			unibn::Octree<tvec3<N>> octree;
			octree.initialize(pts);
#else

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


				#else
				std::vector<std::pair<size_t, N> > drain;

				auto origin = a.now * truncation;

				index.radiusSearch(&origin[0], h2 * truncation, drain, params);
				a.neighbours->reserve(drain.size());
				a.p6ks->reserve(drain.size());
				a.skgs->reserve(drain.size());
				for (const std::pair<size_t, N> &p : drain)
					a.neighbours->emplace_back(&atoms[p.first]);


#endif

			}


			hrc::time_point nne = hrc::now();
			auto nn = duration_cast<nanoseconds>(nne - nns).count();

			std::cout << "\tNN: " << (nn / 1000000.0) << "ms" << std::endl;

			hrc::time_point kerns = hrc::now();

			for (size_t j = 0; j < iteration; ++j) {

				// solve for lambda
#pragma omp parallel for
				for (int i = 0; i < atoms.size(); ++i) {
					Atom<T, N> &a = atoms[i];
					// Rho : density of a particle
					N rho = 0.f;
					auto norm2V = tvec3<N>(0);
					for (size_t l = 0; l < a.neighbours->size(); ++l) {
						Atom<T, N> *b = (*a.neighbours)[l];
						N r;
						tvec3<N> skg = spikyKernelGradient(a.now, b->now, r);
						N p6k = poly6Kernel(r);
						rho += b->mass * p6k;
						norm2V += skg * (1.f / RHO);
						(*a.skgs)[l] = skg;
						(*a.p6ks)[l] = p6k;
					}
					auto norm2 = glm::length2(norm2V);
					N C = (rho / RHO - 1.f);
					a.lambda = -C / (norm2 + CFM_EPSILON);
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

			hrc::time_point kerne = hrc::now();

			auto kern = duration_cast<nanoseconds>(kerne - kerns).count();
			std::cout << "\tKern: " << (kern / 1000000.0) << "ms" << std::endl;


		}


	};


}

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


#endif