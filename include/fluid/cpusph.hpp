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
#include "nanoflann.hpp"
#include <cstdlib>
#include <memory>
#include "oclsph.hpp"
#include "fluid.hpp"
#include "zcurve.h"


namespace cpu {

	using fluid::Particle;
	using fluid::Response;
	using fluid::Ray;
	using glm::tvec3;


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

		explicit Atom() :
				lambda(0),
				now(tvec3<N>(0)),
				deltaP(tvec3<N>(0)),
				omega(tvec3<N>(0)),
				velocity(tvec3<N>(0)),
				neighbours(std::make_unique<std::vector<Atom<T, N> *>>()),
				p6ks(std::make_unique<std::vector<N>>()),
				skgs(std::make_unique<std::vector<tvec3<N>>>()) {}
	};

	template<typename T, typename N>
	struct AtomCloud {
		const std::vector<Atom<T, N>> &pts;
		const N scale;

		AtomCloud(const std::vector<Atom<T, N>> &pts, const N scale) : pts(pts), scale(scale) {}

		inline size_t kdtree_get_point_count() const { return pts.size(); }

		inline N kdtree_get_pt(const size_t idx, const size_t dim) const {
			return pts[idx].now[dim] * scale;
		}

		template<class BBOX>
		bool kdtree_get_bbox(BBOX &) const { return false; }
	};

	template<typename T, typename N>
	class SphSolver : public fluid::SphSolver<T, N> {

	private:

		static constexpr N VD = 0.49;// Velocity dampening;
		static constexpr N RHO = 6378; // Reference density;
		static constexpr N EPSILON = 0.00000001;
		static constexpr N CFM_EPSILON = 600.0; // CFM propagation;
		static constexpr N C = 0.00001;
		static constexpr N VORTICITY_EPSILON = 0.0005;
		static constexpr N CorrK = 0.0001;
		static constexpr N CorrN = 4.f;

		const N h;
		const N h2;
		const N hp2;
		const N CorrDeltaQ;
		const N poly6Factor;
		const N spikyKernelFactor;
		const N p6DeltaQ = poly6Kernel(CorrDeltaQ);


	private :
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

		explicit SphSolver(N h) :
				h(h),
				h2(h * 2),
				hp2(h * h),
				CorrDeltaQ(0.3f * h),
				poly6Factor(315.f / (64.f * glm::pi<N>() * std::pow(h, 9.f))),
				spikyKernelFactor(-(45.f / (glm::pi<N>() * std::pow(h, 6.f)))),
				p6DeltaQ(poly6Kernel(CorrDeltaQ)) {}

		std::vector<surface::Triangle<N>> advance(const fluid::Config<N> &config,
		             std::vector<Particle<T, N>> &xs,
		             const std::vector<fluid::MeshCollider<N>> &colliders) override {

			std::vector<Atom<T, N>> atoms(xs.size());

#pragma omp parallel for
			for (int i = 0; i < static_cast<int>(atoms.size()); ++i) {
				Particle<T, N> &p = xs[i];
				Atom<T, N> &a = atoms[i];
				a.particle = &p;
				a.velocity = (config.constantForce * p.mass) * config.dt + p.velocity;
				a.mass = p.mass;
				a.now = (a.velocity * config.dt) + (p.position / config.scale);
			}

			const N truncation = 30.f;
			using namespace nanoflann;
			const AtomCloud<T, N> cloud = AtomCloud<T, N>(atoms, truncation);
			typedef KDTreeSingleIndexAdaptor<
					L2_Simple_Adaptor<N, AtomCloud<T, N> >,
					AtomCloud<T, N>, 3> kd_tree_t;

			kd_tree_t index(3, cloud, KDTreeSingleIndexAdaptorParams(10));
			index.buildIndex();
			SearchParams params;
			params.sorted = false;


#pragma omp parallel for
			for (int i = 0; i < static_cast<int>(atoms.size()); ++i) {
				Atom<T, N> &a = atoms[i];
				std::vector<std::pair<size_t, N> > drain;
				auto origin = a.now * truncation;

				index.radiusSearch(&origin[0], h2 * truncation, drain, params);
				a.neighbours->reserve(drain.size());
				a.p6ks->reserve(drain.size());
				a.skgs->reserve(drain.size());
				for (const std::pair<size_t, N> &p : drain)
					a.neighbours->emplace_back(&atoms[p.first]);
			}

			for (size_t j = 0; j < config.iteration; ++j) {

				// solve for lambda
#pragma omp parallel for
				for (int i = 0; i < static_cast<int>(atoms.size()); ++i) {
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
				for (int i = 0; i < static_cast<int>(atoms.size()); ++i) {
					Atom<T, N> &a = atoms[i];
					a.deltaP = tvec3<N>(0);

					for (size_t l = 0; l < a.neighbours->size(); ++l) {
						Atom<T, N> *b = (*a.neighbours)[l];
						N corr = -CorrK *
						         std::pow((*a.p6ks)[l] / p6DeltaQ, CorrN);
						N factor = (a.lambda + b->lambda + corr) / RHO;
						a.deltaP = (*a.skgs)[l] * factor + a.deltaP;
					}

					auto current = Response<N>((a.now + a.deltaP) * config.scale, a.velocity);


					// TOOD impl mesh collider
//					for (const auto &f : colliders) {
//						Ray<N> ray = Ray<N>(a.particle->position,
//						                    current.getPosition(),
//						                    current.getVelocity());
//						current = f(ray);
//					}


					auto clamped = glm::min(config.max,
					                        glm::max(config.min, current.getPosition()));

					a.now = clamped / config.scale;
					a.velocity = current.getVelocity();
				}
			}


#pragma omp parallel for
			for (int i = 0; i < static_cast<int>(atoms.size()); ++i) {
				Atom<T, N> &a = atoms[i];
				auto deltaX = a.now - a.particle->position / config.scale;
				a.particle->position = a.now * config.scale;
				a.particle->mass = a.mass;
				a.particle->velocity = (deltaX * (1.f / config.dt) + a.velocity) * VD;
			}
		}

	};


}

#endif //LIBFLUID_CPUSPH_HPP
