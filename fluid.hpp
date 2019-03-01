#ifndef LIBFLUID_FLUID_H
#define LIBFLUID_FLUID_H

#define GLM_ENABLE_EXPERIMENTAL

#include <ostream>
#include <functional>
#include <vector>
#include <numeric>
#include <iostream>
#include <algorithm>
#include "glm/ext.hpp"
#include "glm/glm.hpp"
#include "Octree.hpp"
#include <cstdlib>
#include <memory>


namespace fluid {


	typedef glm::vec3 vec3t;
	typedef float num_t;


	template<typename T>
	struct Particle {
		T t;
		num_t mass{};
		vec3t position;
		vec3t velocity;
		std::vector<size_t> neighbours; // TODO fill

		Particle(T t, num_t mass, const vec3t &position, const vec3t &velocity) :
				t(t), mass(mass), position(position), velocity(velocity) {

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

	struct Ray {

		Ray(const vec3t &prev, const vec3t &origin, const vec3t &velocity) :
				prev(prev), origin(origin), velocity(velocity) {}

		const vec3t &getPrev() const { return prev; }

		const vec3t &getOrigin() const { return origin; }

		const vec3t &getVelocity() const { return velocity; }

	private:
		vec3t prev, origin, velocity;
	};


	struct Response {

		Response(const vec3t &position, const vec3t &velocity) :
				position(position), velocity(velocity) {}

		const vec3t &getPosition() const { return position; }

		const vec3t &getVelocity() const { return velocity; }

	private:
		vec3t position, velocity;
	};


	template<typename T>
	struct Atom {

		Particle<T> *particle;
		std::unique_ptr<std::vector<Atom<T> *>> neighbours;
		std::unique_ptr<std::vector<num_t>> p6ks;
		std::unique_ptr<std::vector<vec3t>> skgs;
		vec3t now;
		num_t lambda;
		vec3t deltaP;
		vec3t omega;
		vec3t velocity;

		explicit Atom(Particle<T> *particle) : particle(particle) {
			lambda = 0;
			now = vec3t(0);
			deltaP = vec3t(0);
			omega = vec3t(0);
			velocity = vec3t(0);
			neighbours = std::make_unique<std::vector<Atom<T> *>>();
			p6ks = std::make_unique<std::vector<num_t>>();
			skgs = std::make_unique<std::vector<vec3t>>();

		}
	};


	template<typename T>
	class SphSolver {

	public:
		explicit SphSolver(num_t h = 0.1, num_t scale = 1) : h(h), scale(scale) {}

	public:

		const num_t h;
		const num_t scale;

		static constexpr num_t VD = 0.49;// Velocity dampening;
		static constexpr num_t RHO = 6378; // Reference density;
		static constexpr num_t EPSILON = 0.00000001;
		static constexpr num_t CFM_EPSILON = 600.0; // CFM propagation;

		static constexpr num_t C = 0.00001;
		static constexpr num_t VORTICITY_EPSILON = 0.0005;
		static constexpr num_t CorrK = 0.0001;
		static constexpr num_t CorrN = 4.f;

	private:


		const num_t poly6Factor = 315.f / (64.f * glm::pi<num_t>() * std::pow(h, 9.f));
		const num_t spikyKernelFactor = -(45.f / (glm::pi<num_t>() * std::pow(h, 6.f)));
		const num_t h2 = h * 2;
		const num_t hp2 = h * h;
		const num_t CorrDeltaQ = 0.3f * h;
		const num_t p6DeltaQ = poly6Kernel(CorrDeltaQ);

		const num_t poly6Kernel(num_t r) {
			return r <= h ? poly6Factor * std::pow(hp2 - r * r, 3.f) : 0.f;
		}

		// 4.9.2 Spiky Kernel(Desbrun and Gascuel (1996))
		const vec3t spikyKernelGradient(vec3t x, vec3t y, num_t &r) {
			r = glm::distance(x, y);
			return !(r <= h && r >= EPSILON) ?
			       vec3t(0) :
			       (x - y) * (spikyKernelFactor * (std::pow(h - r, 2.f) / r));
		}

	public:

		void advance(num_t dt, size_t iteration,
		             std::vector<Particle<T>> &xs,
		             const std::function<vec3t(const Particle<T> &)> &constForce,
		             const std::vector<std::function<const Response(Ray &)> > &colliders
		) {


			std::vector<Atom<T>> atoms;

			std::transform(xs.begin(), xs.end(), std::back_inserter(atoms),
			               [constForce, dt, this](Particle<T> &p) {
				               auto a = Atom<T>(&p);
				               a.velocity = constForce(p) * dt + p.velocity;
				               a.now = (a.velocity * dt) + (p.position / scale);
				               return a;
			               });

			// create Octree for fast lookup
			std::vector<vec3t> pts;
			std::transform(atoms.begin(), atoms.end(), std::back_inserter(pts),
			               [](const Atom<T> &a) { return a.now; });

			unibn::Octree<vec3t> octree;
			octree.initialize(pts);

//			// NN search
#pragma omp parallel for
			for (size_t i = 0; i < atoms.size(); ++i) {
				Atom<T> &a = atoms[i];
				std::vector<uint32_t> results;
				octree.radiusNeighbors<unibn::L2Distance<vec3t> >(a.now, (float) h2, results);

				a.neighbours->reserve(results.size());
				a.p6ks->reserve(results.size());
				a.skgs->reserve(results.size());
				for (uint32_t &result : results) a.neighbours->emplace_back(&atoms[result]);

			}

			for (size_t j = 0; j < iteration; ++j) {

				// solve for lambda
#pragma omp parallel for
				for (size_t i = 0; i < atoms.size(); ++i) {
					Atom<T> &a = atoms[i];
					num_t rho = 0.f;
					auto norm2V = vec3t(0);
					for (size_t l = 0; l < a.neighbours->size(); ++l) {
						Atom<T> *b = (*a.neighbours)[l];
						num_t r;
						vec3t skg = spikyKernelGradient(a.now, b->now, r);
						num_t p6k = poly6Kernel(r);
						rho += b->particle->mass * p6k;
						norm2V += skg * (1.f / RHO);
						(*a.skgs)[l] = skg;
						(*a.p6ks)[l] = p6k;
					}
					auto norm2 = glm::length2(norm2V);
					num_t C = (rho / RHO - 1.f);
					a.lambda = -C / (norm2 + CFM_EPSILON);
				}

				// solve for delta p
#pragma omp parallel for
				for (size_t i = 0; i < atoms.size(); ++i) {
					Atom<T> &a = atoms[i];
					a.deltaP = vec3t(0);

					for (size_t l = 0; l < a.neighbours->size(); ++l) {
						Atom<T> *b = (*a.neighbours)[l];
						num_t corr = -CorrK *
						             std::pow((*a.p6ks)[l] / p6DeltaQ, CorrN);
						num_t factor = (a.lambda + b->lambda + corr) / RHO;
						a.deltaP = (*a.skgs)[l] * factor + a.deltaP;
					}

					auto current = Response((a.now + a.deltaP) * scale, a.velocity);
					for (const auto &f : colliders) {
						Ray ray = Ray(a.particle->position,
						              current.getPosition(),
						              current.getVelocity());
						current = f(ray);
					}

					a.now = current.getPosition() / scale;
					a.velocity = current.getVelocity();
				}


			}

			// finalise
			for (Atom<T> &a : atoms) {
				auto deltaX = a.now - a.particle->position / scale;
				a.particle->position = a.now * scale;
				a.particle->velocity = (deltaX * (1.f / dt) + a.velocity) * VD;
			}
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