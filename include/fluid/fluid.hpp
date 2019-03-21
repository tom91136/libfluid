#ifndef LIBFLUID_FLUID_HPP
#define LIBFLUID_FLUID_HPP


#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_SIMD_AVX2
#define GLM_ENABLE_EXPERIMENTAL


#include <algorithm>
#include "glm/ext.hpp"
#include "glm/glm.hpp"


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


	template<typename N>
	struct MeshCollider {
		const std::vector<glm::tvec3<N>> &triangles;

		explicit MeshCollider(const std::vector<glm::tvec3<N>> &triangles) : triangles(triangles) {}
	};

	template<typename N>
	struct Config {

		N dt;
		N scale;
		size_t iteration;
		glm::tvec3<N> constantForce;

		glm::tvec3<N> min;
		glm::tvec3<N> max;

		explicit Config(N dt, N scale, size_t iteration,
		                const glm::tvec3<N> &constantForce,
		                const glm::tvec3<N> &min,
		                const glm::tvec3<N> &max) : dt(dt), scale(scale),
		                                            iteration(iteration),
		                                            constantForce(constantForce),
		                                            min(min), max(max) {}
	};

	template<typename T, typename N>
	class SphSolver {
	public :

		virtual ~SphSolver() = default;

		virtual void advance(const Config<N> &config,
		                     std::vector<Particle<T, N>> &xs,
		                     const std::vector<MeshCollider<N>> &colliders) = 0;
	};


}


#endif // LIBFLUID_FLUID_HPP