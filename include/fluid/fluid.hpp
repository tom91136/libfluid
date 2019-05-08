#ifndef LIBFLUID_FLUID_HPP
#define LIBFLUID_FLUID_HPP

#define GLM_ENABLE_EXPERIMENTAL

//#include <algorithm>
#include <ostream>
#include "glm/ext.hpp"
#include "glm/glm.hpp"
#include "geometry.hpp"


namespace fluid {

	using glm::tvec3;
	using glm::tvec4;

	enum Type {
		Fluid = 0, Obstacle = 1
	};


	template<typename N>
	struct Query {
		tvec3<N> point;
		size_t neighbours;
		uint32_t averageColour;
		Query(tvec3<N> point, size_t neighbours, uint32_t averageColour) :
				point(point), neighbours(neighbours), averageColour(averageColour) {}
	};


	template<typename T, typename N>
	struct Particle {
		T id;
		Type type;
		N mass;
		tvec3<N> position;
		tvec3<N> velocity;
		uint32_t colour;

		Particle() : type(Type::Fluid) {}

		explicit Particle(T t, Type type, N mass, const uint32_t &colour,
		                  const tvec3<N> &position,
		                  const tvec3<N> &velocity) :
				id(t), type(type), mass(mass),
				position(position),
				velocity(velocity),
				colour(colour) {}

		friend std::ostream &operator<<(std::ostream &os, const Particle &particle) {
			os << "t: " << particle.id
			   << ", mass: " << particle.mass
			   << ", colour: 0x" << std::hex << particle.colour
			   << ", pos: " << glm::to_string(particle.position)
			   << ", vel: " << glm::to_string(particle.velocity);
			return os;
		}

		bool operator==(const Particle &rhs) const {
			return id == rhs.id &&
			       type == rhs.type &&
			       mass == rhs.mass &&
			       colour == rhs.colour &&
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
		const std::vector<geometry::Triangle<N>> triangles;

		explicit MeshCollider(const std::vector<geometry::Triangle<N>> &triangles) :
				triangles(triangles) {}
	};


	template<typename N>
	struct RigidBody {
		const std::vector<tvec3<N>> obstacles;
		explicit RigidBody(const std::vector<tvec3<N>> &obstacles) : obstacles(obstacles) {}

	};


	template<typename N>
	struct Well {
		tvec3<N> centre;
		N force;
		friend std::ostream &operator<<(std::ostream &os, const Well &well) {
			return os << "Well(" << glm::to_string(well.centre) << " @" << well.force << ")";
		}
	};

	template<typename N>
	struct Source {
		tvec3<N> centre;
		tvec3<N> velocity;
		size_t rate, tag;
		uint32_t colour;
		friend std::ostream &operator<<(std::ostream &os, const Source &source) {
			return os << "Source[ " << source.tag
			          << "(0x" << std::hex << source.colour
			          << ") ](" << glm::to_string(source.centre)
			          << "<" << glm::to_string(source.velocity)
			          << "v @" << source.rate << "p)";
		}
	};

	template<typename N>
	struct Drain {
		tvec3<N> centre;
		N width, depth;
		friend std::ostream &operator<<(std::ostream &os, const Drain &drain) {
			return os << "Drain(" << glm::to_string(drain.centre)
			          << " @" << drain.width << "x" << drain.depth << ")";
		}
	};


	template<typename N>
	struct Config {

		const N dt;
		const N scale;
		const N resolution;
		const N isolevel;
		const size_t iteration;

		const tvec3<N> constantForce;
		const std::vector<fluid::Well<N>> wells;
		const std::vector<fluid::Source<N>> sources;
		const std::vector<fluid::Drain<N>> drains;

		const tvec3<N> minBound, maxBound;

		explicit Config(N dt, N scale, N resolution, N isolevel, size_t iteration,
		                const tvec3<N> &constantForce,
		                const std::vector<fluid::Well<N>> &wells,
		                const std::vector<fluid::Source<N>> &sources,
		                const std::vector<fluid::Drain<N>> &drains,
		                const tvec3<N> &min, const tvec3<N> &max)
				: dt(dt), scale(scale),
				  resolution(resolution), isolevel(isolevel), iteration(iteration),
				  constantForce(constantForce),
				  wells(wells), sources(sources), drains(drains),
				  minBound(min), maxBound(max) {}
		friend std::ostream &operator<<(std::ostream &os, const Config &config) {
			os << "dt: " << config.dt << " scale: " << config.scale << " iteration: "
			   << config.iteration << " constantForce: " << glm::to_string(config.constantForce)
			   << " minBound: "
			   << glm::to_string(config.minBound)
			   << " maxBound: " << glm::to_string(config.maxBound);
			return os;
		}
	};


	template<typename N>
	struct Result {
		std::vector<geometry::MeshTriangle<N>> triangles{};
		std::vector<Query<N>> queries{};
		Result() = default;
		Result(const std::vector<geometry::MeshTriangle<N>> &triangles,
		       const std::vector<Query<N>> &queries) : triangles(triangles), queries(queries) {}
	};

	template<typename T, typename N>
	class SphSolver {
	public :

		virtual ~SphSolver() = default;

		virtual Result<N> advance(
				const Config<N> &config,
				std::vector<Particle<T, N>> &xs,
				const std::vector<tvec3<N>> &queries,
				const std::vector<MeshCollider<N>> &colliders) = 0;
	};

}


#endif // LIBFLUID_FLUID_HPP