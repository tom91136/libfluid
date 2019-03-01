#include <catch2/catch.hpp>
#include "fluid.hpp"
#include <chrono>


TEST_CASE("Solver is correct") {

	using namespace fluid;

	typedef glm::vec3 vec3t;
	typedef float num_t;

	std::vector<fluid::Particle<size_t>> expected = {
			Particle<size_t>(0, 1.0, vec3t(27.181, 32.112, 27.181), vec3t(0.346, 0.729, 0.346)),
			Particle<size_t>(1, 1.0, vec3t(22.772, 27.702, 22.772), vec3t(0.168, 0.550, 0.168)),
			Particle<size_t>(2, 1.0, vec3t(19.554, 24.484, 19.554), vec3t(0.017, 0.400, 0.017)),
			Particle<size_t>(3, 1.0, vec3t(34.452, 39.382, 34.452), vec3t(-0.001, 0.382, -0.001)),
			Particle<size_t>(4, 1.0, vec3t(41.787, 46.717, 41.787), vec3t(-0.018, 0.365, -0.018)),
			Particle<size_t>(5, 1.0, vec3t(48.213, 53.143, 48.213), vec3t(0.018, 0.401, 0.018)),
			Particle<size_t>(6, 1.0, vec3t(55.548, 60.479, 55.548), vec3t(0.001, 0.384, 0.001)),
			Particle<size_t>(7, 1.0, vec3t(70.446, 75.377, 70.446), vec3t(-0.017, 0.366, -0.017)),
			Particle<size_t>(8, 1.0, vec3t(67.228, 72.159, 67.228), vec3t(-0.168, 0.215, -0.168)),
			Particle<size_t>(9, 1.0, vec3t(62.819, 67.749, 62.819), vec3t(-0.346, 0.037, -0.346))
	};


	std::vector<fluid::Particle<size_t>> actual;
	for (size_t i = 0; i < 10; ++i) {
		actual.emplace_back(i, 1, vec3t(i * 10), vec3t(0));
	}

	std::vector<std::function<const fluid::Response(fluid::Ray &)> > colliders = {
			[](const fluid::Ray &x) -> fluid::Response {
				return fluid::Response(glm::clamp(x.getOrigin(), (num_t) 0.f, (num_t) 500.f),
				                       x.getVelocity());
			}
	};

	std::unique_ptr<fluid::SphSolver<size_t>> solver(new fluid::SphSolver<size_t>(0.1, 500));

	for (size_t j = 0; j < 5; ++j) {
		using namespace std;
		using namespace std::chrono;
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
		solver->advance(0.0083, 5, actual,
		                [](const fluid::Particle<size_t> &x) {
			                return vec3t(0, x.mass * 9.8, 0);
		                }, colliders
		);
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		auto elapsed = duration_cast<milliseconds>(t2 - t1).count();
		std::cout << "Iter" << j << "@ " << elapsed << "ms" << std::endl;
	}


	for (size_t j = 0; j < 10; ++j) {
		REQUIRE(expected[j].t == actual[j].t);
		REQUIRE(abs(expected[j].mass - actual[j].mass) < 0.001);
		REQUIRE(glm::distance(expected[j].velocity, actual[j].velocity) < 0.5);
		REQUIRE(glm::distance(expected[j].position, actual[j].position) < 0.5);
	}

	actual.clear();
}