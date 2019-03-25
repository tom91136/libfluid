
#include <catch2/catch.hpp>
#include <chrono>
#include <iostream>
#include "fluid/fluid.hpp"
#include "fluid/cpusph.hpp"

using namespace fluid;
using glm::tvec3;

typedef float num_t;
typedef tvec3<num_t> v3n;
typedef size_t p_t;

TEST_CASE("Solver is correct") {


	using fluid::Fluid;
	std::vector<fluid::Particle<p_t, num_t>> expected;
	expected.emplace_back(0, Fluid, 1.0, v3n(27.181, 32.112, 27.181), v3n(0.346, 0.729, 0.346));
	expected.emplace_back(1, Fluid, 1.0, v3n(22.772, 27.702, 22.772), v3n(0.168, 0.550, 0.168));
	expected.emplace_back(2, Fluid, 1.0, v3n(19.554, 24.484, 19.554), v3n(0.017, 0.400, 0.017));
	expected.emplace_back(3, Fluid, 1.0, v3n(34.452, 39.382, 34.452), v3n(-0.001, 0.382, -0.001));
	expected.emplace_back(4, Fluid, 1.0, v3n(41.787, 46.717, 41.787), v3n(-0.018, 0.365, -0.018));
	expected.emplace_back(5, Fluid, 1.0, v3n(48.213, 53.143, 48.213), v3n(0.018, 0.401, 0.018));
	expected.emplace_back(6, Fluid, 1.0, v3n(55.548, 60.479, 55.548), v3n(0.001, 0.384, 0.001));
	expected.emplace_back(7, Fluid, 1.0, v3n(70.446, 75.377, 70.446), v3n(-0.017, 0.366, -0.017));
	expected.emplace_back(8, Fluid, 1.0, v3n(67.228, 72.159, 67.228), v3n(-0.168, 0.215, -0.168));
	expected.emplace_back(9, Fluid, 1.0, v3n(62.819, 67.749, 62.819), v3n(-0.346, 0.037, -0.346));


	std::vector<fluid::Particle<p_t, num_t>> actual;
	for (size_t i = 0; i < 10; ++i) {
		actual.emplace_back(i, Fluid, 1.f, tvec3<num_t>(i * 10), tvec3<num_t>(0));
	}

	std::unique_ptr<fluid::SphSolver<p_t, num_t>> solver(new cpu::SphSolver<p_t, num_t>(0.1));


	auto config = fluid::Config<num_t>(
			static_cast<num_t>(0.0083 * 1),
			500,
			5,
			tvec3<num_t>(0, 9.8, 0),
			v3n(0), v3n(500));

	std::vector<fluid::MeshCollider<num_t >> colliders;

	for (size_t j = 0; j < 5; ++j) {
		using namespace std;
		using namespace std::chrono;
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
		solver->advance(config, actual, colliders);
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		auto elapsed = duration_cast<milliseconds>(t2 - t1).count();
		std::cout << "Iter" << j << "@ " << elapsed << "ms" << std::endl;
	}

	for (size_t j = 0; j < 10; ++j) {
		// FIXME values are wrong
		REQUIRE(expected[j].id == actual[j].id);
		REQUIRE(abs(expected[j].mass - actual[j].mass) < 0.001);
		REQUIRE(glm::distance(expected[j].velocity, actual[j].velocity) < 0.95);
		REQUIRE(glm::distance(expected[j].position, actual[j].position) < 0.95);
	}

	actual.clear();
}