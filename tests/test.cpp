
#include <catch2/catch.hpp>
#include <chrono>
#include <iostream>
#include <mmf.hpp>
#include "fluid/fluid.hpp"
#include "fluid/cpusph.hpp"
#include "mio/mmap.hpp"

using namespace fluid;
using glm::tvec3;

typedef float num_t;
typedef tvec3<num_t> v3n;
typedef size_t p_t;


mio::mmap_sink createSink(const std::string &path, const size_t length) {
	std::ofstream file(path, std::ios::out | std::ios::trunc);
	std::string s(length, ' ');
	file << s;

	std::error_code error;
	mio::mmap_sink sink = mio::make_mmap_sink(path, 0, mio::map_entire_file, error);
	if (error) {
		std::cout << "MMF(" << path << ") failed:" << error.message() << std::endl;
		exit(1);
	} else {
		std::cout << "MMF(" << path << ") size: "
		          << (double) sink.size() / (1024 * 1024) << "MB" << std::endl;
	}
	return sink;
}

template<typename T>
struct Foo {
	T a, b, c;
	char _c{};
	int _i{};
	long _l{};

	Foo(T a, T b, T c, char _c, int _i, long _l) : a(a), b(b), c(c), _c(_c), _i(_i), _l(_l) {}

	Foo() = default;

	bool operator==(const Foo &rhs) const {
		return a == rhs.a && b == rhs.b && c == rhs.c &&
		       _c == rhs._c && _i == rhs._i && _l == rhs._l;
	}

	bool operator!=(const Foo &rhs) const {
		return !(rhs == *this);
	}

	friend std::ostream &operator<<(std::ostream &os, const Foo &foo) {
		os << "a: " << foo.a << " b: " << foo.b << " c: " << foo.c <<
		   " _c: " << foo._c << " _i: " << foo._i << " _l: " << foo._l;
		return os;
	}
};

static const char a[] = "a";
static const char b[] = "b";
static const char c[] = "c";

static const char _c[] = "_c";
static const char _i[] = "_i";
static const char _l[] = "_l";

template<typename T>
static inline auto fooEntries() {
	return mmf::makeDef(
			DECL_MEMBER(a, CLS(Foo<T>), a),
			DECL_MEMBER(b, CLS(Foo<T>), b),
			DECL_MEMBER(c, CLS(Foo<T>), c),
			DECL_MEMBER(_c, CLS(Foo<T>), _c),
			DECL_MEMBER(_i, CLS(Foo<T>), _i),
			DECL_MEMBER(_l, CLS(Foo<T>), _l)
	);
}


TEST_CASE("mmf ipc is isomorphic") {

	auto fe = fooEntries<float>();

	auto meta = mmf::meta::writeMetaPacked<decltype(fe)>();

	std::vector<char> buffer(meta.first);

	auto expected = Foo<float>(
			glm::pi<float>(),
			glm::half_pi<float>(),
			glm::quarter_pi<float>(),
			'z', 42, 44);


	SECTION("offset is correct") {
		REQUIRE(meta.first ==
		        sizeof(Foo<float>::a) + sizeof(Foo<float>::b) + sizeof(Foo<float>::c) +
		        sizeof(Foo<float>::_c) + sizeof(Foo<float>::_i) + sizeof(Foo<float>::_l)
		);
	}

	SECTION("offset is correct after write") {
		size_t offset = mmf::writer::writePacked(buffer, expected, 0, fe);
		REQUIRE(meta.first == offset);
	}

	SECTION("object is not actually equal before read") {
		auto actual = Foo<float>();
		REQUIRE(actual != expected);
	}

	SECTION("offset is correct after read") {
		auto actual = Foo<float>();
		// no need to write first
		size_t offset = mmf::reader::readPacked(buffer, actual, 0, fe);
		REQUIRE(meta.first == offset);
	}

	SECTION("data is preserved") {
		auto actual = Foo<float>();
		mmf::writer::writePacked(buffer, expected, 0, fe);
		mmf::reader::readPacked(buffer, actual, 0, fe);
		REQUIRE(actual == expected);
	}


}

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