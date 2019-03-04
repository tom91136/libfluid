
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_SIMD_AVX2
#define GLM_ENABLE_EXPERIMENTAL


#include <memory>
#include <chrono>
#include <random>
#include <iostream>
#include <fstream>
#include <experimental/filesystem>
#include "fluid.hpp"
#include "surface.hpp"
#include "cl.hpp"

#include "mio/mmap.hpp"
#include "omp.h"


typedef float num_t;
using glm::tvec3;

template<typename T>
size_t write_t(mio::mmap_sink &sink, T x, size_t offset) {
	union {
		T d;
		unsigned char bytes[sizeof(T)];
	} u{};
	u.d = x;
	for (size_t i = 0; i < sizeof(T); ++i)
		sink[offset + i] = u.bytes[i];
	return offset + sizeof(T);
}

template<typename T, typename N>
constexpr size_t probe_particle_size() {
	typedef fluid::Particle<T, N> PTN;
	return sizeof(PTN::t) + sizeof(PTN::type) + sizeof(PTN::mass) +
	       sizeof(PTN::position.x) + sizeof(PTN::position.y) + sizeof(PTN::position.z) +
	       sizeof(PTN::velocity.x) + sizeof(PTN::velocity.y) + sizeof(PTN::velocity.z);
}

template<typename N>
constexpr size_t probe_triangle_size() {
	typedef surface::Triangle<N> TN;
	return sizeof(TN::v0.x) + sizeof(TN::v0.y) + sizeof(TN::v0.z) +
	       sizeof(TN::v1.x) + sizeof(TN::v1.y) + sizeof(TN::v1.z) +
	       sizeof(TN::v2.x) + sizeof(TN::v2.y) + sizeof(TN::v2.z);
}


template<typename T, typename N>
void write_particles(mio::mmap_sink &sink, std::vector<fluid::Particle<T, N>> &xs) {
	using namespace std::chrono;
	size_t offset = 0;
	long init = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	offset = write_t(sink, init, offset);
	offset = write_t(sink, xs.size(), offset);
	for (const fluid::Particle<T, N> &p :  xs) {
		offset = write_t(sink, p.t, offset);
		offset = write_t(sink, p.type, offset);
		offset = write_t(sink, p.mass, offset);
		offset = write_t(sink, p.position.x, offset);
		offset = write_t(sink, p.position.y, offset);
		offset = write_t(sink, p.position.z, offset);
		offset = write_t(sink, p.velocity.x, offset);
		offset = write_t(sink, p.velocity.y, offset);
		offset = write_t(sink, p.velocity.z, offset);
	}
}

template<typename N>
void write_triangles(mio::mmap_sink &sink, std::vector<surface::Triangle<N>> &xs) {
	using namespace std::chrono;
	size_t offset = 0;
	long init = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	offset = write_t(sink, init, offset);
	offset = write_t(sink, xs.size(), offset);
	for (const surface::Triangle<N> &t :  xs) {
		offset = write_t(sink, t.v0.x, offset);
		offset = write_t(sink, t.v0.y, offset);
		offset = write_t(sink, t.v0.z, offset);
		offset = write_t(sink, t.v1.x, offset);
		offset = write_t(sink, t.v1.y, offset);
		offset = write_t(sink, t.v1.z, offset);
		offset = write_t(sink, t.v2.x, offset);
		offset = write_t(sink, t.v2.y, offset);
		offset = write_t(sink, t.v2.z, offset);
	}
}


mio::mmap_sink mkMmf(const std::experimental::filesystem::path &path, const size_t length) {
	const auto pathStr = path.string();
	std::ofstream file(pathStr, std::ios::out | std::ios::trunc);
	std::string s(length, ' ');
	file << s;

	std::error_code error;
	mio::mmap_sink sink = mio::make_mmap_sink(pathStr, 0, mio::map_entire_file, error);
	if (error) {
		std::cout << "MMF(" << path << ") failed:" << error.message() << std::endl;
		exit(1);
	} else {
		std::cout << "MMF(" << path << ") size: "
		          << (double) sink.size() / (1024 * 1024) << "MB" << std::endl;
	}
	return sink;
}

template<typename N>
size_t makeCube(size_t offset, N spacing, const size_t count,
                tvec3<N> origin,
                std::vector<fluid::Particle<size_t, num_t >> &xs) {

	auto len = static_cast<size_t>(std::cbrt(count));

	for (size_t x = 0; x < len / 2; ++x) {
		for (size_t y = 0; y < len; ++y) {
			for (size_t z = 0; z < len; ++z) {
				auto pos = (tvec3<N>(x, y, z) * spacing) + origin;
				xs.emplace_back(++offset, fluid::Fluid, 1.0, pos, tvec3<num_t>(0));
			}
		}
	}
	return offset;
}


void checkClNN3() {
	namespace compute = boost::compute;
	std::cout << "OpenCL devices: " << std::endl;

	for (const auto &device : compute::system::devices()) {
		std::cout << "\t" << device.name() << std::endl;
	}

	compute::device defaultDevice = compute::system::default_device();
	std::cout << "\tDefault: " << defaultDevice.name() << std::endl;

	auto clo = new CLOps(defaultDevice);


	std::random_device rd;
	std::mt19937 mt(12345);
	std::uniform_real_distribution<num_t> dist(0, 600.f);


	std::vector<tvec3<num_t >> data;
	for (int m = 0; m < 10000; ++m)
		data.emplace_back(dist(mt), dist(mt), dist(mt));


	unibn::Octree<tvec3<num_t>> octree;
	octree.initialize(data);

	// compile once
	clo->nn3<num_t>(30.f, 1000, data);

	using namespace std::chrono;
	using hrc = high_resolution_clock;

	hrc::time_point cls = hrc::now();
	clo->nn3<num_t>(30.f, 1000, data);
	hrc::time_point cle = hrc::now();


	hrc::time_point cpus = hrc::now();
	std::vector<std::vector<uint32_t >> drain(data.size());
#pragma omp parallel for
	for (size_t m = 0; m < data.size(); ++m)
		octree.radiusNeighbors<unibn::L2Distance<tvec3<num_t>>>(data[m], 30.f, drain[m]);

	hrc::time_point cpue = hrc::now();

	auto cl = duration_cast<nanoseconds>(cle - cls).count();
	auto cpu = duration_cast<nanoseconds>(cpue - cpus).count();

	std::cout << "CL  : " << (cl / 1000000.0) << "ms"
	          << "CPU : " << (cpu / 1000000.0) << "ms "
	          << drain.size() << std::endl;
}


//#define DO_SURFACE

int main(int argc, char *argv[]) {

	using namespace std::chrono;
	using hrc = high_resolution_clock;

	omp_set_num_threads(4);
	size_t pcount = 6000 * 2;
	size_t iter = 100;


	std::vector<fluid::Particle<size_t, num_t >> xs;
	size_t offset = 0;
	offset = makeCube(offset, 28.f, pcount / 2, tvec3<num_t>(-500, -350, -250), xs);
	offset = makeCube(offset, 28.f, pcount / 2, tvec3<num_t>(100, -350, -250), xs);

	float i = 0;

	std::vector<std::function<const fluid::Response<num_t>(
			fluid::Ray<num_t> &)> > colliders = {
			[&i](const fluid::Ray<num_t> &x) -> fluid::Response<num_t> {
				float d = (sin(i) * 200);
				return fluid::Response<num_t>(
						tvec3<num_t>(
								glm::clamp(x.getOrigin().x, (num_t) -500 + d,
								           (num_t) 500.f + d),
								glm::clamp(x.getOrigin().y, (num_t) -500, (num_t) 500.f),
								glm::clamp(x.getOrigin().z, (num_t) -500, (num_t) 500.f)),
						x.getVelocity());
			}
	};


	std::cout << "Mark" << std::endl;


	auto mmfPSink = mkMmf(std::experimental::filesystem::current_path() /= "particles.mmf",
	                      pcount * probe_particle_size<size_t, num_t>() + sizeof(long) * 2);

	auto mmfTSink = mkMmf(std::experimental::filesystem::current_path() /= "triangles.mmf",
	                      probe_triangle_size<num_t>() * 500000 + sizeof(long));

	std::cout << "Go" << std::endl;

	float D = 10.f;
	auto P = static_cast<size_t>(1000.f / D);

	const surface::MCLattice<num_t> &lattice = surface::createLattice<num_t>(P, P, P, -500, D);

	std::unique_ptr<fluid::SphSolver<size_t, num_t>> solver(
			new fluid::SphSolver<size_t, num_t>(0.1, 650)); // less = less space between particle

	using hrc = high_resolution_clock;

	hrc::time_point start = hrc::now();
	for (size_t j = 0; j < iter; ++j) {

		i += M_PI / 50;

		hrc::time_point t1 = hrc::now();
		solver->advance(static_cast<num_t> (0.0083 * 1.25), 2, xs,
		                [](const fluid::Particle<size_t, num_t> &x) {
			                return tvec3<num_t>(0, x.mass * 9.8, 0);
		                }, colliders
		);
		hrc::time_point t2 = hrc::now();


		hrc::time_point s1 = hrc::now();


#ifdef DO_SURFACE

		std::vector<tvec3<num_t>> pts;
		std::transform(xs.begin(), xs.end(), std::back_inserter(pts),
		               [](const fluid::Particle<size_t, num_t> &a) { return a.position; });


		unibn::Octree<tvec3<num_t>> octree;
		octree.initialize(pts);

		auto triangles = surface::parameterise<num_t>(100.f,
		                                              lattice, [&octree, &xs](
						const tvec3<num_t> &a) -> num_t {
					std::vector<uint32_t> results;
					octree.radiusNeighbors<unibn::L2Distance<tvec3<num_t>>>(a, 35.f, results);
					num_t v = 0;
					for (uint32_t &result : results)
						v += ((100 * 100) /
						      glm::length2(xs[result].position - a)) * 2;
					return v;
				});
		std::cout <<
		          "\tLattice   = " << lattice.size() <<
		          "\n\tTriangles = " << triangles.size() <<
		          "\n\tParticles = " << xs.size() <<
		          std::endl;

//		for (auto t : triangles) {
//			std::cout << "[" << t << "]" << std::endl;
//		}
		write_triangles(mmfTSink, triangles);

#endif
		hrc::time_point s2 = hrc::now();


		hrc::time_point mmt1 = hrc::now();
		write_particles(mmfPSink, xs);
		hrc::time_point mmt2 = hrc::now();

//		sleep(1);


		auto solve = duration_cast<nanoseconds>(t2 - t1).count();
		auto param = duration_cast<nanoseconds>(s2 - s1).count();
		auto mmf = duration_cast<nanoseconds>(mmt2 - mmt1).count();
		std::cout << "Iter" << j << "@ "
		          << "Solver:" << (solve / 1000000.0) << "ms "
		          << "Surface:" << (param / 1000000.0) << "ms "
		          << "IPC:" << (mmf / 1000000.0) << "ms "
		          << "Total= " << (solve + param + mmf) / 1000000.0 << "ms "
		          << std::endl;
	}
	hrc::time_point end = hrc::now();
	auto elapsed = duration_cast<milliseconds>(end - start).count();
	std::cout << "Done: " << elapsed << "ms @ " << std::endl;

//
//	for (auto x : xs) {
//		std::cout << "[" << x << "]" << std::endl;
//	}

	xs.clear();

	return EXIT_SUCCESS;
}