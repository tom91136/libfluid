
#include <memory>
#include <chrono>
#include <random>
#include <iostream>
#include <fstream>
#include "fluid.hpp"
#include "surface.hpp"
#include "boost/compute.hpp"
#include "mio/mmap.hpp"
#include "msgpack.hpp"


typedef glm::vec3 vec3t;
typedef float num_t;


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


void write_particles(mio::mmap_sink &sink, std::vector<fluid::Particle<size_t>> &xs) {
	using namespace std::chrono;
	size_t offset = 0;
	long init = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	offset = write_t(sink, init, offset);
	for (fluid::Particle<size_t> p :  xs) {
		offset = write_t(sink, p.t, offset);
		offset = write_t(sink, p.mass, offset);
		offset = write_t(sink, p.position.x, offset);
		offset = write_t(sink, p.position.y, offset);
		offset = write_t(sink, p.position.z, offset);
		offset = write_t(sink, p.velocity.x, offset);
		offset = write_t(sink, p.velocity.y, offset);
		offset = write_t(sink, p.velocity.z, offset);
	}
}


int main(int argc, char *argv[]) {
	using namespace std;
	using namespace std::chrono;

	namespace compute = boost::compute;
	compute::device deviceDevice = compute::system::default_device();
	std::cout << "OpenCL devices: " << std::endl;

	for (const auto &device : compute::system::devices()) {
		std::cout << "\t" << device.name() << std::endl;
	}
	std::cout << "\tDefault: " << deviceDevice.name() << std::endl;

	std::vector<fluid::Particle<size_t>> xs;

	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<num_t> dist(0, 500.f);

	size_t pcount = 10000;

	for (size_t i = 0; i < pcount; ++i) {
		xs.emplace_back(i, 1, vec3t(dist(mt), dist(mt), dist(mt)), vec3t(0));
	}

	std::vector<std::function<const fluid::Response(fluid::Ray &)> > colliders = {
			[](const fluid::Ray &x) -> fluid::Response {
				return fluid::Response(glm::clamp(x.getOrigin(), (num_t) 0.f, (num_t) 1000.f),
				                       x.getVelocity());
			}
	};

	std::unique_ptr<fluid::SphSolver<size_t>> solver(new fluid::SphSolver<size_t>(0.1, 500));

	std::cout << "Mark" << std::endl;


	std::string name = "ipc.mmf";
	std::ofstream file(name, ios::out | ios::trunc);
	string s(pcount * (sizeof(float) * 7 + sizeof(size_t)) + 8, ' ');
	file << s;

	std::error_code error;
	mio::mmap_sink rw_mmap = mio::make_mmap_sink(name, 0, mio::map_entire_file, error);
	if (error) {
		std::cout << "MMF failed:" << error.message() << std::endl;
		exit(1);
	} else {
		std::cout << "MMF size: " << rw_mmap.size() << std::endl;
	}

	std::cout << "Go" << std::endl;


	const std::vector<surface::Cell> &lattice = surface::createLattice(10, 0, 500, 0, 500, 0, 500);


	using hrc = high_resolution_clock;

	hrc::time_point start = hrc::now();
	for (size_t j = 0; j < 10; ++j) {

		hrc::time_point t1 = hrc::now();
		solver->advance(0.0083, 3, xs,
		                [](const fluid::Particle<size_t> &x) {
			                return vec3t(0, x.mass * 9.8, 0);
		                }, colliders
		);
		hrc::time_point t2 = hrc::now();

		hrc::time_point s1 = hrc::now();

//		std::vector<vec3t> pts;
//		std::transform(xs.begin(), xs.end(), std::back_inserter(pts),
//		               [](const fluid::Particle<size_t> &a) { return a.position; });
//		unibn::Octree<vec3t> octree;
//		octree.initialize(pts);

//		auto triangles = surface::parameterise(lattice, [&octree, &xs](const vec3t &a) -> num_t {
//			std::vector<uint32_t> results;
//			octree.radiusNeighbors<unibn::L2Distance<vec3t> >(a, 25.f, results);
//			num_t v = 0;
//			for (uint32_t &result : results)
//				v += ((100 * 100) / glm::length2(xs[result].position - a)) * 2;
//			return v;
//		});

		hrc::time_point s2 = hrc::now();


		hrc::time_point mmt1 = hrc::now();
//		write_particles(rw_mmap, xs);
		hrc::time_point mmt2 = hrc::now();


		auto solve = duration_cast<nanoseconds>(t2 - t1).count();
		auto param = duration_cast<nanoseconds>(s2 - s1).count();
		auto mmf = duration_cast<nanoseconds>(mmt2 - mmt1).count();
		std::cout << "Iter" << j << "@ "
		          << "solver:" << (solve / 1000000.0) << "ms "
		          << "param:" << (param / 1000000.0) << "ms "
		          << "mmf:" << (mmf / 1000000.0) << "ms "
		          << "Total= " << (solve + param + mmf) / 1000000.0 << "ms "
//		          << "Ts= " << triangles.size() << " "
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