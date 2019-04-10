#define GLM_FORCE_SIMD_AVX2
#define GLM_ENABLE_EXPERIMENTAL
#ifdef __INTEL_COMPILER
#define GLM_FORCE_PURE
#else
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#endif

#include <memory>
#include <chrono>
#include <random>
#include <iostream>
#include <fstream>
#include <thread>

#include "fluid/fluid.hpp"
#include "fluid/surface.hpp"
#include "fluid/oclsph.hpp"
#include "fluid/cpusph.hpp"
#include "fluid/mmf.hpp"
#include <condition_variable>
#include <atomic>


#include "omp.h"

using json = nlohmann::json;

typedef float num_t;
using glm::tvec3;


static const char id_[] = "id";
static const char type_[] = "type";
static const char mass_[] = "mass";
static const char position_x_[] = "position.x";
static const char position_y_[] = "position.y";
static const char position_z_[] = "position.z";
static const char velocity_x_[] = "velocity.x";
static const char velocity_y_[] = "velocity.y";
static const char velocity_z_[] = "velocity.z";

template<typename T, typename N>
static inline auto particleEntries() {
	return mmf::makeEntries(
			DECL_MEMBER(id_, CLS(fluid::Particle<T, N>), id),
			DECL_MEMBER(type_, CLS(fluid::Particle<T, N>), type),
			DECL_MEMBER(mass_, CLS(fluid::Particle<T, N>), mass),
			DECL_MEMBER(position_x_, CLS(fluid::Particle<T, N>), position.x),
			DECL_MEMBER(position_y_, CLS(fluid::Particle<T, N>), position.y),
			DECL_MEMBER(position_z_, CLS(fluid::Particle<T, N>), position.z),
			DECL_MEMBER(velocity_x_, CLS(fluid::Particle<T, N>), velocity.x),
			DECL_MEMBER(velocity_y_, CLS(fluid::Particle<T, N>), velocity.y),
			DECL_MEMBER(velocity_z_, CLS(fluid::Particle<T, N>), velocity.z)
	);
}

static const char v0_x_[] = "v0.x";
static const char v0_y_[] = "v0.y";
static const char v0_z_[] = "v0.z";
static const char v1_x_[] = "v1.x";
static const char v1_y_[] = "v1.y";
static const char v1_z_[] = "v1.z";
static const char v2_x_[] = "v2.x";
static const char v2_y_[] = "v2.y";
static const char v2_z_[] = "v2.z";
static const char n0_x_[] = "n0.x";
static const char n0_y_[] = "n0.y";
static const char n0_z_[] = "n0.z";
static const char n1_x_[] = "n1.x";
static const char n1_y_[] = "n1.y";
static const char n1_z_[] = "n1.z";
static const char n2_x_[] = "n2.x";
static const char n2_y_[] = "n2.y";
static const char n2_z_[] = "n2.z";


template<typename N>
static inline auto triangleEntries() {
	return mmf::makeEntries(
			DECL_MEMBER(v0_x_, CLS(geometry::Triangle<N>), v0.x),
			DECL_MEMBER(v0_y_, CLS(geometry::Triangle<N>), v0.y),
			DECL_MEMBER(v0_z_, CLS(geometry::Triangle<N>), v0.z),
			DECL_MEMBER(v1_x_, CLS(geometry::Triangle<N>), v1.x),
			DECL_MEMBER(v1_y_, CLS(geometry::Triangle<N>), v1.y),
			DECL_MEMBER(v1_z_, CLS(geometry::Triangle<N>), v1.z),
			DECL_MEMBER(v2_x_, CLS(geometry::Triangle<N>), v2.x),
			DECL_MEMBER(v2_y_, CLS(geometry::Triangle<N>), v2.y),
			DECL_MEMBER(v2_z_, CLS(geometry::Triangle<N>), v2.z)
	);
}

template<typename N>
static inline auto meshTriangleEntries() {
	return mmf::makeEntries(
			DECL_MEMBER(v0_x_, CLS(geometry::MeshTriangle<N>), v0.x),
			DECL_MEMBER(v0_y_, CLS(geometry::MeshTriangle<N>), v0.y),
			DECL_MEMBER(v0_z_, CLS(geometry::MeshTriangle<N>), v0.z),
			DECL_MEMBER(v1_x_, CLS(geometry::MeshTriangle<N>), v1.x),
			DECL_MEMBER(v1_y_, CLS(geometry::MeshTriangle<N>), v1.y),
			DECL_MEMBER(v1_z_, CLS(geometry::MeshTriangle<N>), v1.z),
			DECL_MEMBER(v2_x_, CLS(geometry::MeshTriangle<N>), v2.x),
			DECL_MEMBER(v2_y_, CLS(geometry::MeshTriangle<N>), v2.y),
			DECL_MEMBER(v2_z_, CLS(geometry::MeshTriangle<N>), v2.z),
			DECL_MEMBER(n0_x_, CLS(geometry::MeshTriangle<N>), n0.x),
			DECL_MEMBER(n0_y_, CLS(geometry::MeshTriangle<N>), n0.y),
			DECL_MEMBER(n0_z_, CLS(geometry::MeshTriangle<N>), n0.z),
			DECL_MEMBER(n1_x_, CLS(geometry::MeshTriangle<N>), n1.x),
			DECL_MEMBER(n1_y_, CLS(geometry::MeshTriangle<N>), n1.y),
			DECL_MEMBER(n1_z_, CLS(geometry::MeshTriangle<N>), n1.z),
			DECL_MEMBER(n2_x_, CLS(geometry::MeshTriangle<N>), n2.x),
			DECL_MEMBER(n2_y_, CLS(geometry::MeshTriangle<N>), n2.y),
			DECL_MEMBER(n2_z_, CLS(geometry::MeshTriangle<N>), n2.z)
	);
}


static const char timestamp_[] = "timestamp";
static const char entries_[] = "entries";

struct Header {
	long timestamp;
	size_t entries;


	Header() {}
	explicit Header(size_t entries) : timestamp(
			std::chrono::duration_cast<std::chrono::milliseconds>(
					std::chrono::system_clock::now().time_since_epoch()).count()
	), entries(entries) {}
	friend std::ostream &operator<<(std::ostream &os, const Header &header) {
		return os
				<< "Header(timestamp=" << header.timestamp << ", entries=" << header.entries << ")";
	}
};

static inline auto headerEntries() {
	return mmf::makeEntries(
			DECL_MEMBER(timestamp_, CLS(Header), timestamp),
			DECL_MEMBER(entries_, CLS(Header), entries)
	);
}

template<typename T, typename N>
void write_particles(mio::mmap_sink &sink, const std::vector<fluid::Particle<T, N>> &xs) {

	Header header = Header(xs.size());
	size_t offset = mmf::writer::writePacked(sink, header, 0, headerEntries());
	for (const fluid::Particle<T, N> &p :  xs) {
		offset = mmf::writer::writePacked(sink, p, offset, particleEntries<T, N>());
	}
}

template<typename N>
void write_triangles(mio::mmap_sink &sink, const std::vector<surface::MeshTriangle<N>> &xs) {
	Header header = Header(xs.size());
	size_t offset = mmf::writer::writePacked(sink, header, 0, headerEntries());
	for (const surface::MeshTriangle<N> &t :  xs) {
		offset = mmf::writer::writePacked(sink, t, offset, meshTriangleEntries<N>());
	}
}

template<typename N>
fluid::MeshCollider<N> readCollider(mio::mmap_source &source) {
	Header header;
	size_t offset = mmf::reader::readPacked(source, header, 0, headerEntries());
	std::cout << header << "\n";
	std::vector<geometry::Triangle<N>> ts(header.entries);
	for (size_t i = 0; i < header.entries; ++i) {
		geometry::Triangle<N> t{};
		offset = mmf::reader::readPacked(source, t, offset, triangleEntries<N>());
		ts[i] = t;
	}
	return fluid::MeshCollider<N>(ts);
}


mio::mmap_source openMmf(const std::string &path) {

	std::error_code error;

//	mio::mmap_source source;
//	source.map(path, error);


	mio::mmap_source source = mio::make_mmap_source(path, error);
	if (error) {
		std::cout << "MMF(" << path << ") failed:" << error.message() << std::endl;
		exit(1);
	} else {
		std::cout << "MMF(" << path << ") size: "
		          << (double) source.size() / (1024 * 1024) << "MB" << std::endl;
	}
	return source;
}

mio::mmap_sink mkMmf(const std::string &path, const size_t length) {
	std::ofstream file(path, std::ios::out | std::ios::trunc);
	std::string s(std::max<size_t>(1024 * 4, length), ' '); // XXX may fail if than 4k
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

template<typename N>
size_t makeCube(size_t offset, N spacing, const size_t count,
                tvec3<N> origin,
                std::vector<fluid::Particle<size_t, num_t >> &xs) {

	auto len = static_cast<size_t>(std::cbrt(count));

	for (size_t x = 0; x < len; ++x) {
		for (size_t y = 0; y < len; ++y) {
			for (size_t z = 0; z < len; ++z) {
				auto pos = (tvec3<N>(x, y, z) * spacing) + origin;
				xs.emplace_back(offset++, fluid::Fluid, 1.0, pos, tvec3<num_t>(0));
			}
		}
	}
	return offset;
}


template<typename T, typename N>
struct ParticleCloud {
	const std::vector<fluid::Particle<T, N>> &pts;

	explicit ParticleCloud(const std::vector<fluid::Particle<T, N>> &pts) : pts(pts) {}

	inline size_t kdtree_get_point_count() const { return pts.size(); }

	inline N kdtree_get_pt(const size_t idx, const size_t dim) const {
		return pts[idx].position[dim];
	}

	template<class BBOX>
	bool kdtree_get_bbox(BBOX &) const { return false; }

};


//#define DO_SURFACE


void writeFile(const std::string &filename, const std::string &content) {
	std::ofstream file;
	file.open(filename);
	file << content;
	file.close();
}

void run();

int main(int argc, char *argv[]) {
	run();
	return EXIT_SUCCESS;
}


void run() {

	auto particleType = mmf::meta::writeMetaPacked<decltype(particleEntries<size_t, num_t>())>();
	auto triangleType = mmf::meta::writeMetaPacked<decltype(triangleEntries<num_t>())>();
	auto meshTriangleType = mmf::meta::writeMetaPacked<decltype(meshTriangleEntries<num_t>())>();
	auto headerType = mmf::meta::writeMetaPacked<decltype(headerEntries())>();

	writeFile("header.json", headerType.second.dump(1));
	writeFile("particle.json", particleType.second.dump(1));
	writeFile("triangle.json", triangleType.second.dump(1));
	writeFile("mesh_triangle.json", meshTriangleType.second.dump(1));

	for (const auto &t : {headerType, particleType, triangleType, meshTriangleType}) {
		std::cout << "size=" << t.first << std::endl;
		std::cout << t.second.dump(3) << std::endl;
	}

	using namespace std::chrono;
	using hrc = high_resolution_clock;

	size_t cores = std::max<size_t>(1, std::thread::hardware_concurrency() / 2);
	omp_set_num_threads(cores);
	std::cout << "OMP nCores: " << cores << std::endl;

	const size_t pcount = (64) * 1000;
	const size_t iter = 5000;
	const size_t solverIter = 5;
	const num_t scaling = 1000; // less = less space between particle

	std::vector<fluid::Particle<size_t, num_t >> prepared;
	size_t offset = 0;
	offset = makeCube(offset, 28.f, pcount, tvec3<num_t>(-500, -350, -250), prepared);
//	offset = makeCube(offset, 28.f, pcount / 2, tvec3<num_t>(100, -350, -250), xs);

	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_int_distribution<std::mt19937::result_type> dist(1, 500);

//	for (size_t i = 0; i < pcount; ++i) {
//		xs.emplace_back(i, fluid::Fluid, 1.0,
//		                tvec3<num_t>(dist(rng), dist(rng), dist(rng)),
//		                tvec3<num_t>(0));
//	}


	std::cout << "Mark" << std::endl;
	auto mmfCSink = openMmf("colliders.mmf");
	auto mmfPSink = mkMmf("particles.mmf", pcount * particleType.first + headerType.first);
	auto mmfTSink = mkMmf("triangles.mmf", 500000 * 10 * triangleType.first + headerType.first);

	std::cout << "Go" << std::endl;

	float D = 15.f;
	auto P = static_cast<size_t>(2000.f / D);

//	const surface::MCLattice<num_t> &lattice = surface::createLattice<num_t>(P, P, P, -1000, D);



#ifdef _WIN32
	const auto kernelPaths = "C:\\Users\\Tom\\libfluid\\include\\fluid\\";
#elif __APPLE__
	const auto kernelPaths = "/Users/tom/libfluid/include/fluid/";
#elif defined(__linux__) || defined(__unix__) || defined(_POSIX_VERSION)
	const auto kernelPaths = "/home/tom/libfluid/include/fluid/";
#else
#   error "Unknown compiler"
#endif


	clutil::enumeratePlatformToCout();


	const std::vector<std::string> signatures = {
			"Ellesmere", "Quadro", "ATI", "1050", "980", "NEO", "Tesla"};
	const auto imploded = clutil::mkString<std::string>(signatures, [](auto x) { return x; });
	auto found = clutil::findDeviceWithSignature({signatures,});

	if (found.empty()) {
		throw std::runtime_error("No CL device found with signature:`" + imploded + "`");
	}

	std::cout << "Matching devices(" << found.size() << "):" << std::endl;
	for (const auto &d : found) std::cout << "\t" << d.getInfo<CL_DEVICE_NAME>() << std::endl;

	if (found.size() > 1) {
		std::cout << "Found more than one device signature:`" << imploded
		          << "`"  ", using the first one." << std::endl;
	}

	const auto device = found.front();



	// std::unique_ptr<fluid::SphSolver<size_t, num_t>> solver(new cpu::SphSolver<size_t, num_t>(0.1));
	std::unique_ptr<fluid::SphSolver<size_t, num_t>> solver(new ocl::SphSolver(
			0.1,
			kernelPaths,
			device));

	using hrc = high_resolution_clock;


	auto min = tvec3<num_t>(-1000);
	auto max = tvec3<num_t>(2000, 2000, 2000);

	auto config = fluid::Config<num_t>(
			static_cast<num_t>(0.0083 * 1.5),
			scaling,
			solverIter,
			tvec3<num_t>(0, 9.8, 0),
			min, max);

	float i = 0;




	std::mutex m;
	std::condition_variable flush;
	std::atomic_bool ready(false);
	std::atomic_bool terminate(false);

	std::vector<fluid::Particle<size_t, num_t >> particles(prepared);
	std::vector<geometry::MeshTriangle<num_t>> triangles;


	std::thread mmfXferThread([&flush, &m, &ready, &terminate,
			                          &mmfPSink, &mmfTSink, &particles, &triangles] {
		std::cout << "Xfer thread init" << std::endl;
		while (!terminate.load()) {

			hrc::time_point waitStart = hrc::now();
			std::unique_lock<std::mutex> lock(m);
			flush.wait(lock, [&ready] { return ready.load(); });
			ready = false;
			hrc::time_point waitEnd = hrc::now();

			hrc::time_point xferStart = hrc::now();
			write_particles(mmfPSink, particles);
			write_triangles(mmfTSink, triangles);
			hrc::time_point xferEnd = hrc::now();

			auto solve = duration_cast<nanoseconds>(xferEnd - xferStart).count();
			auto wait = duration_cast<nanoseconds>(waitEnd - waitStart).count();
			std::cout << "\tXfer: " << (solve / 1000000.0) << "ms" <<
			          " (waited " << (wait / 1000000.0) << "ms)" <<
			          " nTriangle:" << triangles.size() <<
			          " nParticle:" << particles.size()
			          << std::endl;
			lock.unlock();
			flush.notify_one();
		}
	});

	hrc::time_point start = hrc::now();
	for (size_t j = 0; j < iter; ++j) {
		i += glm::pi<num_t>() / 50;
		auto xx = (std::sin(i) * 350) * 1;
		auto zz = (std::cos(i) * 500) * 1;
		config.minBound = min + tvec3<num_t>(xx, 1, zz);
		config.maxBound = max + tvec3<num_t>(xx, 1, zz);


//		std::cout << "Read collider: start" << std::endl;
//		auto collider = readCollider<num_t>(mmfCSink);
//		std::cout << "Read collider: end" << std::endl;

		const std::vector<fluid::MeshCollider<num_t>> colliders = {
//				collider,

//			fluid::MeshCollider<num_t>(
//					{
//							surface::MeshTriangle<num_t>(
//									tvec3<num_t>(0, 0, -600),
//									tvec3<num_t>(0, 0, 600),
//									tvec3<num_t>(300, 1200, 0)),
//
//							surface::MeshTriangle<num_t>(
//									tvec3<num_t>(0, 0, -600),
//									tvec3<num_t>(0, 0, 600),
//									tvec3<num_t>(-300, 1200, 0))
//
//					})
		};

		hrc::time_point solveStart = hrc::now();
		triangles = solver->advance(config, particles, colliders);
		hrc::time_point solveEnd = hrc::now();
		ready = true;
		flush.notify_one();

		auto solve = duration_cast<nanoseconds>(solveEnd - solveStart).count();
		std::cout << "[" << j << "]" <<
		          " Solver:" << (solve / 1000000.0) << "ms " <<
		          " nTriangle:" << triangles.size() <<
		          " nParticle:" << particles.size()
		          << std::endl;

	}

	hrc::time_point end = hrc::now();
	auto elapsed = duration_cast<milliseconds>(end - start).count();
	std::cout << "Done: " << elapsed << "ms @ " << iter << std::endl;
	terminate = true;
	flush.notify_one();
	mmfXferThread.join();
	std::cout << "Xfer thread joined" << std::endl;
	prepared.clear();
}
