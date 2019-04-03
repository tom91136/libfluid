#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_SIMD_AVX2
#define GLM_ENABLE_EXPERIMENTAL

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
static const char normal_x_[] = "normal.x";
static const char normal_y_[] = "normal.y";
static const char normal_z_[] = "normal.z";

template<typename N>
static inline auto triangleEntries() {
	return mmf::makeEntries(
			DECL_MEMBER(v0_x_, CLS(surface::Triangle<N>), v0.x),
			DECL_MEMBER(v0_y_, CLS(surface::Triangle<N>), v0.y),
			DECL_MEMBER(v0_z_, CLS(surface::Triangle<N>), v0.z),
			DECL_MEMBER(v1_x_, CLS(surface::Triangle<N>), v1.x),
			DECL_MEMBER(v1_y_, CLS(surface::Triangle<N>), v1.y),
			DECL_MEMBER(v1_z_, CLS(surface::Triangle<N>), v1.z),
			DECL_MEMBER(v2_x_, CLS(surface::Triangle<N>), v2.x),
			DECL_MEMBER(v2_y_, CLS(surface::Triangle<N>), v2.y),
			DECL_MEMBER(v2_z_, CLS(surface::Triangle<N>), v2.z)
//			ENTRY(normal_x_, CLS(surface::Triangle<N>), normal.x),
//			ENTRY(normal_y_, CLS(surface::Triangle<N>), normal.y),
//			ENTRY(normal_z_, CLS(surface::Triangle<N>), normal.z)
	);
}


static const char timestamp_[] = "timestamp";
static const char entries_[] = "entries";

struct Header {
	long timestamp;
	size_t entries;

	explicit Header(size_t entries) : timestamp(
			std::chrono::duration_cast<std::chrono::milliseconds>(
					std::chrono::system_clock::now().time_since_epoch()).count()
	), entries(entries) {}
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
void write_triangles(mio::mmap_sink &sink, const std::vector<surface::Triangle<N>> &xs) {
	Header header = Header(xs.size());
	size_t offset = mmf::writer::writePacked(sink, header, 0, headerEntries());
	for (const surface::Triangle<N> &t :  xs) {
		offset = mmf::writer::writePacked(sink, t, offset, triangleEntries<N>());
	}
}


mio::mmap_sink mkMmf(const std::string &path, const size_t length) {
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
	auto headerType = mmf::meta::writeMetaPacked<decltype(headerEntries())>();

	writeFile("header.json", headerType.second.dump(1));
	writeFile("particle.json", particleType.second.dump(1));
	writeFile("triangle.json", triangleType.second.dump(1));

	for (const auto &t : {particleType, triangleType, headerType}) {
		std::cout << "size=" << t.first << std::endl;
		std::cout << t.second.dump(3) << std::endl;
	}

	using namespace std::chrono;
	using hrc = high_resolution_clock;

	size_t cores = std::max<size_t>(1, std::thread::hardware_concurrency() / 2);
	omp_set_num_threads(4);
	std::cout << "OMP nCores: " << cores << std::endl;

	const size_t pcount = (64) * 1000;
	const size_t iter = 5000;
	const size_t solverIter = 5;
	const num_t scaling = 600; // less = less space between particle

	std::vector<fluid::Particle<size_t, num_t >> xs;
	size_t offset = 0;
//	offset = makeCube(offset, 28.f, pcount / 2, tvec3<num_t>(-500, -350, -250), xs);
//	offset = makeCube(offset, 28.f, pcount / 2, tvec3<num_t>(100, -350, -250), xs);

	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_int_distribution<std::mt19937::result_type> dist(1, 500);

	for (size_t i = 0; i < pcount; ++i) {
		xs.emplace_back(i, fluid::Fluid, 1.0,
		                tvec3<num_t>(dist(rng), dist(rng), dist(rng)),
		                tvec3<num_t>(0));
	}


	std::cout << "Mark" << std::endl;

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

	hrc::time_point start = hrc::now();


	auto min = tvec3<num_t>(-1000);
	auto max = tvec3<num_t>(1000, 1000, 1000);

	auto config = fluid::Config<num_t>(
			static_cast<num_t>(0.0083 * 1.66),
			scaling,
			solverIter,
			tvec3<num_t>(0, 9.8, 0),
			min, max);

	float i = 0;

	const std::vector<fluid::MeshCollider<num_t>> colliders = {
			fluid::MeshCollider<num_t>(
					{surface::Triangle<num_t>(
							tvec3<num_t>(0),
							tvec3<num_t>(0),
							tvec3<num_t>(0))})
	};

	for (size_t j = 0; j < iter; ++j) {
		i += glm::pi<num_t>() / 50;
		auto xx = (std::sin(i) * 350) * 1;
		auto zz = (std::cos(i) * 220) * 1;
		config.minBound = min + tvec3<num_t>(xx, 1, zz);
		config.maxBound = max + tvec3<num_t>(xx, 1, zz);

		hrc::time_point t1 = hrc::now();
		auto triangles = solver->advance(config, xs, colliders);
		hrc::time_point t2 = hrc::now();


		hrc::time_point s1 = hrc::now();


#ifdef DO_SURFACE


		num_t NR = 32.f;
		const num_t MBC = 100.f;
		const num_t MBCC = MBC * MBC;


		using namespace nanoflann;
		const ParticleCloud<size_t, num_t> cloud = ParticleCloud<size_t, num_t>(xs);
		typedef KDTreeSingleIndexAdaptor<
				L2_Simple_Adaptor<num_t, ParticleCloud<size_t, num_t> >,
				ParticleCloud<size_t, num_t>, 3> kd_tree_t;

		kd_tree_t index(3, cloud, KDTreeSingleIndexAdaptorParams(10));
		index.buildIndex();


		auto triangles = surface::parameterise<num_t>(100.f, lattice, [&index, &xs, MBCC, NR](
				const tvec3<num_t> &a) -> num_t {
			std::vector<std::pair<size_t, num_t> > results;

			nanoflann::SearchParams params;
			params.sorted = false;

			index.radiusSearch(&a[0], (NR * NR), results, params);

			num_t v = 0;
			for (const std::pair<size_t, num_t> &p : results)
				v += (MBCC /
					  glm::length2(xs[p.first].position - a)) * 2;
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

#endif
		hrc::time_point s2 = hrc::now();


		hrc::time_point mmt1 = hrc::now();
#ifdef DO_SURFACE
		write_triangles(mmfTSink, triangles);
#endif
		write_particles(mmfPSink, xs);
		write_triangles(mmfTSink, triangles);
		std::cout << "Trgs=" << triangles.size() << std::endl;

		hrc::time_point mmt2 = hrc::now();


		auto solve = duration_cast<nanoseconds>(t2 - t1).count();
		auto param = duration_cast<nanoseconds>(s2 - s1).count();
		auto mmf = duration_cast<nanoseconds>(mmt2 - mmt1).count();
		std::cout << "Iter" << j << "@ "
		          << "Solver:" << (solve / 1000000.0) << "ms "
		          << "Surface:" << (param / 1000000.0) << "ms "
		          << "IPC:" << (mmf / 1000000.0) << "ms "
		          << "Total= " << (solve + param + mmf) / 1000000.0 << "ms @"
		          << xs.size()
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

}