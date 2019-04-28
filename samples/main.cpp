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

#include "structures.hpp"

#include <condition_variable>
#include <atomic>


#include "omp.h"

using json = nlohmann::json;

typedef float num_t;
using glm::tvec3;


template<typename N>
size_t makeCube(size_t offset, N spacing, const size_t count,
                tvec3<N> origin,
                std::vector<fluid::Particle<size_t, num_t >> &xs) {

	auto len = static_cast<size_t>(std::cbrt(count));

	for (size_t x = 0; x < len; ++x) {
		for (size_t y = 0; y < len; ++y) {
			for (size_t z = 0; z < len; ++z) {
				auto pos = (tvec3<N>(x, y, z) * spacing) + origin;
				xs.emplace_back(offset++, fluid::Fluid, 1.0, pos, tvec3<num_t>(0), 0xFF0000FF);
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


cl::Device resolveDeviceVerbose(const std::vector<std::string> &signatures) {
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

	return found.front();
}

void run() {

	using mmf::meta::writeMetaPacked;


	auto headerType = writeMetaPacked<decltype(strucures::headerDef())>();

	auto wellType = writeMetaPacked<decltype(strucures::wellDef<num_t>())>();
	auto sourceType = writeMetaPacked<decltype(strucures::sourceDef<num_t>())>();
	auto drainType = writeMetaPacked<decltype(strucures::drainDef<num_t>())>();
	auto sceneMetaType = writeMetaPacked<decltype(strucures::sceneMetaDef<num_t>())>();

	auto particleType = writeMetaPacked<decltype(strucures::particleDef<size_t, num_t>())>();
	auto triangleType = writeMetaPacked<decltype(strucures::triangleDef<num_t>())>();
	auto meshTriangleType = writeMetaPacked<decltype(strucures::meshTriangleDef<num_t>())>();


	const auto defs = json({
			                       {"header",       headerType.second},

			                       {"well",         wellType.second},
			                       {"source",       sourceType.second},
			                       {"drain",        drainType.second},
			                       {"sceneMeta",    sceneMetaType.second},

			                       {"particle",     particleType.second},
			                       {"triangle",     triangleType.second},
			                       {"meshTriangle", meshTriangleType.second}
	                       });

	writeFile("defs.json", defs.dump(1));

	std::cout << defs.dump(3) << std::endl;

	using namespace std::chrono;
	using hrc = high_resolution_clock;

	size_t cores = std::max<size_t>(1, std::thread::hardware_concurrency() / 2);
	omp_set_num_threads(cores);
	std::cout << "OMP nCores: " << cores << std::endl;

	const size_t pcount = 32 * 1000;
	const size_t iter = 30000;
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

	using miommf::createSink;
	using miommf::createSource;

	std::cout << "Creating mmf sink+source" << std::endl;

	auto sceneSource = createSource("scene.mmf");
	auto colliderSource = createSource("colliders.mmf");

	auto particleSink = createSink("particles.mmf",
	                               pcount * 10 * particleType.first + headerType.first);
	auto triangleSink = createSink("triangles.mmf",
	                               500000 * 10 * triangleType.first + headerType.first);

	std::cout << "Go!" << std::endl;


#ifdef _WIN32
	const auto kernelPaths = "C:\\Users\\Tom\\libfluid\\include\\fluid\\";
#elif __APPLE__
	const auto kernelPaths = "/Users/tom/libfluid/include/fluid/";
#elif defined(__linux__) || defined(__unix__) || defined(_POSIX_VERSION)
	const auto kernelPaths = "/home/tom/libfluid/include/fluid/";
#else
#   error "Unknown compiler"
#endif

	const std::vector<std::string> signatures = {
			"Oclgrind", "gfx906", "Ellesmere", "Quadro", "ATI", "1050", "1080", "980", "NEO",
			"Tesla"};
	clutil::enumeratePlatformToCout();
	const auto device = resolveDeviceVerbose(signatures);

	// std::unique_ptr<fluid::SphSolver<size_t, num_t>> solver(new cpu::SphSolver<size_t, num_t>(0.1));
	std::unique_ptr<fluid::SphSolver<size_t, num_t>> solver(new ocl::SphSolver(
			0.1,
			kernelPaths,
			device));

	using hrc = high_resolution_clock;


	auto min = tvec3<num_t>(-1000);
	auto max = tvec3<num_t>(2000, 2000, 2000);


	float i = 0;


	std::mutex m;
	std::condition_variable flush;

	strucures::Scene<num_t> scene(
			strucures::SceneMeta<num_t>(
					false, false,
					solverIter, 1.5, scaling, 1.f, 9.8f,
					tvec3<num_t>(0, 0, 0),
					tvec3<num_t>(1000, 1000, 1000)),
			{}, {}, {});

	std::atomic_bool ready(false);
	std::atomic_bool copied(false);
	std::atomic_bool terminate(false);

	std::vector<fluid::Particle<size_t, num_t >> particles(prepared);
	std::vector<geometry::MeshTriangle<num_t>> triangles;


	std::thread mmfXferThread([&flush, &m, &ready, &copied, &terminate,
			                          &sceneSource, &particleSink, &triangleSink,
			                          &particles, &triangles, &scene] {
		std::cout << "Xfer thread init" << std::endl;
		while (!terminate.load()) {

			hrc::time_point waitStart = hrc::now();
			std::unique_lock<std::mutex> lock(m);
			flush.wait(lock, [&ready] { return ready.load(); });
			ready = false;
			const std::vector<fluid::Particle<size_t, num_t >> particlesBuffer = particles;
			const std::vector<geometry::MeshTriangle<num_t>> trianglesBuffer = triangles;

			copied = true;
			lock.unlock();
			flush.notify_one();


			hrc::time_point waitEnd = hrc::now();

			hrc::time_point sceneStart = hrc::now();
			int suspendTick = 0;

			if (miommf::canRead(sceneSource)) {
				strucures::Scene<num_t> sceneBuffer = strucures::readScene<num_t>(sceneSource);
//				std::cout << sceneBuffer << "\n";
//				for (const auto &x : sceneBuffer.wells) std::cout << x << "\n";
//				for (const auto &x : sceneBuffer.sources) std::cout << x << "\n";
//				for (const auto &x : sceneBuffer.drains) std::cout << x << "\n";
				while (sceneBuffer.meta.suspend) {
					suspendTick++;
					std::this_thread::sleep_for(std::chrono::microseconds(1));
					strucures::readSceneMeta<num_t>(sceneSource, sceneBuffer.meta);
				}
				if (sceneBuffer.meta.solverIter != 0) scene = sceneBuffer; // TODO check
			}


			hrc::time_point sceneEnd = hrc::now();


			hrc::time_point xferStart = hrc::now();
//			strucures::writeParticles(particleSink, particlesBuffer);
			strucures::writeTriangles(triangleSink, trianglesBuffer);
			hrc::time_point xferEnd = hrc::now();

			auto solve = duration_cast<nanoseconds>(xferEnd - xferStart).count();
			auto wait = duration_cast<nanoseconds>(waitEnd - waitStart).count();
			auto sceneRead = duration_cast<nanoseconds>(sceneEnd - sceneStart).count();
#ifdef DEBUG
			std::cout << "\tXfer: " << (solve / 1000000.0) << "ms" <<
			          " (waited " << (wait / 1000000.0) << "ms)" <<
			          " Scene read " << (sceneRead / 1000000.0) << "ms (" << suspendTick
			          << " ticks)" <<
			          " nTriangle:" << trianglesBuffer.size() <<
			          " nParticle:" << particlesBuffer.size()
			          << std::endl;
#endif
		}
	});

	hrc::time_point start = hrc::now();
	for (size_t j = 0; j < iter; ++j) {
//		i += glm::pi<num_t>() / 50;
//		auto xx = (std::sin(i) * 350) * 0;
//		auto zz = (std::cos(i) * 500) * 0;
//		min + tvec3<num_t>(xx, 1, zz);
//		max + tvec3<num_t>(xx, 1, zz);

		auto config = fluid::Config<num_t>(
				static_cast<num_t>(0.0083 * scene.meta.solverStep),
				scene.meta.solverScale,
				scene.meta.surfaceRes,
				100,
				scene.meta.solverIter,
				tvec3<num_t>(0, scene.meta.gravity, 0),
				// copy , gets modified in xfer thread
				std::vector<fluid::Well<float>>(scene.wells),
				std::vector<fluid::Source<float>>(scene.sources),
				std::vector<fluid::Drain<float>>(scene.drains),
				scene.meta.minBound, scene.meta.maxBound
		);

//		std::cout << "Read collider: start" << std::endl;
//		auto collider = strucures::readCollider<num_t>(colliderSource);
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


		std::unique_lock<std::mutex> lock(m);
		flush.wait(lock, [&copied] { return copied.load(); });
		copied = false;

		auto solve = duration_cast<nanoseconds>(solveEnd - solveStart).count();
#ifdef DEBUG
		std::cout << "[" << j << "]" <<
		          " Solver:" << (solve / 1000000.0) << "ms " <<
		          " nTriangle:" << triangles.size() <<
		          " nParticle:" << particles.size()
		          << std::endl;
#endif

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
