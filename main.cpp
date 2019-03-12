
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_SIMD_AVX2
#define GLM_ENABLE_EXPERIMENTAL

#define FLANN_S
#define FLANN

#include <memory>
#include <chrono>
#include <random>
#include <iostream>
#include <fstream>
//#include <experimental/filesystem>
#include "fluid.hpp"
#include "surface.hpp"
#include "cl.hpp"
#include "nanoflann.hpp"

#include "mio/mmap.hpp"
#include "omp.h"
#include <nlohmann/json.hpp>
#include "writer.hpp"

using json = nlohmann::json;

typedef float num_t;
using glm::tvec3;


static const char t_[] = "t";
static const char type_[] = "type";
static const char mass_[] = "mass";
static const char position_x_[] = "position.x";
static const char position_y_[] = "position.y";
static const char position_z_[] = "position.z";
static const char velocity_x_[] = "velocity.x";
static const char velocity_y_[] = "velocity.y";
static const char velocity_z_[] = "velocity.z";

template<typename T, typename N>
const static inline auto particleEntries() {
	return writer::makeEntries(
			ENTRY(t_, CLS(fluid::Particle<T, N>), t),
			ENTRY(type_, CLS(fluid::Particle<T, N>), type),
			ENTRY(mass_, CLS(fluid::Particle<T, N>), mass),
			ENTRY(position_x_, CLS(fluid::Particle<T, N>), position.x),
			ENTRY(position_y_, CLS(fluid::Particle<T, N>), position.y),
			ENTRY(position_z_, CLS(fluid::Particle<T, N>), position.z),
			ENTRY(velocity_x_, CLS(fluid::Particle<T, N>), velocity.x),
			ENTRY(velocity_y_, CLS(fluid::Particle<T, N>), velocity.y),
			ENTRY(velocity_z_, CLS(fluid::Particle<T, N>), velocity.z)
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

template<typename N>
const static inline auto triangleEntries() {
	return writer::makeEntries(
			ENTRY(v0_x_, CLS(surface::Triangle<N>), v0.x),
			ENTRY(v0_y_, CLS(surface::Triangle<N>), v0.y),
			ENTRY(v0_z_, CLS(surface::Triangle<N>), v0.z),
			ENTRY(v1_x_, CLS(surface::Triangle<N>), v1.x),
			ENTRY(v1_y_, CLS(surface::Triangle<N>), v1.y),
			ENTRY(v1_z_, CLS(surface::Triangle<N>), v1.z),
			ENTRY(v2_x_, CLS(surface::Triangle<N>), v2.x),
			ENTRY(v2_y_, CLS(surface::Triangle<N>), v2.y),
			ENTRY(v2_z_, CLS(surface::Triangle<N>), v2.z)
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

const static inline auto headerEntries() {
	return writer::makeEntries(
			ENTRY(timestamp_, CLS(Header), timestamp),
			ENTRY(entries_, CLS(Header), entries)
	);
}

template<typename T, typename N>
void write_particles(mio::mmap_sink &sink, std::vector<fluid::Particle<T, N>> &xs) {

	Header header = Header(xs.size());
	size_t offset = writer::writePacked(sink, header, 0, headerEntries());
	for (const fluid::Particle<T, N> &p :  xs) {
		offset = writer::writePacked(sink, p, offset, particleEntries<T, N>());
	}
}

template<typename N>
void write_triangles(mio::mmap_sink &sink, std::vector<surface::Triangle<N>> &xs) {
	Header header = Header(xs.size());
	size_t offset = writer::writePacked(sink, header, 0, headerEntries());
	for (const surface::Triangle<N> &t :  xs) {
		offset = writer::writePacked(sink, t, offset, triangleEntries<N>());
	}
}


mio::mmap_sink mkMmf(const std::string &path, const size_t length) {
	const auto pathStr = path;//.string();
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


template<typename T, typename N>
struct ParticleCloud {
	const std::vector<fluid::Particle<T, N>> &pts;

	ParticleCloud(const std::vector<fluid::Particle<T, N>> &pts) : pts(pts) {}

	inline size_t kdtree_get_point_count() const { return pts.size(); }

	inline N kdtree_get_pt(const size_t idx, const size_t dim) const {
		return pts[idx].position[dim];
	}

	template<class BBOX>
	bool kdtree_get_bbox(BBOX &) const { return false; }

};

template<typename N>
struct PointCloud {
	const std::vector<tvec3<N>> pts;
	const N scale;

	PointCloud(const std::vector<tvec3<N>> &pts, N scale) : pts(pts), scale(scale) {}

	inline size_t kdtree_get_point_count() const { return pts.size(); }

	inline N kdtree_get_pt(const size_t idx, const size_t dim) const {
		return pts[idx][dim];
	}

	template<class BBOX>
	bool kdtree_get_bbox(BBOX & /* bb */) const { return false; }

};


void checkClNN3() {
	namespace compute = boost::compute;
	std::cout << "OpenCL devices: " << std::endl;

	for (const auto &device : compute::system::devices()) {
		std::cout << "\t" << device.name() << std::endl;


		auto clo = new CLOps(device);
		clo->doIt();

		for (int j = 0; j < 10; ++j) {

			std::random_device rd;
			std::mt19937 mt(12345);
			num_t scale = 10.f;

			std::uniform_real_distribution<num_t> dist(0, 1.f);
			num_t R = 0.1f;

			std::vector<tvec3<num_t >> data;
			for (int m = 0; m < 20000; ++m)
				data.emplace_back(dist(mt), dist(mt), dist(mt));



			// compile once
//		clo->nn3<num_t>(30.f, 1000, data);

			using namespace std::chrono;
			using hrc = high_resolution_clock;

			hrc::time_point cls = hrc::now();
//		clo->nn3<num_t>(30.f, 1000, data);
			hrc::time_point cle = hrc::now();


			hrc::time_point nnfs = hrc::now();

			{
				using namespace nanoflann;


				std::vector<tvec3<num_t >> scaled;

				std::transform(data.begin(), data.end(), std::back_inserter(scaled),
				               [scale](const auto x) { return x * scale; });

				PointCloud<num_t> cloud = PointCloud<num_t>(scaled, 1);
				typedef KDTreeSingleIndexAdaptor<
						L2_Simple_Adaptor<num_t, PointCloud<num_t> >,
						PointCloud<num_t>,
						3
				> my_kd_tree_t;

				my_kd_tree_t index(3, cloud, KDTreeSingleIndexAdaptorParams(10));
				index.buildIndex();


				nanoflann::SearchParams params;
				params.sorted = false;

				std::vector<std::vector<std::pair<size_t, num_t>>> drain;
//#pragma omp parallel for
				for (size_t m = 0; m < data.size(); ++m) {
					std::vector<std::pair<size_t, num_t>> drainA;
					index.radiusSearch(&data[m][0], R * R * scale, drainA, params);
					drain.push_back(drainA);
				}

//			std::cout << "D=" << drain.size() << "" <<std::endl;
			}


			hrc::time_point nnfe = hrc::now();


			hrc::time_point cpus = hrc::now();

			{

				std::vector<tvec3<num_t >> scaled;

				std::transform(data.begin(), data.end(), std::back_inserter(scaled),
				               [scale](const auto x) { return x * scale; });


				unibn::Octree<tvec3<num_t>> octree;
				octree.initialize(scaled);
				std::vector<std::vector<uint32_t >> drain;
//#pragma omp parallel for
				for (size_t m = 0; m < data.size(); ++m) {
					std::vector<uint32_t> drainA;
					octree.radiusNeighbors<unibn::L2Distance<tvec3<num_t>>>(data[m], R * scale,
					                                                        drainA);
					drain.push_back(drainA);
				}
//			std::cout << "D=" << drain.size() << "" <<std::endl;
			}

			hrc::time_point cpue = hrc::now();

			auto cl = duration_cast<nanoseconds>(cle - cls).count();
			auto cpu = duration_cast<nanoseconds>(cpue - cpus).count();
			auto nnf = duration_cast<nanoseconds>(nnfe - nnfs).count();

			std::cout << "CL  : " << (cl / 1000000.0) << "ms "
			          << "OCT : " << (cpu / 1000000.0) << "ms "
			          << "NNF : " << (nnf / 1000000.0) << "ms "
			          << std::endl;
		}


	}

//	compute::device defaultDevice = compute::system::default_device();
//	std::cout << "\tDefault: " << defaultDevice.name() << std::endl;


}


//#define DO_SURFACE



void writeFile(std::string filename, std::string content) {
	std::ofstream file;
	file.open(filename);
	file << content;
	file.close();
}


int main(int argc, char *argv[]) {


	auto particleType = writer::writeMetaPacked<decltype(particleEntries<size_t, num_t>())>();
	auto triangleType = writer::writeMetaPacked<decltype(triangleEntries<num_t>())>();
	auto headerType = writer::writeMetaPacked<decltype(headerEntries())>();

	writeFile("header.json", headerType.second.dump(1));
	writeFile("particle.json", particleType.second.dump(1));
	writeFile("triangle.json", triangleType.second.dump(1));

	for (const auto &t : {particleType, triangleType, headerType}) {
		std::cout << "size=" << t.first << std::endl;
		std::cout << t.second.dump(3) << std::endl;
	}

	using namespace std::chrono;
	using hrc = high_resolution_clock;

	omp_set_num_threads(1);
	size_t pcount = 6000  * 3;
	size_t iter = 0;



	auto clo = new CLOps(compute::system::default_device());
	clo->doIt();


//	checkClNN3();
//	checkClNN3();
//	checkClNN3();


	std::vector<fluid::Particle<size_t, num_t >> xs;
	size_t offset = 0;
	offset = makeCube(offset, 28.f, pcount / 2, tvec3<num_t>(-500, -350, -250), xs);
	offset = makeCube(offset, 28.f, pcount / 2, tvec3<num_t>(100, -350, -250), xs);

	float i = 0;


	const num_t Xscale = 2.f;
	const num_t Yscale = 0.8f;
	const num_t Zscale = 0.7f;

	std::vector<std::function<const fluid::Response<num_t>(
			fluid::Ray<num_t> &)> > colliders = {
			[&i, Xscale, Yscale, Zscale](const fluid::Ray<num_t> &x) -> fluid::Response<num_t> {
				float xx = (sin(i) * 220);
				float zz = (cos(i) * 50);


				return fluid::Response<num_t>(
						tvec3<num_t>(
								glm::clamp(x.getOrigin().x, (num_t) -500.f * Xscale + xx,
								           (num_t) 500.f * Xscale + xx),
								glm::clamp(x.getOrigin().y, (num_t) -500.f * Yscale,
								           (num_t) 500.f * Yscale),
								glm::clamp(x.getOrigin().z, (num_t) -500.f * Zscale + zz,
								           (num_t) 500.f * Zscale + zz)),
						x.getVelocity());
			}
	};


	std::cout << "Mark" << std::endl;


	auto mmfPSink = mkMmf("particles.mmf", pcount * particleType.first + headerType.first);
	auto mmfTSink = mkMmf("triangles.mmf", 500000 * triangleType.first + headerType.first);

	std::cout << "Go" << std::endl;

	float D = 20.f;
	auto P = static_cast<size_t>(2000.f / D);

	const surface::MCLattice<num_t> &lattice = surface::createLattice<num_t>(P, P, P, -1000, D);

	std::unique_ptr<fluid::SphSolver<size_t, num_t>> solver(
			new fluid::SphSolver<size_t, num_t>(0.1, 550)); // less = less space between particle

	using hrc = high_resolution_clock;

	hrc::time_point start = hrc::now();
	for (size_t j = 0; j < iter; ++j) {

		i += glm::pi<num_t>() / 50;

		hrc::time_point t1 = hrc::now();
		solver->advance(static_cast<num_t> (0.0083 * 1), 2, xs,
		                [](const fluid::Particle<size_t, num_t> &x) {
			                return tvec3<num_t>(0, x.mass * 9.8, 0);
		                }, colliders
		);
		hrc::time_point t2 = hrc::now();


		hrc::time_point s1 = hrc::now();


#ifdef DO_SURFACE


		num_t NR = 32.f;
		const num_t MBC = 100.f;
		const num_t MBCC = MBC * MBC;

#ifndef FLANN_S

		std::vector<tvec3<num_t>> pts;
		std::transform(xs.begin(), xs.end(), std::back_inserter(pts),
					   [](const fluid::Particle<size_t, num_t> &a) { return a.position; });


		unibn::Octree<tvec3<num_t>> octree;
		octree.initialize(pts);
		auto triangles = surface::parameterise<num_t>(100.f,lattice, [&octree, &xs, MBCC, NR](
				const tvec3<num_t> &a) -> num_t {

					std::vector<uint32_t> results;
					octree.radiusNeighbors<unibn::L2Distance<tvec3<num_t>>>(a, NR, results);
					num_t v = 0;
					for (uint32_t &result : results)
						v += (MBCC /
							  glm::length2(xs[result].position - a)) * 2;
					return v;
				});

#else

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


#endif


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
		hrc::time_point mmt2 = hrc::now();



//		sleep(1);


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

	return EXIT_SUCCESS;
}