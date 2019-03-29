#include <utility>

#include <utility>

#include <utility>
#include <iomanip>


#ifndef LIBFLUID_CLSPH_HPP
#define LIBFLUID_CLSPH_HPP

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_ENABLE_EXCEPTIONS

#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_SIMD_AVX2
#define GLM_ENABLE_EXPERIMENTAL


#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
#include <memory>
#include <algorithm>
#include <CL/cl2.hpp>

#include <sys/types.h>
#include <sys/stat.h>

#include "fluid.hpp"
#include "oclsph_type.h"
#include "zcurve.h"
#include "mc.h"
#include "ska_sort.hpp"

#define DEBUG

namespace fsutils {

	bool statDir(const std::string &path) {
		struct stat info{};
		if (stat(path.c_str(), &info) != 0) return false;
		return static_cast<bool>(info.st_mode & S_IFDIR);
	}


}

namespace clutil {


	template<typename T>
	std::string mkString(const std::vector<T> &xs, const std::function<std::string(T)> &f) {
		return std::accumulate(xs.begin(), xs.end(), std::string(),
							   [&f](const std::string &acc, const T &x) {
								   return acc + (acc.length() > 0 ? "," : "") + f(x);
							   });
	}

	static std::vector<cl::Device>
	findDeviceWithSignature(const std::vector<std::string> &needles) {
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		std::vector<cl::Device> matching;
		for (auto &p : platforms) {
			std::vector<cl::Device> devices;
			try {
				p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
			} catch (const std::exception &e) {
				std::cerr << "Enumeration failed at `" << p.getInfo<CL_PLATFORM_NAME>()
						  << "` : "
						  << e.what() << std::endl;
			}
			std::copy_if(devices.begin(), devices.end(), std::back_inserter(matching),
						 [needles](const cl::Device &device) {
							 return std::any_of(needles.begin(), needles.end(),
												[&device](auto needle) {
													return device.getInfo<CL_DEVICE_NAME>().find(
															needle) !=
														   std::string::npos;
												});
						 });
		}
		return matching;
	}

	static void enumeratePlatformToCout() {
		std::cout << "Enumerating OpenCL platforms:" << std::endl;

		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		auto platform = cl::Platform::getDefault();
		for (auto &p : platforms) {
			try {

				std::cout << "\t├─┬Platform"
						  << (platform == p ? "(Default):" : ":")
						  << p.getInfo<CL_PLATFORM_NAME>()
						  << "\n\t│ ├Vendor     : " << p.getInfo<CL_PLATFORM_VENDOR>()
						  << "\n\t│ ├Version    : " << p.getInfo<CL_PLATFORM_VERSION>()
						  << "\n\t│ ├Profile    : " << p.getInfo<CL_PLATFORM_PROFILE>()
						  << "\n\t│ ├Extensions : " << p.getInfo<CL_PLATFORM_EXTENSIONS>()
						  << "\n\t│ └Devices"
						  << std::endl;
				std::vector<cl::Device> devices;
				p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
				for (auto &d : devices) {
					std::cout
							<< "\t│\t     └┬Name    : " << d.getInfo<CL_DEVICE_NAME>()
							<< "\n\t│\t      ├Type    : " << d.getInfo<CL_DEVICE_TYPE>()
							<< "\n\t│\t      ├Vendor  : " << d.getInfo<CL_DEVICE_VENDOR_ID>()
							<< "\n\t│\t      ├Avail.  : " << d.getInfo<CL_DEVICE_AVAILABLE>()
							<< "\n\t│\t      ├Max CU. : "
							<< d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()
							<< "\n\t│\t      └Version : " << d.getInfo<CL_DEVICE_VERSION>()
							<< std::endl;
				}
			} catch (const std::exception &e) {
				std::cerr << "Enumeration failed at `" << p.getInfo<CL_PLATFORM_NAME>()
						  << "` : "
						  << e.what() << std::endl;
			}
		}

	}

	static cl::Program loadProgramFromFile(
			const cl::Context &context,
			const std::string &file,
			const std::string &include,
			const std::string &flags = "") {
		std::cout << "Compiling CL kernel:`" << file << "` using " << std::endl;
		std::ifstream t(file);


		if (!fsutils::statDir(include))
			throw std::runtime_error("Unable to stat dir:`" + include + "`");
		if (!t.good()) throw std::runtime_error("Unable to read file:`" + file + "`");


		std::stringstream source;
		source << t.rdbuf();
		cl::Program program = cl::Program(context, source.str());

		const auto printBuildInfo = [&program]() {
			auto log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
			std::cerr << "Compiler output(" << log.size() << "):\n" << std::endl;
			for (auto &pair : log) {
				std::cerr << ">" << pair.second << std::endl;
			}
		};
		const std::string clFlags = " -cl-std=CL1.2"
									" -w"
									" -cl-mad-enable"
									" -cl-no-signed-zeros"
									" -cl-unsafe-math-optimizations"
									" -cl-finite-math-only";
		const std::string build = clFlags + " -I " + include + " " + flags;
		std::cout << "Using args:`" << build << "`" << std::endl;
		try {
			program.build(build.c_str());
		} catch (...) {
			std::cerr << "Program failed to compile, source:\n" << source.str() << std::endl;
			printBuildInfo();
			throw;
		}
		std::cout << "Program compiled" << std::endl;
		printBuildInfo();
		return program;
	}

	template<typename N, typename T>
	static inline T gen_type3(N x, N y, N z) {
		return {{static_cast<N>(x), static_cast<N>(y), static_cast<N>(z)}};
	}

	template<typename N>
	static inline cl_float3 float3(N x, N y, N z) {
		return {{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)}};
	}

	inline cl_float4 float4(float x, float y, float z, float w) { return {{x, y, z, w}}; }

	template<typename N>
	static inline cl_float3 float3(N x) {
		return {{static_cast<float>(x), static_cast<float>(x), static_cast<float>(x)}};
	}

	inline cl_float4 float4(float x) { return {{x, x, x, x}}; }


	template<typename N>
	inline glm::tvec3<N> clToVec3(cl_float3 v) {
		return glm::tvec3<float>(v.x, v.y, v.z);
	}

	inline cl_float3 vec3ToCl(glm::tvec3<float> v) {
		return float3(v.x, v.y, v.z);
	}

	inline uint3 uvec3ToCl(glm::tvec3<size_t> v) {
		return gen_type3<uint, uint3>(v.x, v.y, v.z);
	}

}

namespace ocl {

	using glm::tvec3;


	template<typename T, typename N>
	class SphSolver : public fluid::SphSolver<T, N> {

	private:

		const N h;
		const cl::Device device;
		const cl::Context context;
		const cl::Program clsph;
		const cl::Program clmc;

		cl::CommandQueue queue;

	public:
		explicit SphSolver(N h, const std::string &kernelPath, const cl::Device &device) :
				h(h),
				device(device),
				context(cl::Context(device)),
				queue(cl::CommandQueue(context, device, cl::QueueProperties::OutOfOrder)),
				clsph(clutil::loadProgramFromFile(
						context,
						kernelPath + "oclsph_kernel.cl",
						kernelPath,
						"-DH=((float)" + std::to_string(h) + ")")),
				clmc(clutil::loadProgramFromFile(
						context,
						kernelPath + "oclmc_kernel.cl",
						kernelPath,
						"-DH=((float)" + std::to_string(h) + ")")) {
			checkSize();

		}

	private:

		static inline ClSphType resolve(fluid::Type t) {
			switch (t) {
				case fluid::Type::Fluid: return ClSphType::Fluid;
				case fluid::Type::Obstacle: return ClSphType::Obstacle;
				default: throw std::logic_error("unhandled branch");
			}
		}

		static inline fluid::Type resolve(ClSphType t) {
			switch (t) {
				case ClSphType::Fluid: return fluid::Type::Fluid;
				case ClSphType::Obstacle: return fluid::Type::Obstacle;
				default: throw std::logic_error("unhandled branch");
			}
		}


		std::vector<surface::Triangle<N>> sampleLattice(
				N isolevel, N scale,
				const tvec3<N> min, N step,
				const surface::Lattice<N> &lattice) {

			std::vector<surface::Triangle<N>> triangles;

#ifndef _MSC_VER
#pragma omp declare reduction (merge : std::vector<surface::Triangle<N>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for collapse(3) reduction(merge: triangles)
#endif
			for (size_t x = 0; x < lattice.xSize() - 1; ++x) {
				for (size_t y = 0; y < lattice.ySize() - 1; ++y) {
					for (size_t z = 0; z < lattice.zSize() - 1; ++z) {
						std::array<N, 8> ns{};
						std::array<tvec3<N>, 8> vs{};
						for (size_t j = 0; j < 8; ++j) {
							tvec3<size_t> offset = tvec3<size_t>(x, y, z) +
												   surface::CUBE_OFFSETS[j];
							ns[j] = lattice(offset.x, offset.y, offset.z);

							vs[j] = (tvec3<N>(offset) * step + min) * scale;



//							lattice(offset.x, offset.y, offset.z)


//							nns[j] = (tvec3<N>(offset) * step + min) * scale;

						}
						surface::marchSingle(isolevel, ns, vs, triangles);
					}
				}
			}
			return triangles;
		}


		class Stopwatch {

			typedef std::chrono::time_point<std::chrono::system_clock> time;

			struct Entry {
				const std::string name;
				const time begin;
				time end;
				Entry(std::string name,
					  const time &begin) : name(std::move(name)), begin(begin) {}
			};

			std::string name;
			std::vector<Entry> entries;

		public:
			explicit Stopwatch(std::string name) : name(std::move(name)) {}

		public:
			std::function<void(void)> start(const std::string &name) {
				const size_t size = entries.size();
				entries.push_back(Entry(name, std::chrono::system_clock::now()));
				return [size, this]() { entries[size].end = std::chrono::system_clock::now(); };
			}

			friend std::ostream &operator<<(std::ostream &os, const Stopwatch &stopwatch) {
				os << "Stopwatch[ " << stopwatch.name << "]:\n";

				size_t maxLen = std::max_element(stopwatch.entries.begin(), stopwatch.entries.end(),
												 [](const Entry &l, const Entry &r) {
													 return l.name.size() < r.name.size();
												 })->name.size() + 3;

				for (const Entry &e: stopwatch.entries) {
					os << "    ->"
					   << "`" << e.name << "` "
					   << std::setw(static_cast<int>(maxLen - e.name.size())) << " : " <<
					   (std::chrono::duration_cast<std::chrono::nanoseconds>(
							   e.end - e.begin).count() / 1000'000.0) << "ms" << std::endl;
				}
				return os;
			}

		};


		void checkSize() {

			std::vector<size_t> expected(_SIZES, _SIZES + _SIZES_LENGTH);
			std::vector<size_t> actual(_SIZES_LENGTH, 0);

			try {
				cl::Buffer buffer(context, CL_MEM_WRITE_ONLY, sizeof(_SIZES));
				cl::KernelFunctor<cl::Buffer &>(clsph, "check_size")
						(cl::EnqueueArgs(queue, cl::NDRange(_SIZES_LENGTH)), buffer);
				cl::copy(queue, buffer, actual.begin(), actual.end());
				queue.finish();
			} catch (cl::Error &exc) {
				std::cerr << "Kernel failed to execute: " << exc.what() << " -> "
						  << clResolveError(exc.err()) << "(" << exc.err() << ")" << std::endl;
				throw;
			}

#ifdef DEBUG
			std::cout << "Actual(" << _SIZES_LENGTH << ")  ="
					  << clutil::mkString<size_t>(actual, [](auto x) { return std::to_string(x); })
					  << std::endl;


			std::cout << "Expected(" << _SIZES_LENGTH << ")="
					  << clutil::mkString<size_t>(expected,
												  [](auto x) { return std::to_string(x); })
					  << std::endl;
#endif


			assert(expected == actual);
		}

		std::vector<ClSphAtom> advectAndCopy(const fluid::Config<N> &config,
											 const std::vector<fluid::Particle<T, N>> &xs) {

			std::vector<ClSphAtom> atoms(xs.size());
#pragma omp parallel for
			for (int i = 0; i < static_cast<int>(xs.size()); ++i) {
				const fluid::Particle<T, N> &p = xs[i];
				const tvec3<N> velocity =
						(p.mass * config.constantForce) * config.dt + p.velocity;
				const tvec3<N> pStar = (velocity * config.dt) + (p.position / config.scale);
				ClSphAtom &atom = atoms[i];
				ClSphParticle &particle = atom.particle;
				particle.id = p.id,
				particle.type = resolve(p.type),
				particle.mass = p.mass,
				particle.position = clutil::vec3ToCl(p.position),
				particle.velocity = clutil::vec3ToCl(velocity);
				atom.pStar = clutil::vec3ToCl(pStar);
			}
			return atoms;
		}


		const std::tuple<ClSphConfig, tvec3<N>, tvec3<size_t> > computeBoundAndIndex(
				const fluid::Config<N> &config,
				std::vector<ClSphAtom> &atoms) const {

			tvec3<N> minExtent(std::numeric_limits<N>::max());
			tvec3<N> maxExtent(std::numeric_limits<N>::min());
#if _OPENMP > 201307
#pragma omp declare reduction(glmMin: tvec3<N>: omp_out = glm::min(omp_in, omp_out))
#pragma omp declare reduction(glmMax: tvec3<N>: omp_out = glm::max(omp_in, omp_out))
#pragma omp parallel for reduction(glmMin:minExtent) reduction(glmMax:maxExtent)
#endif
			for (int i = 0; i < static_cast<int>(atoms.size()); ++i) {
				minExtent = glm::min(clutil::clToVec3<N>(atoms[i].pStar), minExtent);
				maxExtent = glm::max(clutil::clToVec3<N>(atoms[i].pStar), maxExtent);
			}

			const N padding = h * 2;
			minExtent -= padding;
			maxExtent += padding;

#pragma omp parallel for
			for (int i = 0; i < static_cast<int>(atoms.size()); ++i) {
				const float3 pStar = atoms[i].pStar;
				atoms[i].zIndex = zCurveGridIndexAtCoord(
						static_cast<size_t>((pStar.x - minExtent.x) / h),
						static_cast<size_t>((pStar.y - minExtent.y) / h),
						static_cast<size_t>((pStar.z - minExtent.z) / h));
			}

			ClSphConfig clConfig;
			clConfig.dt = config.dt;
			clConfig.scale = config.scale;
			clConfig.iteration = static_cast<size_t>(config.iteration);
			clConfig.minBound = clutil::vec3ToCl(config.minBound);
			clConfig.maxBound = clutil::vec3ToCl(config.maxBound);
			return std::make_tuple(clConfig, minExtent,
								   glm::tvec3<size_t>((maxExtent - minExtent) / h));
		}


		void overwrite(std::vector<fluid::Particle<T, N>> &xs,
					   const std::vector<ClSphParticle> &hostParticles) {
#pragma omp parallel for
			for (int i = 0; i < static_cast<int>(xs.size()); ++i) {
				const ClSphParticle &p = hostParticles[i];
				fluid::Particle<T, N> &particle = xs[i];
				particle.id = static_cast<T>(p.id);
				particle.type = resolve(p.type);
				particle.mass = static_cast<N>(p.mass);
				particle.position = clutil::clToVec3<N>(p.position);
				particle.velocity = clutil::clToVec3<N>(p.velocity);
//				std::cout << "GPU >> "
//				          << " t=" << particle.t
//				          << " p=" << glm::to_string(particle.position)
//				          << " v=" << glm::to_string(particle.velocity) << "\n";
			}
		}

	public:
		std::vector<surface::Triangle<N>> advance(const fluid::Config<N> &config,
												  std::vector<fluid::Particle<T, N>> &xs,
												  const std::vector<fluid::MeshCollider<N>> &colliders) override {


			Stopwatch watch = Stopwatch("CPU advance");

			ClMcConfig mcConfig;
			mcConfig.sampleResolution = 2.5f;
			mcConfig.particleSize = 50.f;
			mcConfig.particleInfluence = 0.8;


			auto advect = watch.start("CPU advect+copy");
			std::vector<ClSphAtom> hostAtoms = advectAndCopy(config, xs);
			const size_t atomsN = xs.size();
			advect();


			auto bound = watch.start("CPU bound+zindex");

			ClSphConfig clConfig;
			tvec3<N> minExtent;
			tvec3<size_t> extent;

			std::tie(clConfig, minExtent, extent) = computeBoundAndIndex(config, hostAtoms);

			bound();

			auto sortz = watch.start("CPU sortz");

			std::sort(hostAtoms.begin(), hostAtoms.end(),
					  [](const ClSphAtom &l, const ClSphAtom &r) {
						  return l.zIndex < r.zIndex;
					  });

			// ska_sort(hostAtoms.begin(), hostAtoms.end(),
			//  [](const ClSphAtom &a) { return a.zIndex; });

			sortz();


			auto gridtable = watch.start("CPU gridtable");


			const size_t gridTableN = zCurveGridIndexAtCoord(extent.x, extent.y, extent.z);

			std::vector<uint> hostGridTable(gridTableN);
			uint gridIndex = 0;
			for (size_t i = 0; i < gridTableN; ++i) {
				hostGridTable[i] = gridIndex;
				while (gridIndex != atomsN && hostAtoms[gridIndex].zIndex == i) {
					gridIndex++;
				}
			}

			gridtable();


			auto collider_concat = watch.start("CPU collider++");


			std::vector<ClSphTraiangle> hostColliderMesh;

			for (int i = 0; i < static_cast<int>(colliders.size()); ++i) {
				auto xs = colliders[i].triangles;
				for (int j = 0; j < static_cast<int>(xs.size()); ++j) {
					ClSphTraiangle trig;
					trig.a = clutil::vec3ToCl(xs[i].v0);
					trig.b = clutil::vec3ToCl(xs[i].v1);
					trig.c = clutil::vec3ToCl(xs[i].v2);
					hostColliderMesh.push_back(trig);
				}
			}

			collider_concat();


#ifdef DEBUG
			std::cout << "Atoms = " << atomsN
					  << " Extent = " << glm::to_string(extent)
					  << " GridTable = " << hostGridTable.size()
					  << std::endl;
#endif

			auto functors = watch.start("CPU functors");


			auto lambdaKernel = cl::KernelFunctor<
					ClSphConfig, cl::Buffer &, uint, cl::Buffer &, uint
			>(clsph, "sph_lambda");
			auto deltaKernel = cl::KernelFunctor<
					ClSphConfig &, cl::Buffer &, uint, cl::Buffer &, uint,
					cl::Buffer &, uint
			>(clsph, "sph_delta");
			auto finaliseKernel = cl::KernelFunctor<
					ClSphConfig &, cl::Buffer &, cl::Buffer &
			>(clsph, "sph_finalise");


			auto createFieldKernel = cl::KernelFunctor<
					ClSphConfig &, cl::Buffer &, uint, cl::Buffer &, uint,
					float3, ClMcConfig,
					cl::Buffer &, uint3, uint3
			>(clsph, "sph_create_field");


			functors();


			const tvec3<size_t> sampleSize = tvec3<size_t>(
					glm::floor(tvec3<N>(extent) * mcConfig.sampleResolution)) + tvec3<size_t>(1);

			auto mcLattice = surface::Lattice<N>(sampleSize.x, sampleSize.y, sampleSize.z, -1);
			std::vector<ClSphParticle> hostParticles(atomsN);

			auto kernel_exec = watch.start("\t[GPU] ===total===");

			try {

				auto kernel_copy = watch.start("\t[GPU] kernel_copy");
				cl::Buffer deviceAtoms(
						queue, hostAtoms.begin(), hostAtoms.end(), false);
				cl::Buffer deviceColliderMesh(
						queue, hostColliderMesh.begin(), hostColliderMesh.end(), true);
				cl::Buffer deviceGridTable(
						queue, hostGridTable.begin(), hostGridTable.end(), true);

				cl::Buffer deviceParticles(context,
										   CL_MEM_WRITE_ONLY, sizeof(ClSphParticle) * atomsN);
				cl::Buffer deviceFields(context,
										CL_MEM_WRITE_ONLY, sizeof(N) * mcLattice.size());

#ifdef DEBUG
				queue.finish();
#endif
				kernel_copy();


				auto lambda_delta = watch.start(
						"\t[GPU] sph-lambda/delta*" + std::to_string(config.iteration));


				for (size_t itr = 0; itr < config.iteration; ++itr) {
					lambdaKernel(
							cl::EnqueueArgs(queue, cl::NDRange(atomsN)),
							clConfig,
							deviceAtoms, static_cast<uint>(atomsN),
							deviceGridTable, static_cast<uint>(gridTableN));
					deltaKernel(
							cl::EnqueueArgs(queue, cl::NDRange(atomsN)),
							clConfig,
							deviceAtoms, static_cast<uint>(atomsN),
							deviceGridTable, static_cast<uint>(gridTableN),
							deviceColliderMesh, static_cast<uint>(hostColliderMesh.size())

					);
				}
#ifdef DEBUG
				queue.finish();
#endif
				lambda_delta();


				auto finalise = watch.start("\t[GPU] sph-finalise");


				finaliseKernel(cl::EnqueueArgs(queue, cl::NDRange(atomsN)),
							   clConfig, deviceAtoms, deviceParticles);
#ifdef DEBUG
				queue.finish();
#endif
				finalise();

				auto create_field = watch.start("\t[GPU] mc-field");

				createFieldKernel(
						cl::EnqueueArgs(queue, cl::NDRange(
								sampleSize.x, sampleSize.y, sampleSize.z)),
						clConfig,
						deviceAtoms, static_cast<uint>(atomsN),
						deviceGridTable, static_cast<uint>(gridTableN),
						clutil::vec3ToCl(minExtent), mcConfig,
						deviceFields, clutil::uvec3ToCl(sampleSize), clutil::uvec3ToCl(extent));
#ifdef DEBUG
				queue.finish();
#endif
				create_field();

#ifdef DEBUG
				queue.finish();
#endif

				auto kernel_return = watch.start("\t[GPU] kernel_return");
				cl::copy(queue, deviceFields, mcLattice.begin(), mcLattice.end());
				cl::copy(queue, deviceParticles, hostParticles.begin(), hostParticles.end());
#ifdef DEBUG
				queue.finish();
#endif
				kernel_return();

			} catch (const cl::Error &exc) {
				std::cerr << "Kernel failed to execute: " << exc.what() << " -> "
						  << clResolveError(exc.err()) << "(" << exc.err() << ")" << std::endl;
				throw;
			}
			kernel_exec();


			auto write_back = watch.start("write_back");
			overwrite(xs, hostParticles);
			write_back();

//			cl::copy(atoms, actualAtoms.begin(), actualAtoms.end());
//		for (auto b : actualAtoms) {
//			showAtom(b);
//		}


			auto march = watch.start("CPU mc");

			std::vector<surface::Triangle<N>> triangles =
					sampleLattice(100, config.scale,
								  minExtent, h / mcConfig.sampleResolution, mcLattice);


			march();


//			std::vector<unsigned short> outIdx;
//			std::vector<tvec3<N>> outVert;
//			hrc::time_point vbiStart = hrc::now();
//			surface::indexVBO2<N>(triangles, outIdx, outVert);
//			hrc::time_point vbiEnd = hrc::now();
//			auto vbi = duration_cast<nanoseconds>(vbiEnd - vbiStart).count();
//
//			std::cout
//					<< "\n\tTrigs = " << triangles.size()
//					<< "\n\tIdx   = " << outIdx.size()
//					<< "\n\tVert  = " << outVert.size()
//					<< "\n\tVBI   = " << (vbi / 1000000.0) << "ms\n"
//					<< std::endl;



#ifdef DEBUG
			std::cout << "MC lattice: " << mcLattice.size() << " Grid=" << glm::to_string(extent)
					  << " res="
					  << mcLattice.xSize() << "x"
					  << mcLattice.ySize() << "x"
					  << mcLattice.zSize()
					  << std::endl;
			std::cout << watch << std::endl;
#endif

			return triangles;
		}


	};

}

#endif //LIBFLUID_CLSPH_HPP
