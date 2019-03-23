
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


	static std::vector<cl::Device> findDeviceWithSignature(const std::string &needle) {
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
			             [needle](const cl::Device &device) {
				             return device.getInfo<CL_DEVICE_NAME>().find(needle) !=
				                    std::string::npos;
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
				queue(cl::CommandQueue(context, device, cl::QueueProperties::None)),
				clsph(clutil::loadProgramFromFile(
						context,
						kernelPath + "oclsph_kernel.cl",
						kernelPath,
						"-DH=((float)" + std::to_string(h) + ")")),
				clmc(clutil::loadProgramFromFile(
						context,
						kernelPath + "oclmc_kernel.cl",
						kernelPath,
						"-DH=((float)" + std::to_string(h) + ")")) {}

	private:

		tvec3<N> lerp(N isolevel, tvec3<N> p1, tvec3<N> p2, N v1, N v2) {
			if (std::abs(isolevel - v1) < 0.00001) return p1;
			if (std::abs(isolevel - v2) < 0.00001) return p2;
			if (std::abs(v1 - v2) < 0.00001) return p1;
			return p1 + (p2 - p1) * ((isolevel - v1) / (v2 - v1));
		}

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

		std::vector<surface::Triangle<N>> advance(const fluid::Config<N> &config,
		                                          std::vector<fluid::Particle<T, N>> &xs,
		                                          const std::vector<fluid::MeshCollider<N>> &colliders) override {

			using hrc = std::chrono::high_resolution_clock;
			using std::chrono::nanoseconds;
			using std::chrono::milliseconds;
			using std::chrono::duration_cast;


			hrc::time_point aabbCpyS = hrc::now();

			tvec3<N> min(std::numeric_limits<N>::max());
			tvec3<N> max(std::numeric_limits<N>::min());

			const size_t atomsN = xs.size();
			std::vector<ClSphAtom> hostAtoms(xs.size());


#if _OPENMP > 201307
#pragma omp declare reduction(glmMin: tvec3<N>: omp_out = glm::min(omp_in, omp_out))
#pragma omp declare reduction(glmMax: tvec3<N>: omp_out = glm::max(omp_in, omp_out))
#pragma omp parallel for reduction(glmMin:min) reduction(glmMax:max)
#endif
			for (int i = 0; i < static_cast<int>(xs.size()); ++i) {
				const fluid::Particle<T, N> &p = xs[i];
				const tvec3<N> velocity =
						(p.mass * config.constantForce) * config.dt + p.velocity;
				const tvec3<N> pStar = (velocity * config.dt) + (p.position / config.scale);
				ClSphAtom &atom = hostAtoms[i];
				ClSphParticle &particle = atom.particle;
				particle.id = p.t,
				particle.type = resolve(p.type),
				particle.mass = p.mass,
				particle.position = clutil::vec3ToCl(p.position),
				particle.velocity = clutil::vec3ToCl(velocity);
				atom.pStar = clutil::vec3ToCl(pStar);
				min = glm::min(pStar, min);
				max = glm::max(pStar, max);
			}
//
//			for (int i = 0; i < static_cast<int>(xs.size()); ++i) {
//				const float3 &pStar = hostAtoms[i].pStar;
//				min.x = glm::min(pStar.x, min.x);
//				min.y = glm::min(pStar.y, min.y);
//				min.z = glm::min(pStar.z, min.z);
//				max.x = glm::max(pStar.x, max.x);
//				max.y = glm::max(pStar.y, max.y);
//				max.z = glm::max(pStar.z, max.z);
//			}


			N padding = h * 2;
			min -= padding;
			max += padding;
			glm::tvec3<size_t> sizes((max - min) / h);

			const size_t gridTableN = zCurveGridIndexAtCoord(sizes.x, sizes.y, sizes.z);
			hrc::time_point aabbCpyE = hrc::now();

			hrc::time_point zCurveS = hrc::now();


#pragma omp parallel for
			for (int i = 0; i < static_cast<int>(hostAtoms.size()); ++i) {
				const float3 pStar = hostAtoms[i].pStar;
				hostAtoms[i].zIndex = zCurveGridIndexAtCoord(
						static_cast<size_t>((pStar.x - min.x) / h),
						static_cast<size_t>((pStar.y - min.y) / h),
						static_cast<size_t>((pStar.z - min.z) / h));
			}


			hrc::time_point zCurveE = hrc::now();

			hrc::time_point sortS = hrc::now();

			// ska_sort(hostAtoms.begin(), hostAtoms.end(),
			//  [](const ClSphAtom &a) { return a.zIndex; });

			std::sort(hostAtoms.begin(), hostAtoms.end(),
			          [](const ClSphAtom &l, const ClSphAtom &r) {
				          return l.zIndex < r.zIndex;
			          });

			hrc::time_point sortE = hrc::now();

			hrc::time_point gtS = hrc::now();

			std::vector<uint> hostGridTable(gridTableN);
			uint gridIndex = 0;
			for (size_t i = 0; i < gridTableN; ++i) {
				hostGridTable[i] = gridIndex;
				while (gridIndex != atomsN && hostAtoms[gridIndex].zIndex == i) {
					gridIndex++;
				}
			}

			hrc::time_point gtE = hrc::now();

#ifdef DEBUG

			std::cout << "atomsN = " << atomsN
			          << " AABB:" << glm::to_string(sizes)
			          << " min:" << glm::to_string(min)
			          << " max:" << glm::to_string(max)
			          << " gridTable = " << hostGridTable.size() << " gridTableN = " << gridTableN
			          << std::endl;

			std::cout << "Go! " << std::endl;

#endif


			hrc::time_point gpuXferS = hrc::now();

			std::vector<ClSphParticle> copiedParticles(atomsN);

			size_t divisior = 2;

			std::vector<float> copiedFields(
					sizes.x * divisior * sizes.y * divisior * sizes.z * divisior);


			hrc::time_point da1 = hrc::now();
			cl::Buffer deviceAtoms(queue, hostAtoms.begin(), hostAtoms.end(), false);
			hrc::time_point da2 = hrc::now();

#ifdef DEBUG
			std::cout << "Device atoms: "
			          << (duration_cast<nanoseconds>(da2 - da1).count() / 1000000.0) << "ms"
			          << std::endl;
			queue.finish();
#endif


			hrc::time_point dgt1 = hrc::now();
			cl::Buffer deviceGridTable(queue, hostGridTable.begin(), hostGridTable.end(), true);
			hrc::time_point dgt2 = hrc::now();

#ifdef DEBUG
			std::cout << "Device GT  : "
			          << (duration_cast<nanoseconds>(dgt2 - dgt1).count() / 1000000.0) << "ms"
			          << std::endl;
			queue.finish();
#endif


			cl::Buffer deviceResult(queue, copiedParticles.begin(), copiedParticles.end(), false);

#ifdef DEBUG
			queue.finish();
#endif


			cl::Buffer deviceFields(queue, copiedFields.begin(), copiedFields.end(), false);


			hrc::time_point gpuXferE = hrc::now();

			hrc::time_point gpuFunctorS = hrc::now();

			auto lambdaKernel = cl::KernelFunctor<
					ClSphConfig, cl::Buffer &, uint, cl::Buffer &, uint
			>(clsph, "sph_lambda");
			auto deltaKernel = cl::KernelFunctor<
					ClSphConfig &, cl::Buffer &, uint, cl::Buffer &, uint
			>(clsph, "sph_delta");
			auto finaliseKernel = cl::KernelFunctor<
					ClSphConfig &, cl::Buffer &, cl::Buffer &
			>(clsph, "sph_finalise");


			auto createFieldKernel = cl::KernelFunctor<
					float3, float, uint3,
					ClSphConfig &, cl::Buffer &, uint, cl::Buffer &, uint,
					cl::Buffer &
			>(clsph, "sph_create_field");


			ClSphConfig clConfig;
			clConfig.dt = config.dt;
			clConfig.scale = config.scale;
			clConfig.iteration = static_cast<size_t>(config.iteration);
			clConfig.min = clutil::vec3ToCl(config.min);
			clConfig.max = clutil::vec3ToCl(config.max);

			hrc::time_point gpuFunctorE = hrc::now();

			hrc::time_point gpuKernelS = hrc::now();
			try {
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
							deviceGridTable, static_cast<uint>(gridTableN));
				}


				finaliseKernel(cl::EnqueueArgs(queue, cl::NDRange(atomsN)),
				               clConfig, deviceAtoms, deviceResult);

				createFieldKernel(
						cl::EnqueueArgs(queue, cl::NDRange(sizes.x * divisior,
								sizes.y * divisior,
						                                   sizes.z * divisior)),
						clutil::vec3ToCl(min), h/2, clutil::uvec3ToCl(sizes * divisior),
						clConfig,
						deviceAtoms, static_cast<uint>(atomsN),
						deviceGridTable, static_cast<uint>(gridTableN), deviceFields);

			} catch (const cl::Error &exc) {
				std::cerr << "Kernel failed to execute: " << exc.what() << " -> "
				          << clResolveError(exc.err()) << "(" << exc.err() << ")" << std::endl;
				throw;
			}
#ifdef DEBUG
			queue.finish();
#endif

			hrc::time_point gpuKernelE = hrc::now();


			hrc::time_point gpuXferRS = hrc::now();
			cl::copy(queue, deviceFields, copiedFields.begin(), copiedFields.end());

			cl::copy(queue, deviceResult, copiedParticles.begin(), copiedParticles.end());
#ifdef DEBUG
			queue.finish();
#endif

			hrc::time_point gpuXferRE = hrc::now();

#pragma omp parallel for
			for (int i = 0; i < xs.size(); ++i) {
				const ClSphParticle &p = copiedParticles[i];
				fluid::Particle<T, N> &particle = xs[i];

				particle.t = static_cast<T>(p.id);
				particle.type = resolve(p.type);
				particle.mass = static_cast<N>(p.mass);
				particle.position = clutil::clToVec3<N>(p.position);
				particle.velocity = clutil::clToVec3<N>(p.velocity);
//				std::cout << "GPU >> "
//				          << " t=" << particle.t
//				          << " p=" << glm::to_string(particle.position)
//				          << " v=" << glm::to_string(particle.velocity) << "\n";
			}

//			cl::copy(atoms, actualAtoms.begin(), actualAtoms.end());
//		for (auto b : actualAtoms) {
//			showAtom(b);
//		}


#ifdef DEBUG

			auto aabbCpy = duration_cast<nanoseconds>(aabbCpyE - aabbCpyS).count();
			auto zCurve = duration_cast<nanoseconds>(zCurveE - zCurveS).count();
			auto sort = duration_cast<nanoseconds>(sortE - sortS).count();
			auto gt = duration_cast<nanoseconds>(gtE - gtS).count();
			auto gpuKernel = duration_cast<nanoseconds>(gpuKernelE - gpuKernelS).count();
			auto gpuXfer = duration_cast<nanoseconds>(gpuXferE - gpuXferS).count();
			auto gpuXferR = duration_cast<nanoseconds>(gpuXferRE - gpuXferRS).count();
			auto gpuFunctor = duration_cast<nanoseconds>(gpuFunctorE - gpuFunctorS).count();
			std::cout
					<< "\tCPU aabbCpy= " << (aabbCpy / 1000000.0) << "ms\n"
					<< "\tCPU zCurve = " << (zCurve / 1000000.0) << "ms\n"
					<< "\tCPU sort   = " << (sort / 1000000.0) << "ms\n"
					<< "\tCPU gt     = " << (gt / 1000000.0) << "ms\n"
					<< "\tGPU xfer     = " << (gpuXfer / 1000000.0) << "ms\n"
					<< "\tGPU xferR    = " << (gpuXferR / 1000000.0) << "ms\n"
					<< "\tGPU functor  = " << (gpuFunctor / 1000000.0) << "ms\n"
					<< "\tGPU kernel   = " << (gpuKernel / 1000000.0) << "ms\n"
					<< std::endl;
#endif


			const static std::array<std::tuple<size_t, size_t, size_t>, 8> &verticies = {
					std::make_tuple(0, 0, 0),
					std::make_tuple(1, 0, 0),
					std::make_tuple(1, 1, 0),
					std::make_tuple(0, 1, 0),
					std::make_tuple(0, 0, 1),
					std::make_tuple(1, 0, 1),
					std::make_tuple(1, 1, 1),
					std::make_tuple(0, 1, 1),
			};


			std::vector<surface::Triangle<N>> triangles;
			std::vector<tvec3<N>> points;


			static const float isolevel = 100.f;

#ifndef _MSC_VER
#pragma omp declare reduction (merge : std::vector<surface::Triangle<N>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for collapse(3) reduction(merge: triangles)
#endif
			for (size_t x = 0; x < (sizes.x * divisior) - 1; ++x) {
				for (size_t y = 0; y < (sizes.y * divisior) - 1; ++y) {
					for (size_t z = 0; z < (sizes.z * divisior) - 1; ++z) {

						std::array<N, 8> ns{};
						std::array<tvec3<N>, 8> vs{};

						for (size_t j = 0; j < 8; ++j) {

							size_t ox = x + std::get<0>(verticies[j]);
							size_t oy = y + std::get<1>(verticies[j]);
							size_t oz = z + std::get<2>(verticies[j]);


							vs[j] = tvec3<N>(
									ox * (h /divisior) + min.x,
									oy * (h /divisior) + min.y,
									oz * (h /divisior) + min.z);

							ns[j] = copiedFields[index3d(ox, oy, oz,
							                             sizes.x * divisior,
							                             sizes.y * divisior,
							                             sizes.z * divisior)];

						}

						size_t ci = 0;
						if (ns[0] < isolevel) ci |= 1;
						if (ns[1] < isolevel) ci |= 2;
						if (ns[2] < isolevel) ci |= 4;
						if (ns[3] < isolevel) ci |= 8;
						if (ns[4] < isolevel) ci |= 16;
						if (ns[5] < isolevel) ci |= 32;
						if (ns[6] < isolevel) ci |= 64;
						if (ns[7] < isolevel) ci |= 128;

						/* Cube is entirely in/out of the surface */
						if (surface::edgeTable[ci] == 0)
							std::vector<surface::Triangle<N>>
							();

						std::array<tvec3<N>, 12> ts;
						/* Find the vertices where the surface intersects the cube */
						if (surface::edgeTable[ci] & 1 << 0)
							ts[0] = lerp(isolevel, vs[0], vs[1], ns[0], ns[1]);
						if (surface::edgeTable[ci] & 1 << 1)
							ts[1] = lerp(isolevel, vs[1], vs[2], ns[1], ns[2]);
						if (surface::edgeTable[ci] & 1 << 2)
							ts[2] = lerp(isolevel, vs[2], vs[3], ns[2], ns[3]);
						if (surface::edgeTable[ci] & 1 << 3)
							ts[3] = lerp(isolevel, vs[3], vs[0], ns[3], ns[0]);
						if (surface::edgeTable[ci] & 1 << 4)
							ts[4] = lerp(isolevel, vs[4], vs[5], ns[4], ns[5]);
						if (surface::edgeTable[ci] & 1 << 5)
							ts[5] = lerp(isolevel, vs[5], vs[6], ns[5], ns[6]);
						if (surface::edgeTable[ci] & 1 << 6)
							ts[6] = lerp(isolevel, vs[6], vs[7], ns[6], ns[7]);
						if (surface::edgeTable[ci] & 1 << 7)
							ts[7] = lerp(isolevel, vs[7], vs[4], ns[7], ns[4]);
						if (surface::edgeTable[ci] & 1 << 8)
							ts[8] = lerp(isolevel, vs[0], vs[4], ns[0], ns[4]);
						if (surface::edgeTable[ci] & 1 << 9)
							ts[9] = lerp(isolevel, vs[1], vs[5], ns[1], ns[5]);
						if (surface::edgeTable[ci] & 1 << 10)
							ts[10] = lerp(isolevel, vs[2], vs[6], ns[2], ns[6]);
						if (surface::edgeTable[ci] & 1 << 11)
							ts[11] = lerp(isolevel, vs[3], vs[7], ns[3], ns[7]);

						for (size_t i = 0; surface::triTable[ci][i] != -1; i += 3) {
//						points.emplace_back(ts[surface::triTable[ci][i + 0]]);
//						points.emplace_back(ts[surface::triTable[ci][i + 1]]);
//						points.emplace_back(ts[surface::triTable[ci][i + 2]]);

							triangles.emplace_back(
									ts[surface::triTable[ci][i]] * config.scale,
									ts[surface::triTable[ci][i + 1]] * config.scale,
									ts[surface::triTable[ci][i + 2]] * config.scale);
						}


					}
				}
			}


			return triangles;
		}

	};

}

#endif //LIBFLUID_CLSPH_HPP
