#ifndef LIBFLUID_CLSPH_HPP
#define LIBFLUID_CLSPH_HPP

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_ENABLE_EXCEPTIONS

#include "clsph_types.h"
#include <CL/cl2.hpp>
#include <iostream>
#include <sstream>
#include <chrono>
#include <vector>
#include <memory>
#include <algorithm>
#include "fluid.hpp"
#include "zcurve.h"
#include "ska_sort.hpp"

using glm::tvec3;
namespace clutil {

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
}

namespace clsph {


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

	static const cl::Program loadProgramFromFile(const std::string &file) {
		std::ifstream t(file);
		std::stringstream source;
		source << t.rdbuf();
		cl::Program program = cl::Program(source.str());

		auto printBuildInfo = [&program]() {
			cl_int buildErr = CL_SUCCESS;
			for (auto &pair : program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr)) {
				std::cerr << pair.second << std::endl << std::endl;
			}
		};

		try {
			program.build(" -cl-std=CL1.2 -w"
			              " -cl-mad-enable"
			              " -cl-no-signed-zeros"
			              " -cl-unsafe-math-optimizations"
			              " -cl-finite-math-only"
			              " -I /home/tom/libfluid");
		} catch (...) {
			std::cerr << "Program failed to compile" << std::endl;
			printBuildInfo();
			throw;
		}
		std::cout << "Program compiled" << std::endl;
		printBuildInfo();
		return program;
	}


	template<typename T, typename N>
	class CLOps {

	private:
		const cl::Platform platform;
		const cl::Program program;

	public:

		explicit CLOps(const cl::Platform &platform = cl::Platform::getDefault()) :
				platform(platform),
				program(loadProgramFromFile("../clsph_kernel.cl")) {
			std::cout << "Using default platform: `" << platform.getInfo<CL_PLATFORM_NAME>() << "`"
			          << std::endl;
		}

	private :
		void showAtom(ClSphAtom atom) {
			std::cout << "P " << atom.particle.id << " " << atom.particle.mass << " " << atom.lambda
			          << std::endl;
		}


	public:


		static ClSphType resolve(fluid::Type t) {
			switch (t) {
				case fluid::Type::Fluid: return ClSphType::Fluid;
				case fluid::Type::Obstacle: return ClSphType::Obstacle;
			}
		}

		static fluid::Type resolve(ClSphType t) {
			switch (t) {
				case ClSphType::Fluid: return fluid::Type::Fluid;
				case ClSphType::Obstacle: return fluid::Type::Obstacle;
			}
		}


		constexpr static float H = 0.1f;

		const N HD2 = 0.1 / 2;


		void run(
				std::vector<fluid::Particle<T, N>> &particles,
				size_t iter, N scale,
				glm::tvec3<N> constForce,
				N dt
		) {
			using hrc = std::chrono::high_resolution_clock;

			hrc::time_point aabbCpyS = hrc::now();

			glm::tvec3<N> min(std::numeric_limits<N>::max());
			glm::tvec3<N> max(std::numeric_limits<N>::min());
			N sideLength = (HD2 * 2);

			const size_t atomsN = particles.size();
			std::vector<ClSphAtom> hostAtoms(particles.size());

//#pragma omp parallel for reduction(min:min.x) reduction(min:min.y) reduction(min:min.z) reduction(max:max.x) reduction(max:max.y) reduction(max:max.z)
			for (size_t i = 0; i < particles.size(); ++i) {
				const fluid::Particle<T, N> &p = particles[i];
				const glm::tvec3<N> velocity = (p.mass * constForce) * dt + p.velocity;
				const glm::tvec3<N> pStar = (velocity * dt) + (p.position / scale);

				hostAtoms[i] = (ClSphAtom{
						.particle = {
								.id = p.t,
								.type = resolve(p.type),
								.mass = p.mass,
								.position = clutil::vec3ToCl(p.position),
								.velocity = clutil::vec3ToCl(velocity)
						},
						.pStar = clutil::vec3ToCl(pStar),
				});

				min.x = glm::min(pStar.x, min.x);
				min.y = glm::min(pStar.y, min.y);
				min.z = glm::min(pStar.z, min.z);

				max.x = glm::max(pStar.x, max.x);
				max.y = glm::max(pStar.y, max.y);
				max.z = glm::max(pStar.z, max.z);
			}

			N padding = sideLength * 2;
			min -= padding;
			max += padding;
			glm::tvec3<size_t> sizes((max - min) / sideLength);

			const size_t gridTableN = zCurveGridIndexAtCoord(sizes.x, sizes.y, sizes.z);
			hrc::time_point aabbCpyE = hrc::now();

			hrc::time_point zCurveS = hrc::now();


#pragma omp parallel for
			for (size_t i = 0; i < hostAtoms.size(); ++i) {
				const float3 pStar = hostAtoms[i].pStar;
				hostAtoms[i].zIndex = zCurveGridIndexAtCoord(
						static_cast<size_t>((pStar.x - min.x) / sideLength),
						static_cast<size_t>((pStar.y - min.y) / sideLength),
						static_cast<size_t>((pStar.z - min.z) / sideLength));
			}


			hrc::time_point zCurveE = hrc::now();

			hrc::time_point sortS = hrc::now();

//			ska_sort(hostAtoms.begin(), hostAtoms.end(),
//			         [](const ClSphAtom &a) { return a.zIndex; });

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


			std::cout << "atomsN = " << atomsN
			          << " AABB:" << glm::to_string(sizes)
			          << " min:" << glm::to_string(min)
			          << " max:" << glm::to_string(max)
			          << " gridTable = " << hostGridTable.size() << " gridTableN = " << gridTableN
			          << std::endl;

			std::cout << "Go! " << std::endl;

			hrc::time_point gpuS = hrc::now();


			cl::Buffer deviceAtoms(hostAtoms.begin(), hostAtoms.end(), false);
			cl::Buffer deviceGridTable(hostGridTable.begin(), hostGridTable.end(), true);
			cl::Buffer deviceResult(CL_MEM_WRITE_ONLY, hostAtoms.size() * sizeof(ClSphParticle));

			auto lambdaKernel = cl::KernelFunctor<
					const ClSphConfig &, cl::Buffer &, uint, cl::Buffer &, uint
			>(program, "sph_lambda");
			auto deltaKernel = cl::KernelFunctor<
					const ClSphConfig &, cl::Buffer &, uint, cl::Buffer &, uint
			>(program, "sph_delta");
			auto finaliseKernel = cl::KernelFunctor<
					const ClSphConfig &, cl::Buffer &, cl::Buffer &
			>(program, "sph_finalise");

			const ClSphConfig config = {
					.scale = scale,
					.dt =0.0083f,
					.iteration = static_cast<size_t>(iter)
			};

			std::cout << "SPH run" << std::endl;

			try {

				for (size_t itr = 0; itr < iter; ++itr) {
					lambdaKernel(
							cl::EnqueueArgs(cl::NDRange(atomsN)),
							config,
							deviceAtoms, static_cast<uint>(atomsN),
							deviceGridTable, static_cast<uint>(gridTableN));

					deltaKernel(
							cl::EnqueueArgs(cl::NDRange(atomsN)),
							config,
							deviceAtoms, static_cast<uint>(atomsN),
							deviceGridTable, static_cast<uint>(gridTableN));
				}
				finaliseKernel(cl::EnqueueArgs(cl::NDRange(atomsN)),
				               config, deviceAtoms, deviceResult);

			} catch (const std::exception &exc) {
				std::cerr << "Kernel failed to execute: " << exc.what() << std::endl;
				throw;
			}


			std::vector<ClSphParticle> copiedParticles(atomsN);
//			std::vector<ClSphAtom> copiedAtoms(atomsN);

			cl::copy(deviceResult, copiedParticles.begin(), copiedParticles.end());


//#pragma omp parallel for
			for (int i = 0; i < particles.size(); ++i) {
				const ClSphParticle &p = copiedParticles[i];
				fluid::Particle<T, N> &particle = particles[i];

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

			hrc::time_point gpuE = hrc::now();

			auto aabbCpy = std::chrono::duration_cast<std::chrono::nanoseconds>(
					aabbCpyE - aabbCpyS).count();
			auto zCurve = std::chrono::duration_cast<std::chrono::nanoseconds>(
					zCurveE - zCurveS).count();
			auto sort = std::chrono::duration_cast<std::chrono::nanoseconds>(sortE - sortS).count();
			auto gt = std::chrono::duration_cast<std::chrono::nanoseconds>(gtE - gtS).count();
			auto gpu = std::chrono::duration_cast<std::chrono::nanoseconds>(gpuE - gpuS).count();
			std::cout
					<< "\tCPU aabbCpy= " << (aabbCpy / 1000000.0) << "ms\n"
					<< "\tCPU zCurve = " << (zCurve / 1000000.0) << "ms\n"
					<< "\tCPU sort   = " << (sort / 1000000.0) << "ms\n"
					<< "\tCPU gt     = " << (gt / 1000000.0) << "ms\n"
					<< "\tGPU jacobi = " << (gpu / 1000000.0) << "ms\n"
					<< std::endl;


		}


	};
}


#endif //LIBFLUID_CLSPH_HPP
