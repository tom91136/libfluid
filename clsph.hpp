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


	inline glm::tvec3<float> glmT(cl_float3 v) {
		return glm::tvec3<float>(v.x, v.y, v.z);
	}

	inline glm::tvec3<uint> glmT(cl_int3 v) {
		return glm::tvec3<uint>(v.x, v.y, v.z);
	}

	inline cl_float3 glmT(glm::tvec3<float> v) {
		return float3(v.x, v.y, v.z);
	}
}

namespace clsph {

	template<typename T, typename N>
	class CLOps {


	private:
		cl::Program program;

	public:

		CLOps() {
			std::ifstream t("../clsph_kernel.cl");
			std::stringstream source;
			source << t.rdbuf();
			program = cl::Program(source.str());
		}

	private :
		void showAtom(ClSphAtom atom) {
			std::cout << "P " << atom.id << " " << atom.mass << " " << atom.lambda << std::endl;
		}


	public :

		void enumeratePlatformToCout() {
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

			std::cout << "Using default platform: `" << platform.getInfo<CL_PLATFORM_NAME>() << "`"
			          << std::endl;
		}

	private:

		void showBuildInfo() {
			cl_int buildErr = CL_SUCCESS;
			auto buildInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
			for (auto &pair : buildInfo) {
				std::cerr << pair.second << std::endl << std::endl;
			}
		}

	public:

		void prepareProgram() {
			try {
				program.build("-cl-std=CL1.2 -w "
				              "-cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math"
				              " -I /home/tom/libfluid");
			} catch (...) {
				showBuildInfo();
				throw;
			}
			showBuildInfo();
			std::cout << "Program compiled" << std::endl;
		}


		ClSphType resolve(fluid::Type t) {
			switch (t) {
				case fluid::Type::Fluid: return ClSphType::Fluid;
				case fluid::Type::Obstacle: return ClSphType::Obstacle;
			}
		}


		static inline glm::tvec3<size_t> snap(glm::tvec3<N> a) {
			return glm::tvec3<int>(round(a.x / 0.1), round(a.y / 0.1), round(a.z / 0.1));
		}

		static inline size_t hash(glm::tvec3<size_t> a) {
			size_t res = 17;
			res = res * 31 + a.x;
			res = res * 31 + a.y;
			res = res * 31 + a.z;
			return res;
		}

#define  NEIGHBOUR_SIZE  (1 + (2) * 2) // n[L] + C + n[R]
		constexpr static float H = 0.1f;
		constexpr static float H2 = H * 2;
		constexpr static float HH = H * H;
		constexpr static float HHH = H * H * H;


		typedef struct Entry2 {
			glm::tvec3<size_t> key;
			size_t value;
		} __attribute__ ((aligned)) Entry2;

		static inline size_t
		findSlotUnfenced(const std::vector<Entry2> &buckets, const glm::tvec3<size_t> x) {
			size_t i = hash(x) % buckets.size();

//			glm::notEqual(buckets[i].key , x)

			while (buckets[i].value != 0 && buckets[i].key != x)
				i = (i + 1) % buckets.size();
			return i;
		}


		void nn_phase_size(const std::vector<ClSphAtom> &atoms, std::vector<Entry2> &buckets) {
			static float NEIGHBOURS[NEIGHBOUR_SIZE] = {-H2, -H, 0, H, H2};

			for (uint i = 0; i < buckets.size(); ++i) {
				buckets[i].value = 0;
			}


			for (const auto &atom : atoms) {
				for (size_t x = 0; x < NEIGHBOUR_SIZE; x++)
					for (size_t y = 0; y < NEIGHBOUR_SIZE; y++)
						for (size_t z = 0; z < NEIGHBOUR_SIZE; z++) {
							glm::tvec3<size_t> snapped = snap(
									clutil::glmT(atom.now) +
									glm::tvec3<float>(NEIGHBOURS[x], NEIGHBOURS[y], NEIGHBOURS[z]));

//							std::cout << "Offset: " << glm::to_string(snapped) << std::endl;


							size_t slot = findSlotUnfenced(buckets, snapped);
							Entry2 &e = buckets[slot];
							if (e.value == 0) e.key = snapped;
							e.value++;
						}
			}

		}


		std::vector<ClSphResult>
		run(const std::vector<fluid::Atom<T, N>> &origin, size_t iter, N scale) {

			using hrc = std::chrono::high_resolution_clock;
			hrc::time_point sphs = hrc::now();

			typedef tvec3<float> v3n;

			std::vector<ClSphAtom> as(origin.size());
			std::vector<uint> ns;

			size_t prefixSum = 0;
			for (int i = 0; i < as.size(); ++i) {
				const fluid::Atom<T, N> &a = origin[i];
				as[i] = (ClSphAtom{
						.id = a.particle->t,
						.type = resolve(a.particle->type),
						.mass = a.particle->mass,
						.position = clutil::glmT(a.particle->position),
						.velocity = clutil::glmT(a.velocity),
						.now = clutil::glmT(a.now),
						.neighbourOffset = static_cast<size_t >(prefixSum),
						.neighbourCount = static_cast<size_t >(a.neighbours->size())
				});
				prefixSum += a.neighbours->size();
				for (int j = 0; j < a.neighbours->size(); ++j) {
					ns.push_back((*a.neighbours)[j]->particle->t);
				}
			}

			std::cout << "As:" << as.size() << " NS: " << ns.size() << " PSum:" << prefixSum
			          << std::endl;


			std::cout << "Go! " << std::endl;


//			std::vector<ClSphResult> backingRes(as.size());

			cl::Buffer atoms(as.begin(), as.end(), false);
			cl::Buffer neighbours(ns.begin(), ns.end(), true);
			cl::Buffer output(CL_MEM_WRITE_ONLY, as.size() * sizeof(ClSphResult));
//			cl::Buffer output(backingRes.begin(), backingRes.end(), false);

			hrc::time_point NNS = hrc::now();

			cl::Buffer nnEntries(CL_MEM_READ_WRITE, as.size() * sizeof(Entry) * 27);

			auto nnPhaseSize =
					cl::KernelFunctor<
							cl::Buffer &,
							cl::Buffer &,
							uint
					>(program, "nn_phase_size");


			try {
				nnPhaseSize(
						cl::EnqueueArgs(
								cl::NDRange(as.size())
						),
						atoms,
						nnEntries, (uint) as.size() * 27
				);
			} catch (const std::exception &exc) {
				std::cerr << "Kernel failed to execute: " << exc.what() << std::endl;
				throw;

			}


			std::vector<Entry> entryHost(as.size() * 27);

			cl::copy(nnEntries, entryHost.begin(), entryHost.end());

			int gmax = 0;
			for (const auto &a : entryHost) {
				if ((a.value) != 0) {

					std::cout << "GPU >> "
					          << " k=" << glm::to_string(clutil::glmT(a.key))
					          << " v=" << (a.value) << "\n";
					gmax += a.value;
				}
			}

			std::cout << "GPU >> t=" << gmax << std::endl;


			gmax= 0;
			std::vector<Entry2> alt(as.size() * 27);
			nn_phase_size(as, alt);
			for (const auto &a : alt) {
				if ((a.value) != 0) {

					std::cout << "CPU >> "
					          << " k=" << glm::to_string((a.key))
					          << " v=" << (a.value) << "\n";
					gmax += a.value;
				}
			}

			std::cout << "CPU >> t=" << gmax << std::endl;








//			std::cout << "]" << std::endl;

			hrc::time_point NNE = hrc::now();
			auto NNEL = std::chrono::duration_cast<std::chrono::nanoseconds>(NNE - NNS).count();

			std::cout << "NNH " << (NNEL / 1000000.0) << "ms\n";

			ClSphConfig config = {
//					.h = 0.1,
					.scale = scale,
					.dt =0.0083f,
					.iteration = static_cast<size_t>(iter)
			};

			auto sphKernel =
					cl::KernelFunctor<
							decltype(config) &,
							cl::Buffer &, uint,
							cl::Buffer &,
							cl::Buffer &
					>(program, "sph");

			std::cout << "SPH run" << std::endl;

			cl_int error;
			try {
				sphKernel(
						cl::EnqueueArgs(
								cl::NDRange(as.size())
						),
						config,
						atoms, (uint) as.size(),
						neighbours,
						output,
						error
				);
			} catch (const std::exception &exc) {
				std::cerr << "Kernel failed to execute: " << exc.what() << std::endl;
				throw;
			}

			std::cout << "Error? " << error << std::endl;


			std::vector<ClSphResult> actual(as.size());
			std::vector<ClSphAtom> actualAtoms(as.size());

			cl::copy(output, actual.begin(), actual.end());
//			cl::copy(atoms, actualAtoms.begin(), actualAtoms.end());


//		for (auto b : actualAtoms) {
//			showAtom(b);
//		}


			hrc::time_point sphe = hrc::now();
			auto solve = std::chrono::duration_cast<std::chrono::nanoseconds>(sphe - sphs).count();

			std::cout << "vector in " << (solve / 1000000.0) << "ms: [ \n";


//			for (auto a : actual) {
//				std::cout << "GPU >> "
//				          << " p=" << glm::to_string(clutil::glmT(a.position))
//				          << " v=" << glm::to_string(clutil::glmT(a.velocity)) << "\n";
//			}
//
//			std::cout << "]" << std::endl;
			std::cout << "Done" << std::endl;
//
			return actual;
		}


	};
}


#endif //LIBFLUID_CLSPH_HPP
