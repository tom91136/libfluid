#ifndef LIBFLUID_CL_H
#define LIBFLUID_CL_H

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_ENABLE_EXCEPTIONS

#include "sph.h"
#include <CL/cl2.hpp>
#include <iostream>
#include <sstream>
#include <vector>
#include <memory>
#include <algorithm>

using glm::tvec3;


class CLOps {

public:

	CLOps() {

	}

	void showAtom(Atom atom) {
		std::cout << "P " << atom.id << " " << atom.mass << " " << atom.lambda << std::endl;
	}


	inline cl_float3 float3(float x, float y, float z) { return {{x, y, z}}; }

	inline cl_float4 float4(float x, float y, float z, float w) { return {{x, y, z, w}}; }

	inline cl_float3 float3(float x) { return {{x, x, x}}; }

	inline cl_float4 float4(float x) { return {{x, x, x, x}}; }


	void doIt() {

		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		auto platform = cl::Platform::getDefault();
		std::cout << "Enumerating OpenCL platforms:" << std::endl;
		for (auto &p : platforms) {
			std::cout << "Platform"
					  << (platform == p ? "(Default):" : ":")
					  << p.getInfo<CL_PLATFORM_NAME>()
					  << "\n\tVendor     : " << p.getInfo<CL_PLATFORM_VENDOR>()
					  << "\n\tVersion    : " << p.getInfo<CL_PLATFORM_VERSION>()
					  << "\n\tProfile    : " << p.getInfo<CL_PLATFORM_PROFILE>()
					  << "\n\tExtensions : " << p.getInfo<CL_PLATFORM_EXTENSIONS>()
					  << std::endl;
		}


		std::ifstream t("../sph.cl");
		std::stringstream source;
		source << t.rdbuf();


		cl::Program program(source.str());

		try {
			program.build("-cl-std=CL1.2 -w");
		} catch (...) {
			cl_int buildErr = CL_SUCCESS;
			auto buildInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
			for (auto &pair : buildInfo) {
				std::cerr << pair.second << std::endl << std::endl;
			}
			throw;
		}


		cl_float4 f4 = float4(1.f, 1.f, 1.f, 1.f);

		Atom a = {
				.now = f4, .mass = 10.f, .id = 10,
				.neighbourOffset = 0,
				.neighbourCount = 3
		};
		Atom b = {
				.now = f4, .mass = 20.f, .id = 20,
				.neighbourOffset = 3,
				.neighbourCount = 4
		};


		std::cout << "Go! " << sizeof(Atom) << std::endl;


		std::vector<Atom> ys = {a, b};
		std::vector<uint> ns = {1, 2, 3, 4, 5, 6, 42};
		cl::Buffer atoms(ys.begin(), ys.end(), false);
		cl::Buffer neighbours(ns.begin(), ns.end(), true);


		cl::Buffer output(CL_MEM_READ_WRITE, ys.size() * sizeof(cl_float));


		Config config = {.h = 0.1, .scale = 100.0, .iteration = 1,};


		auto sphKernel =
				cl::KernelFunctor<
						decltype(config) &,
						cl::Buffer &, uint,
						cl::Buffer &,
						cl::Buffer &
				>(program, "sph");


		cl_int error;

		std::cout << "SPH run" << std::endl;

		try {
			sphKernel(
					cl::EnqueueArgs(cl::NDRange(ys.size())),
					config,
					atoms, (uint) ys.size(),
					neighbours, output
			);
		} catch (const std::exception &exc) {
			std::cerr << "Kernel failed to execute: " << exc.what() << std::endl;
			throw;

		}


		std::vector<float> actual(ys.size(), -1);
		std::vector<Atom> actualAtoms(ys.size());

		cl::copy(output, actual.begin(), actual.end());
		cl::copy(atoms, actualAtoms.begin(), actualAtoms.end());


		for (auto b : actualAtoms) {
			showAtom(b);
		}

		std::cout << "vector: [ ";
		for (auto a : actual) {
			std::cout << a << ",";
		}

		std::cout << "]" << std::endl;


		std::cout << "Done" << std::endl;


	}


};


#endif //LIBFLUID_CL_H
