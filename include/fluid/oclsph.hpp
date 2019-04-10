#ifndef LIBFLUID_CLSPH_HPP
#define LIBFLUID_CLSPH_HPP

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_ENABLE_EXCEPTIONS

#define GLM_ENABLE_EXPERIMENTAL


#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
#include <memory>
#include <algorithm>

#include <sys/types.h>
#include <sys/stat.h>

#include "cl2.hpp"
#include "clutils.hpp"
#include "fluid.hpp"
#include "oclsph_type.h"
#include "zcurve.h"
#include "mc.h"
#include "ska_sort.hpp"

#define DEBUG


namespace ocl {

	using glm::tvec3;

	typedef cl::KernelFunctor<
			cl::Buffer, cl::Buffer &, cl::Buffer &, uint, // zIdx, grid, gridN

			cl::Buffer &, // pstar
			cl::Buffer &, // mass
			cl::Buffer & // lambda
	> LambdaKernel;

	typedef cl::KernelFunctor<
			cl::Buffer, cl::Buffer &, cl::Buffer &, uint, // zIdx, grid, gridN
			cl::Buffer &, uint, // mesh + N

			cl::Buffer &, // pstar
			cl::Buffer &, // lambda
			cl::Buffer &, // pos
			cl::Buffer &, // vel
			cl::Buffer &  // deltap
	> DeltaKernel;

	typedef cl::KernelFunctor<
			cl::Buffer,

			cl::Buffer &, // pstar
			cl::Buffer &, // pos
			cl::Buffer &  // vel
	> FinaliseKernel;

	typedef cl::KernelFunctor<
			cl::Buffer, cl::Buffer,
			cl::Buffer &, uint,
			float3, uint3, uint3,
			cl::Buffer &, cl::Buffer &
	> EvalLatticeKernel;


	struct ClSphAtoms {
		const size_t size;
		std::vector<uint> zIndex;
		std::vector<float> mass;
		std::vector<float3> pStar;
		std::vector<float3> position;
		std::vector<float3> velocity;

		explicit ClSphAtoms(size_t size) : size(size), zIndex(size), mass(size),
										   pStar(size), position(size), velocity(size) {}
	};

	struct PartiallyAdvected {
		uint zIndex;
		float3 pStar;
		fluid::Particle<size_t, float> particle;
		PartiallyAdvected() {}
		explicit PartiallyAdvected(uint zIndex, float3 pStar,
								   const fluid::Particle<size_t, float> &particle) :
				zIndex(zIndex), pStar(pStar), particle(particle) {}
	};

	class SphSolver : public fluid::SphSolver<size_t, float> {

	private:

		const float h;
		const cl::Device device;
		const cl::Context context;
		const cl::Program clsph;

		cl::CommandQueue queue;

		LambdaKernel lambdaKernel;
		DeltaKernel deltaKernel;
		FinaliseKernel finaliseKernel;
		EvalLatticeKernel evalLatticeKernel;

	public:
		explicit SphSolver(float h, const std::string &kernelPath, const cl::Device &device) :
				h(h),
				device(device),
				context(cl::Context(device)),
				clsph(clutil::loadProgramFromFile(
						context,
						kernelPath + "oclsph_kernel.h",
						kernelPath,
						"-DSPH_H=((float)" + std::to_string(h) + ")")),
				//TODO check capability
				queue(cl::CommandQueue(context, device, cl::QueueProperties::OutOfOrder)),
				lambdaKernel(clsph, "sph_lambda"),
				deltaKernel(clsph, "sph_delta"),
				finaliseKernel(clsph, "sph_finalise"),
				evalLatticeKernel(clsph, "sph_evalLattice") {
			checkSize();
		}

	private:

		static inline ClSphType resolve(fluid::Type t) {
			switch (t) {
				case fluid::Type::Fluid: return ClSphType::Fluid;
				case fluid::Type::Obstacle: return ClSphType::Obstacle;
				default: throw std::logic_error("unhandled branch (fluid::Type->ClSphType)");
			}
		}

		static inline fluid::Type resolve(ClSphType t) {
			switch (t) {
				case ClSphType::Fluid: return fluid::Type::Fluid;
				case ClSphType::Obstacle: return fluid::Type::Obstacle;
				default: throw std::logic_error("unhandled branch (fluid::Type<-ClSphType)");
			}
		}

		static inline std::string to_string(tvec3<size_t> v) {
			return "(" +
				   std::to_string(v.x) + "," +
				   std::to_string(v.y) + "," +
				   std::to_string(v.z) +
				   ")";
		}

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


		std::vector<surface::MeshTriangle<float>> sampleLattice(
				float isolevel, float scale,
				const tvec3<float> min, float step,
				const surface::Lattice<float4> &lattice) {

			std::vector<surface::MeshTriangle<float>> triangles;


			// XXX ICC needs perfectly nested loops like this for omp collapse
			const size_t latticeX = lattice.xSize() - 1;
			const size_t latticeY = lattice.ySize() - 1;
			const size_t latticeZ = lattice.zSize() - 1;

#ifndef _MSC_VER
#pragma omp declare reduction (merge : std::vector<geometry::MeshTriangle<float>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for collapse(3) reduction(merge: triangles)
#endif
			for (size_t x = 0; x < latticeX; ++x) {
				for (size_t y = 0; y < latticeY; ++y) {
					for (size_t z = 0; z < latticeZ; ++z) {
						std::array<float, 8> vertices{};
						std::array<tvec3<float>, 8> normals{};
						std::array<tvec3<float>, 8> pos{};
						for (size_t j = 0; j < 8; ++j) {
							tvec3<size_t> offset =
									tvec3<size_t>(x, y, z) + surface::CUBE_OFFSETS[j];
							float4 v = lattice(offset.x, offset.y, offset.z);
							vertices[j] = v.s0;
							normals[j] = tvec3<float>(v.s1, v.s2, v.s3);
							pos[j] = (tvec3<float>(offset) * step + min) * scale;

						}
						surface::marchSingle(isolevel, vertices, normals, pos, triangles);
					}
				}
			}
			return triangles;
		}

		std::vector<PartiallyAdvected> advectAndCopy(const fluid::Config<float> &config,
													 std::vector<fluid::Particle<size_t, float>> &xs) {
			std::vector<PartiallyAdvected> advected(xs.size());
#pragma omp parallel for
			for (int i = 0; i < static_cast<int>(xs.size()); ++i) {
				fluid::Particle<size_t, float> &p = xs[i];
				p.velocity = (p.mass * config.constantForce) * config.dt + p.velocity;
				advected[i].pStar = clutil::vec3ToCl(
						(p.velocity * config.dt) + (p.position / config.scale));
				advected[i].particle = p;
			}
			return advected;
		}

		const std::tuple<ClSphConfig, tvec3<float>, tvec3<size_t> > computeBoundAndZindex(
				const fluid::Config<float> &config,
				std::vector<PartiallyAdvected> &advection) const {

			tvec3<float> minExtent(std::numeric_limits<float>::max());
			tvec3<float> maxExtent(std::numeric_limits<float>::min());
#if _OPENMP > 201307
#pragma omp declare reduction(glmMin: tvec3<float>: omp_out = glm::min(omp_in, omp_out))
#pragma omp declare reduction(glmMax: tvec3<float>: omp_out = glm::max(omp_in, omp_out))
#pragma omp parallel for reduction(glmMin:minExtent) reduction(glmMax:maxExtent)
#endif
			for (int i = 0; i < static_cast<int>(advection.size()); ++i) {
				const float3 pStar = advection[i].pStar;
				minExtent = glm::min(clutil::clToVec3<float>(pStar), minExtent);
				maxExtent = glm::max(clutil::clToVec3<float>(pStar), maxExtent);
			}

			const float padding = h * 2;
			minExtent -= padding;
			maxExtent += padding;

#pragma omp parallel for
			for (int i = 0; i < static_cast<int>(advection.size()); ++i) {
				const float3 pStar = advection[i].pStar;
				advection[i].zIndex = zCurveGridIndexAtCoord(
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


		template<typename T>
		cl::Buffer readOnlyStruct(T &t) {
			return cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(T), &t);
		}

		void runKernel(clutil::Stopwatch &watch,
					   ClMcConfig &mcConfig, ClSphConfig &sphConfig,
					   std::vector<uint> &hostGridTable,
					   std::vector<ClSphTraiangle> &hostColliderMesh,
					   ClSphAtoms &atoms,
					   const tvec3<float> minExtent, const tvec3<size_t> extent,
					   std::vector<float3> &hostPosition,
					   std::vector<float3> &hostVelocity,
					   surface::Lattice<float4> &hostLattice
		) {
			auto kernel_copy = watch.start("\t[GPU] kernel_copy");

			const tvec3<size_t> sampleSize(
					hostLattice.xSize(), hostLattice.ySize(), hostLattice.zSize());


			const uint colliderMeshN = static_cast<uint>(hostColliderMesh.size());


			cl::Buffer colliderMesh = colliderMeshN == 0 ?
					cl::Buffer(context, CL_MEM_READ_WRITE, 1):
					cl::Buffer(queue, hostColliderMesh.begin(), hostColliderMesh.end(), true);


			const uint gridTableN = static_cast<uint>(hostGridTable.size());
			cl::Buffer gridTable(queue, hostGridTable.begin(), hostGridTable.end(), true);

			cl::Buffer zIndex(queue, atoms.zIndex.begin(), atoms.zIndex.end(), true);
			cl::Buffer mass(queue, atoms.mass.begin(), atoms.mass.end(), true);
			cl::Buffer pStar(queue, atoms.pStar.begin(), atoms.pStar.end(), false);
			cl::Buffer position(queue, atoms.position.begin(), atoms.position.end(), false);
			cl::Buffer velocity(queue, atoms.velocity.begin(), atoms.velocity.end(), false);

			cl::Buffer deltaP(context, CL_MEM_READ_WRITE, sizeof(float3) * atoms.size);
			cl::Buffer lambda(context, CL_MEM_READ_WRITE, sizeof(float) * atoms.size);
			cl::Buffer lattice(context, CL_MEM_WRITE_ONLY, sizeof(float4) * hostLattice.size());

			cl::Buffer sphConfig_ = readOnlyStruct<ClSphConfig>(sphConfig);
			cl::Buffer mcConfig_ = readOnlyStruct<ClMcConfig>(mcConfig);

#ifdef DEBUG
			queue.finish();
#endif
			kernel_copy();

			auto lambda_delta = watch.start(
					"\t[GPU] sph-lambda/delta*" + std::to_string(sphConfig.iteration));

			const cl::NDRange &range = cl::NDRange();

			for (size_t itr = 0; itr < sphConfig.iteration; ++itr) {
				lambdaKernel(cl::EnqueueArgs(queue, cl::NDRange(atoms.size), range),
							 sphConfig_, zIndex, gridTable, gridTableN,
							 pStar, mass, lambda
				);
				deltaKernel(cl::EnqueueArgs(queue, cl::NDRange(atoms.size), range),
							sphConfig_, zIndex, gridTable, gridTableN,
							colliderMesh, colliderMeshN,
							pStar, lambda, position, velocity, deltaP
				);
			}
#ifdef DEBUG
			queue.finish();
#endif
			lambda_delta();

			auto finalise = watch.start("\t[GPU] sph-finalise");

			finaliseKernel(cl::EnqueueArgs(queue, cl::NDRange(atoms.size), range),
						   sphConfig_,
						   pStar, position, velocity
			);
#ifdef DEBUG
			queue.finish();
#endif
			finalise();

			auto create_field = watch.start("\t[GPU] mc-field");

			evalLatticeKernel(
					cl::EnqueueArgs(queue,
									cl::NDRange(sampleSize.x, sampleSize.y, sampleSize.z)),
					sphConfig_, mcConfig_,
					gridTable, gridTableN,
					clutil::vec3ToCl(minExtent), clutil::uvec3ToCl(sampleSize),
					clutil::uvec3ToCl(extent),
					position,
					lattice
			);
#ifdef DEBUG
			queue.finish();
#endif
			create_field();

			auto kernel_return = watch.start("\t[GPU] kernel_return");
			cl::copy(queue, lattice, hostLattice.begin(), hostLattice.end());
			cl::copy(queue, position, hostPosition.begin(), hostPosition.end());
			cl::copy(queue, velocity, hostVelocity.begin(), hostVelocity.end());
#ifdef DEBUG
			queue.finish();
#endif
			kernel_return();
		}

		void overwrite(std::vector<fluid::Particle<size_t, float>> &xs,
					   const std::vector<PartiallyAdvected> &advected,
					   const std::vector<float3> &position,
					   const std::vector<float3> &velocity) {
#pragma omp parallel for
			for (int i = 0; i < static_cast<int>(xs.size()); ++i) {
				xs[i].id = advected[i].particle.id;
				xs[i].type = advected[i].particle.type;
				xs[i].mass = advected[i].particle.mass;
				xs[i].position = clutil::clToVec3<float>(position[i]);
				xs[i].velocity = clutil::clToVec3<float>(velocity[i]);
			}
		}


	public:
		std::vector<surface::MeshTriangle<float>> advance(const fluid::Config<float> &config,
														  std::vector<fluid::Particle<size_t, float>> &xs,
														  const std::vector<fluid::MeshCollider<float>> &colliders) override {


			clutil::Stopwatch watch = clutil::Stopwatch("CPU advance");
			auto total = watch.start("Advance ===total===");

			ClMcConfig mcConfig;
			mcConfig.sampleResolution = 2.f;
			mcConfig.particleSize = 60.f;
			mcConfig.particleInfluence = 0.5;


			auto advect = watch.start("CPU advect+copy");
			std::vector<PartiallyAdvected> advection = advectAndCopy(config, xs);
			const size_t atomsN = advection.size();
			advect();


			auto bound = watch.start("CPU bound+zindex");

			ClSphConfig clConfig;
			tvec3<float> minExtent;
			tvec3<size_t> extent;

			std::tie(clConfig, minExtent, extent) = computeBoundAndZindex(config, advection);

			bound();

			auto sortz = watch.start("CPU sortz");

			std::sort(advection.begin(), advection.end(),
					  [](const PartiallyAdvected &l, const PartiallyAdvected &r) {
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
				while (gridIndex != atomsN && advection[gridIndex].zIndex == i) {
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
					trig.a = clutil::vec3ToCl(xs[j].v0);
					trig.b = clutil::vec3ToCl(xs[j].v1);
					trig.c = clutil::vec3ToCl(xs[j].v2);
					hostColliderMesh.push_back(trig);
				}
			}

			std::cout << "Collider trig: " << hostColliderMesh.size() << "\n";

			collider_concat();

#ifdef DEBUG
			std::cout << "Atoms = " << atomsN
					  << " Extent = " << to_string(extent)
					  << " GridTable = " << hostGridTable.size()
					  << std::endl;
#endif

			auto kernel_alloc = watch.start("CPU host alloc+copy");

			const tvec3<size_t> sampleSize = tvec3<size_t>(
					glm::floor(tvec3<float>(extent) * mcConfig.sampleResolution)) +
											 tvec3<size_t>(1);

			std::vector<float3> hostPosition(advection.size());
			std::vector<float3> hostVelocity(advection.size());
			surface::Lattice<float4> hostLattice(sampleSize.x, sampleSize.y, sampleSize.z,
												 clutil::float4(-1));

			ClSphAtoms atoms(advection.size());
#pragma omp parallel for
			for (int i = 0; i < static_cast<int>(advection.size()); ++i) {
				atoms.zIndex[i] = advection[i].zIndex;
				atoms.pStar[i] = advection[i].pStar;
				atoms.mass[i] = advection[i].particle.mass;
				atoms.position[i] = clutil::vec3ToCl(advection[i].particle.position);
				atoms.velocity[i] = clutil::vec3ToCl(advection[i].particle.velocity);
			}

			kernel_alloc();

			auto kernel_exec = watch.start("\t[GPU] ===total===");
			try {
				runKernel(watch, mcConfig, clConfig,
						  hostGridTable,
						  hostColliderMesh,
						  atoms,
						  minExtent, extent,
						  hostPosition, hostVelocity, hostLattice);
			} catch (const cl::Error &exc) {
				std::cerr << "Kernel failed to execute: " << exc.what() << " -> "
						  << clResolveError(exc.err()) << "(" << exc.err() << ")" << std::endl;
				throw exc;
			}
			kernel_exec();

			auto write_back = watch.start("write_back");
			overwrite(xs, advection, hostPosition, hostVelocity);
			write_back();

			auto march = watch.start("CPU mc");

			std::vector<surface::MeshTriangle<float>> triangles =
					sampleLattice(100, config.scale,
								  minExtent, h / mcConfig.sampleResolution, hostLattice);

			march();


//			std::vector<unsigned short> outIdx;
//			std::vector<tvec3<float>> outVert;
//			hrc::time_point vbiStart = hrc::now();
//			surface::indexVBO2<float>(triangles, outIdx, outVert);
//			hrc::time_point vbiEnd = hrc::now();
//			auto vbi = duration_cast<nanoseconds>(vbiEnd - vbiStart).count();
//
//			std::cout
//					<< "\n\tTrigs = " << triangles.size()
//					<< "\n\tIdx   = " << outIdx.size()
//					<< "\n\tVert  = " << outVert.size()
//					<< "\n\tVBI   = " << (vbi / 1000000.0) << "ms\n"
//					<< std::endl;


			total();

#ifdef DEBUG
			std::cout << "MC lattice: " << hostLattice.size() << " Grid=" << to_string(extent)
					  << " res="
					  << hostLattice.xSize() << "x"
					  << hostLattice.ySize() << "x"
					  << hostLattice.zSize()
					  << std::endl;
			std::cout << watch << std::endl;
#endif

			return triangles;
		}


	};

}

#endif //LIBFLUID_CLSPH_HPP
