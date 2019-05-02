#ifndef LIBFLUID_CLSPH_HPP
#define LIBFLUID_CLSPH_HPP

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_ENABLE_EXCEPTIONS

#define GLM_ENABLE_EXPERIMENTAL

#define CURVE_UINT3_TYPE glm::tvec3<size_t>
#define CURVE_UINT3_CTOR(x, y, z) (glm::tvec3<size_t>((x), (y), (z)))

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
#include "curves.h"
#include "mc.h"
#include "ska_sort.hpp"

//#define DEBUG

namespace ocl {

	using glm::tvec3;
	using clutil::TypedBuffer;
	using clutil::RW;
	using clutil::RO;
	using clutil::WO;

	typedef cl::Buffer ClSphConfigStruct;
	typedef cl::Buffer ClMcConfigStruct;


	typedef cl::KernelFunctor<
			ClSphConfigStruct &,
			cl::Buffer &, cl::Buffer &, uint, // zIdx, grid, gridN
			cl::Buffer &, // type

			cl::Buffer &, // pstar
			cl::Buffer &, // original colour
			cl::Buffer & // colour
	> SphDiffuseKernel;

	typedef cl::KernelFunctor<
			ClSphConfigStruct &,
			cl::Buffer &, cl::Buffer &, uint, // zIdx, grid, gridN
			cl::Buffer &, // type

			cl::Buffer &, // pstar
			cl::Buffer &, // mass
			cl::Buffer & // lambda
	> SphLambdaKernel;

	typedef cl::KernelFunctor<
			ClSphConfigStruct &,
			cl::Buffer &, cl::Buffer &, uint, // zIdx, grid, gridN
			cl::Buffer &, // type

			cl::Buffer &, uint, // mesh + N

			cl::Buffer &, // pstar
			cl::Buffer &, // lambda
			cl::Buffer &, // pos
			cl::Buffer &, // vel
			cl::Buffer &  // deltap
	> SphDeltaKernel;

	typedef cl::KernelFunctor<
			ClSphConfigStruct &,
			cl::Buffer &, // type

			cl::Buffer &, // pstar
			cl::Buffer &, // pos
			cl::Buffer &  // vel
	> SphFinaliseKernel;

	typedef cl::KernelFunctor<
			ClSphConfigStruct &, ClMcConfigStruct &,
			cl::Buffer &, uint,
			cl::Buffer &, // type

			float3, uint3, uint3,
			cl::Buffer &, cl::Buffer &,
			cl::Buffer &, cl::Buffer &
	> McLatticeKernel;

	typedef cl::KernelFunctor<
			ClMcConfigStruct &,
			uint3,
			cl::Buffer &,
			cl::LocalSpaceArg,
			cl::Buffer &
	> McSizeKernel;

	typedef cl::KernelFunctor<
			ClSphConfigStruct &, ClMcConfigStruct &,
			float3, uint3,

			cl::Buffer &,
			cl::Buffer &,

			cl::Buffer &,
			uint,
			cl::Buffer &, cl::Buffer &, cl::Buffer &,
			cl::Buffer &, cl::Buffer &, cl::Buffer &,
			cl::Buffer &, cl::Buffer &, cl::Buffer &
	> McEvalKernel;


	struct ClSphAtoms {
		const size_t size;
		std::vector<uint> zIndex;
		std::vector<ClSphType> type;
		std::vector<float> mass;
		std::vector<uchar4> colour;
		std::vector<float3> pStar;
		std::vector<float3> position;
		std::vector<float3> velocity;

		explicit ClSphAtoms(size_t size) : size(size), zIndex(size),
		                                   type(size), mass(size), colour(size),
		                                   pStar(size), position(size), velocity(size) {}
	};

	struct PartiallyAdvected {
		uint zIndex{};
		float3 pStar{};
		fluid::Particle<size_t, float> particle;
		PartiallyAdvected() = default;
		explicit PartiallyAdvected(uint zIndex, float3 pStar,
		                           const fluid::Particle<size_t, float> &particle) :
				zIndex(zIndex), pStar(pStar), particle(particle) {}
	};

	class SphSolver : public fluid::SphSolver<size_t, float> {

	private:

		const float h;
		const cl::Device device;
		const cl::Context ctx;
		const cl::Program clsph;

		cl::CommandQueue queue;

		SphDiffuseKernel sphDiffuseKernel;
		SphLambdaKernel sphLambdaKernel;
		SphDeltaKernel sphDeltaKernel;
		SphFinaliseKernel sphFinaliseKernel;
		McLatticeKernel mcLatticeKernel;
		McSizeKernel mcSizeKernel;
		McEvalKernel mcEvalKernel;

	public:
		explicit SphSolver(float h, const std::string &kernelPath, const cl::Device &device) :
				h(h),
				device(device),
				ctx(cl::Context(device)),
				clsph(clutil::loadProgramFromFile(
						ctx,
						kernelPath + "oclsph_kernel.h",
						kernelPath,
						"-DSPH_H=((float)" + std::to_string(h) + ")")),
				//TODO check capability
				queue(cl::CommandQueue(ctx, device, cl::QueueProperties::None)),
				sphDiffuseKernel(clsph, "sph_diffuse"),
				sphLambdaKernel(clsph, "sph_lambda"),
				sphDeltaKernel(clsph, "sph_delta"),
				sphFinaliseKernel(clsph, "sph_finalise"),
				mcLatticeKernel(clsph, "mc_lattice"),
				mcSizeKernel(clsph, "mc_size"),
				mcEvalKernel(clsph, "mc_eval") {
			checkSize();
		}

	private:

		static inline ClSphType resolveType(fluid::Type t) {
			switch (t) {
				case fluid::Type::Fluid: return ClSphType::Fluid;
				case fluid::Type::Obstacle: return ClSphType::Obstacle;
				default: throw std::logic_error("unhandled branch (fluid::Type->ClSphType)");
			}
		}

		static inline fluid::Type resolveType(ClSphType t) {
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
				TypedBuffer<size_t, WO> buffer(ctx, _SIZES_LENGTH);
				cl::KernelFunctor<cl::Buffer &>(clsph, "check_size")
						(cl::EnqueueArgs(queue, cl::NDRange(_SIZES_LENGTH)), buffer.actual);
				buffer.drainTo(queue, actual);
				queue.finish();
			} catch (cl::Error &exc) {
				std::cerr << "Kernel failed to execute: " << exc.what() << " -> "
				          << clResolveError(exc.err()) << "(" << exc.err() << ")" << std::endl;
				throw;
			}

#ifdef DEBUG
			std::cout << "Expected(" << _SIZES_LENGTH << ")="
			          << clutil::mkString<size_t>(expected,
			                                      [](auto x) { return std::to_string(x); })
			          << std::endl;
			std::cout << "Actual(" << _SIZES_LENGTH << ")  ="
			          << clutil::mkString<size_t>(actual, [](auto x) { return std::to_string(x); })
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

			std::cout << "Acc2=" << 0 << "Lattice:" << lattice.size()
			          << std::endl;
			return triangles;
		}


		std::vector<PartiallyAdvected> advectAndCopy(const fluid::Config<float> &config,
		                                             std::vector<fluid::Particle<size_t, float>> &xs) {
			std::vector<PartiallyAdvected> advected(xs.size());


#pragma omp parallel for
			for (int i = 0; i < static_cast<int>(xs.size()); ++i) {
				fluid::Particle<size_t, float> &p = xs[i];
				advected[i].particle = p;
				advected[i].pStar = clutil::vec3ToCl(p.position / config.scale);

				if (p.type == fluid::Type::Obstacle) continue;

				tvec3<float> combinedForce = p.mass * config.constantForce;
				for (const fluid::Well<float> &well : config.wells) {
					const float distSquared = glm::distance2(well.centre, p.position);
					const tvec3<float> rHat = (well.centre + p.position) / std::sqrt(distSquared);
					const tvec3<float> forceWell = glm::clamp(
							(rHat * (well.force * p.mass)) / distSquared,
							-10.f, 10.f);
					combinedForce += forceWell;
				}

				p.velocity = combinedForce * config.dt + p.velocity;
				advected[i].pStar = clutil::vec3ToCl(
						(p.velocity * config.dt) + (p.position / config.scale));

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

		inline void finishQueue() {
#ifdef DEBUG
			queue.finish();
#endif
		}

		std::vector<geometry::MeshTriangle<float>> runMcKernels(
				clutil::Stopwatch &watch,
				const tvec3<size_t> sampleSize,
				TypedBuffer<ClSphConfig, RO> &sphConfig,
				TypedBuffer<ClMcConfig, RO> &mcConfig,
				TypedBuffer<uint, RO> &gridTable,
				TypedBuffer<ClSphType, RO> type,
				TypedBuffer<float3, RW> &particlePositions,
				TypedBuffer<uchar4, RW> &particleColours,
				const tvec3<float> minExtent,
				const tvec3<size_t> extent
		) {

			const uint gridTableN = static_cast<uint>(gridTable.length);

			const size_t latticeN = sampleSize.x * sampleSize.y * sampleSize.z;
			TypedBuffer<float4, RW> latticePNs(ctx, latticeN);
			TypedBuffer<uchar4, RW> latticeCs(ctx, latticeN);

			const size_t kernelWorkGroupSize = mcSizeKernel
					.getKernel()
					.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

			const tvec3<size_t> marchRange = sampleSize - tvec3<size_t>(1);
			const size_t marchVolume = marchRange.x * marchRange.y * marchRange.z;

			size_t workGroupSize = kernelWorkGroupSize;
			size_t numWorkGroup = std::ceil(static_cast<float>(marchVolume) / workGroupSize);

#ifdef DEBUG
			std::cout << "[<>]Samples:" << glm::to_string(marchRange)
			          << " MarchVol=" << marchVolume
			          << " WG:" << kernelWorkGroupSize
			          << " nWG:" << numWorkGroup << "\n";

#endif
			auto create_field = watch.start("\t[GPU] mc-field");

			mcLatticeKernel(
					cl::EnqueueArgs(queue,
					                cl::NDRange(sampleSize.x, sampleSize.y, sampleSize.z)),
					sphConfig.actual, mcConfig.actual,
					gridTable.actual, gridTableN,
					type.actual,
					clutil::vec3ToCl(minExtent), clutil::uvec3ToCl(sampleSize),
					clutil::uvec3ToCl(extent),
					particlePositions.actual, particleColours.actual,
					latticePNs.actual, latticeCs.actual
			);
			finishQueue();
			create_field();


			auto partial_trig_sum = watch.start("\t[GPU] mc_psum");

			TypedBuffer<uint, WO> partialTrigSum(ctx, numWorkGroup);
			mcSizeKernel(
					cl::EnqueueArgs(queue,
					                cl::NDRange(numWorkGroup * workGroupSize),
					                cl::NDRange(workGroupSize)),
					mcConfig.actual,
					clutil::uvec3ToCl(sampleSize),
					latticePNs.actual,
					cl::Local(sizeof(uint) * workGroupSize),
					partialTrigSum.actual
			);
			std::vector<uint> hostPartialTrigSum(numWorkGroup, 0);
			partialTrigSum.drainTo(queue, hostPartialTrigSum);

			uint numTrigs = 0;
			for (uint j = 0; j < numWorkGroup; ++j) numTrigs += hostPartialTrigSum[j];

			finishQueue();
			partial_trig_sum();
#ifdef DEBUG
			std::cout << "[<>]Acc=" << numTrigs << std::endl;
#endif
			auto gpu_mc = watch.start("\t[GPU] gpu_mc");
			std::vector<surface::MeshTriangle<float>> triangles(numTrigs);

			if (numTrigs != 0) {

				std::vector<uint> zero{0};
				TypedBuffer<uint, RW> trigCounter(queue, zero);

				TypedBuffer<float3, WO> outVxs(ctx, numTrigs);
				TypedBuffer<float3, WO> outVys(ctx, numTrigs);
				TypedBuffer<float3, WO> outVzs(ctx, numTrigs);

				TypedBuffer<float3, WO> outNxs(ctx, numTrigs);
				TypedBuffer<float3, WO> outNys(ctx, numTrigs);
				TypedBuffer<float3, WO> outNzs(ctx, numTrigs);

				TypedBuffer<uchar4, WO> outCxs(ctx, numTrigs);
				TypedBuffer<uchar4, WO> outCys(ctx, numTrigs);
				TypedBuffer<uchar4, WO> outCzs(ctx, numTrigs);

				mcEvalKernel(
						cl::EnqueueArgs(queue,
						                cl::NDRange(marchVolume)),
						sphConfig.actual, mcConfig.actual,
						clutil::vec3ToCl(minExtent), clutil::uvec3ToCl(sampleSize),
						latticePNs.actual,
						latticeCs.actual,
						trigCounter.actual,
						numTrigs,
						outVxs.actual, outVys.actual, outVzs.actual,
						outNxs.actual, outNys.actual, outNzs.actual,
						outCxs.actual, outCys.actual, outCzs.actual
				);

				finishQueue();
				gpu_mc();

				auto gpu_mc_drain = watch.start("\t[GPU] gpu_mc drain");


				std::vector<float3> hostOutVxs(numTrigs);
				std::vector<float3> hostOutVys(numTrigs);
				std::vector<float3> hostOutVzs(numTrigs);
				std::vector<float3> hostOutNxs(numTrigs);
				std::vector<float3> hostOutNys(numTrigs);
				std::vector<float3> hostOutNzs(numTrigs);
				std::vector<uchar4> hostOutCxs(numTrigs);
				std::vector<uchar4> hostOutCys(numTrigs);
				std::vector<uchar4> hostOutCzs(numTrigs);

				outVxs.drainTo(queue, hostOutVxs);
				outVys.drainTo(queue, hostOutVys);
				outVzs.drainTo(queue, hostOutVzs);

				outNxs.drainTo(queue, hostOutNxs);
				outNys.drainTo(queue, hostOutNys);
				outNzs.drainTo(queue, hostOutNzs);

				outCxs.drainTo(queue, hostOutCxs);
				outCys.drainTo(queue, hostOutCys);
				outCzs.drainTo(queue, hostOutCzs);

				finishQueue();
				gpu_mc_drain();
				auto gpu_mc_assem = watch.start("\t[GPU] gpu_mc assem");

#pragma omp parallel for
				for (int i = 0; i < static_cast<int>(numTrigs); ++i) {
					triangles[i].v0 = clutil::clToVec3<float>(hostOutVxs[i]);
					triangles[i].v1 = clutil::clToVec3<float>(hostOutVys[i]);
					triangles[i].v2 = clutil::clToVec3<float>(hostOutVzs[i]);
					triangles[i].n0 = clutil::clToVec3<float>(hostOutNxs[i]);
					triangles[i].n1 = clutil::clToVec3<float>(hostOutNys[i]);
					triangles[i].n2 = clutil::clToVec3<float>(hostOutNzs[i]);
					triangles[i].c.x = clutil::packARGB(hostOutCxs[i]);
					triangles[i].c.y = clutil::packARGB(hostOutCys[i]);
					triangles[i].c.z = clutil::packARGB(hostOutCzs[i]);
				}
				gpu_mc_assem();
#ifdef DEBUG
				std::cout
						<< "[<>] LatticeDataN:"
						<< (float) (sizeof(float4) * marchVolume) / 1000000.0 << "MB"
						<< " MCGPuN:" << (float) (sizeof(float3) * numTrigs * 6) / 1000000.0
						<< "MB \n";
#endif
			}
			return triangles;
		}


		void runSphKernel(clutil::Stopwatch &watch,
		                  size_t iterations,
		                  TypedBuffer<ClSphConfig, RO> &sphConfig,
		                  TypedBuffer<uint, RO> &gridTable,
		                  TypedBuffer<uint, RO> &zIndex,
		                  TypedBuffer<ClSphType, RO> type,
		                  TypedBuffer<float, RO> &mass,
		                  TypedBuffer<uchar4, RO> &colour,

		                  TypedBuffer<float3, RW> &pStar,
		                  TypedBuffer<float3, RW> &deltaP,
		                  TypedBuffer<float, RW> &lambda,
		                  TypedBuffer<uchar4, RW> &diffused,

		                  TypedBuffer<float3, RW> &position,
		                  TypedBuffer<float3, RW> &velocity,

		                  std::vector<ClSphTraiangle> &hostColliderMesh
		) {
			auto kernel_copy = watch.start("\t[GPU] kernel_copy");


			const uint colliderMeshN = static_cast<uint>(hostColliderMesh.size());

			cl::Buffer colliderMesh = colliderMeshN == 0 ?
			                          cl::Buffer(ctx, CL_MEM_READ_WRITE, 1) :
			                          cl::Buffer(queue, hostColliderMesh.begin(),
			                                     hostColliderMesh.end(), true);

			const uint gridTableN = static_cast<uint>(gridTable.length);

			finishQueue();
			kernel_copy();

			const auto localRange = cl::NDRange();
			const auto globalRange = cl::NDRange(position.length);


			auto diffuse = watch.start("\t[GPU] sph-diffuse");

			sphDiffuseKernel(cl::EnqueueArgs(queue, globalRange, localRange),
			                 sphConfig.actual, zIndex.actual, gridTable.actual, gridTableN,
			                 type.actual,
			                 pStar.actual, colour.actual, diffused.actual
			);

			finishQueue();
			diffuse();


			auto lambda_delta = watch.start("\t[GPU] sph-lambda/delta*" +
			                                std::to_string(iterations));

			for (size_t itr = 0; itr < iterations; ++itr) {
				sphLambdaKernel(cl::EnqueueArgs(queue, globalRange, localRange),
				                sphConfig.actual, zIndex.actual, gridTable.actual, gridTableN,
				                type.actual,
				                pStar.actual, mass.actual, lambda.actual
				);
				sphDeltaKernel(cl::EnqueueArgs(queue, globalRange, localRange),
				               sphConfig.actual, zIndex.actual, gridTable.actual, gridTableN,
				               type.actual,
				               colliderMesh, colliderMeshN,
				               pStar.actual, lambda.actual, position.actual, velocity.actual,
				               deltaP.actual
				);
			}
			finishQueue();
			lambda_delta();

			auto finalise = watch.start("\t[GPU] sph-finalise");
			sphFinaliseKernel(cl::EnqueueArgs(queue, globalRange, localRange),
			                  sphConfig.actual,
			                  type.actual,
			                  pStar.actual, position.actual, velocity.actual
			);
			finishQueue();
			finalise();
		}


	public:
		std::vector<surface::MeshTriangle<float>> advance(const fluid::Config<float> &config,
		                                                  std::vector<fluid::Particle<size_t, float>> &xs,
		                                                  const std::vector<fluid::MeshCollider<float>> &colliders) override {


			clutil::Stopwatch watch = clutil::Stopwatch("CPU advance");
			auto total = watch.start("Advance ===total===");

			ClMcConfig mcConfig;
			mcConfig.isolevel = config.isolevel;
			mcConfig.sampleResolution = config.resolution;
			mcConfig.particleSize = 60.f;
			mcConfig.particleInfluence = 0.5;

			auto sourceDrain = watch.start("CPU source+drain");


			const float spacing = (h * config.scale / 2);
			for (const fluid::Source<float> &source : config.sources) {
				const float size = std::sqrt(source.rate);
				const int width = std::floor(size);
				const int depth = std::ceil(size);
				const auto offset = source.centre - (tvec3<float>(width, 0, depth) / 2 * spacing);
				for (int x = 0; x < width; ++x) {
					for (int z = 0; z < depth; ++z) {
						auto pos = offset + tvec3<float>(x, 0, z) * spacing;
						xs.emplace_back(source.tag, fluid::Type::Fluid, 1, source.colour,
						                pos, source.velocity);
					}
				}
			}

			xs.erase(std::remove_if(xs.begin(), xs.end(),
			                        [&config](const fluid::Particle<size_t, float> &x) {

										if(x.type == fluid::Type::Obstacle) return false;

				                        for (const fluid::Drain<float> &drain: config.drains) {
					                        // FIXME needs to actually erase at surface, not shperically
					                        if (glm::distance(drain.centre, x.position) <
					                            drain.width) {
						                        return true;
					                        }
				                        }
				                        return false;
			                        }), xs.end());

			sourceDrain();


			if (xs.empty()) {
				std::cout << "Particles depleted" << std::endl;
				std::this_thread::sleep_for(std::chrono::milliseconds(5));
				return {};
			}

			auto advect = watch.start("CPU advect+copy");
			std::vector<PartiallyAdvected> advected = advectAndCopy(config, xs);
			const size_t atomsN = advected.size();
			advect();


			auto bound = watch.start("CPU bound+zindex");

			ClSphConfig sphConfig;
			tvec3<float> minExtent;
			tvec3<size_t> extent;

			std::tie(sphConfig, minExtent, extent) = computeBoundAndZindex(config, advected);

			bound();

			auto sortz = watch.start("CPU sortz");

			std::sort(advected.begin(), advected.end(),
			          [](const PartiallyAdvected &l, const PartiallyAdvected &r) {
				          return l.zIndex < r.zIndex;
			          });

			// ska_sort(hostAtoms.begin(), hostAtoms.end(),
			//  [](const ClSphAtom &a) { return a.zIndex; });

			sortz();


			auto gridtable = watch.start("CPU gridtable");


			const size_t gridTableN = zCurveGridIndexAtCoord(extent.x, extent.y, extent.z);

#ifdef DEBUG
			std::cout << "Atoms = " << atomsN
			          << " Extent = " << glm::to_string(minExtent) << " -> " << to_string(extent)
			          << " GridTable = " << gridTableN
			          << std::endl;
#endif

			std::vector<uint> hostGridTable(gridTableN);
			uint gridIndex = 0;
			for (size_t i = 0; i < gridTableN; ++i) {
				hostGridTable[i] = gridIndex;
				while (gridIndex != atomsN && advected[gridIndex].zIndex == i) {
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

#ifdef DEBUG
			std::cout << "Collider trig: " << hostColliderMesh.size() << "\n";
#endif
			collider_concat();


			auto kernel_alloc = watch.start("CPU host alloc+copy");

			const tvec3<size_t> sampleSize = tvec3<size_t>(
					glm::floor(tvec3<float>(extent) *
					           mcConfig.sampleResolution)) + tvec3<size_t>(1);


			ClSphAtoms atoms(advected.size());
#pragma omp parallel for
			for (int i = 0; i < static_cast<int>(advected.size()); ++i) {
				atoms.zIndex[i] = advected[i].zIndex;
				atoms.pStar[i] = advected[i].pStar;
				atoms.type[i] = resolveType(advected[i].particle.type);
				atoms.mass[i] = advected[i].particle.mass;
				atoms.colour[i] = clutil::unpackARGB(advected[i].particle.colour);
				atoms.position[i] = clutil::vec3ToCl(advected[i].particle.position);
				atoms.velocity[i] = clutil::vec3ToCl(advected[i].particle.velocity);
			}

			std::vector<surface::MeshTriangle<float>> triangles;
			auto kernel_exec = watch.start("\t[GPU] ===total===");
			try {

				auto sphConfig_ = TypedBuffer<ClSphConfig, RO>::ofStruct(ctx, sphConfig);
				auto mcConfig_ = TypedBuffer<ClMcConfig, RO>::ofStruct(ctx, mcConfig);

				TypedBuffer<uint, RO> gridTable(queue, hostGridTable);
				TypedBuffer<uint, RO> zIndex(queue, atoms.zIndex);
				TypedBuffer<ClSphType, RO> type(queue, atoms.type);
				TypedBuffer<float, RO> mass(queue, atoms.mass);
				TypedBuffer<uchar4, RO> colour(queue, atoms.colour);
				TypedBuffer<float3, RW> pStar(queue, atoms.pStar);

				TypedBuffer<float3, RW> deltaP(ctx, atoms.size);
				TypedBuffer<float, RW> lambda(ctx, atoms.size);
				TypedBuffer<uchar4, RW> diffused(ctx, atoms.size);

				TypedBuffer<float3, RW> position(queue, atoms.position);
				TypedBuffer<float3, RW> velocity(queue, atoms.velocity);

				kernel_alloc();

				runSphKernel(watch, sphConfig.iteration, sphConfig_,
				             gridTable, zIndex, type, mass, colour,
				             pStar, deltaP, lambda, diffused,
				             position, velocity,
				             hostColliderMesh);

				triangles = runMcKernels(watch, sampleSize, sphConfig_, mcConfig_,
				                         gridTable, type, position, diffused, minExtent, extent);


				std::vector<float3> hostPosition(advected.size());
				std::vector<float3> hostVelocity(advected.size());
				std::vector<uchar4> hostDiffused(advected.size());
				position.drainTo(queue, hostPosition);
				velocity.drainTo(queue, hostVelocity);
				diffused.drainTo(queue, hostDiffused);

				auto write_back = watch.start("write_back");
#pragma omp parallel for
				for (int i = 0; i < static_cast<int>(xs.size()); ++i) {
					xs[i].id = advected[i].particle.id;
					xs[i].type = advected[i].particle.type;
					xs[i].mass = advected[i].particle.mass;

					if (advected[i].particle.type == fluid::Fluid) {
						xs[i].colour = clutil::packARGB(hostDiffused[i]);
						xs[i].position = clutil::clToVec3<float>(hostPosition[i]);
						xs[i].velocity = clutil::clToVec3<float>(hostVelocity[i]);
					} else {
						xs[i].colour = advected[i].particle.colour;
						xs[i].position = advected[i].particle.position;
						xs[i].velocity = advected[i].particle.velocity;
					}

				}
				write_back();

			} catch (const cl::Error &exc) {
				std::cerr << "Kernel failed to execute: " << exc.what() << " -> "
				          << clResolveError(exc.err()) << "(" << exc.err() << ")" << std::endl;
				throw exc;
			}
			kernel_exec();


//
//			auto march = watch.start("CPU mc");
//
//			std::vector<surface::MeshTriangle<float>> triangles =
//					sampleLattice(100, config.scale,
//					              minExtent, h / mcConfig.sampleResolution, hostLattice);
//
//			march();


			total();


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
			std::cout << "Advance complete" << std::endl;
#endif
			return triangles;
		}


	};

}

#endif //LIBFLUID_CLSPH_HPP
