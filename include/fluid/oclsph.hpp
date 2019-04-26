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

#define DEBUG


namespace ocl {

	using glm::tvec3;

	typedef cl::Buffer ClSphConfigStruct;
	typedef cl::Buffer ClMcConfigStruct;

	typedef cl::KernelFunctor<
			ClSphConfigStruct &,
			cl::Buffer &, cl::Buffer &, uint, // zIdx, grid, gridN

			cl::Buffer &, // pstar
			cl::Buffer &, // mass
			cl::Buffer & // lambda
	> SphLambdaKernel;

	typedef cl::KernelFunctor<
			ClSphConfigStruct &,
			cl::Buffer &, cl::Buffer &, uint, // zIdx, grid, gridN
			cl::Buffer &, uint, // mesh + N

			cl::Buffer &, // pstar
			cl::Buffer &, // lambda
			cl::Buffer &, // pos
			cl::Buffer &, // vel
			cl::Buffer &  // deltap
	> SphDeltaKernel;

	typedef cl::KernelFunctor<
			ClSphConfigStruct &,

			cl::Buffer &, // pstar
			cl::Buffer &, // pos
			cl::Buffer &  // vel
	> SphFinaliseKernel;

	typedef cl::KernelFunctor<
			ClSphConfigStruct &, ClMcConfigStruct &,
			cl::Buffer &, uint,
			float3, uint3, uint3,
			cl::Buffer &, cl::Buffer &
	> SphEvalLatticeKernel;

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
			uint,
			cl::Buffer &, cl::Buffer &, cl::Buffer &,
			cl::Buffer &, cl::Buffer &, cl::Buffer &
	> McEvalKernel;


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

		SphLambdaKernel lambdaKernel;
		SphDeltaKernel deltaKernel;
		SphFinaliseKernel finaliseKernel;
		SphEvalLatticeKernel evalLatticeKernel;
		McSizeKernel mcSizeKernel;
		McEvalKernel mcEvalKernel;

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
				evalLatticeKernel(clsph, "sph_evalLattice"),
				mcSizeKernel(clsph, "mc_size"),
				mcEvalKernel(clsph, "mc_eval") {
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

		std::vector<geometry::MeshTriangle<float>> runKernel(clutil::Stopwatch &watch,
		                                                     ClMcConfig &mcConfig,
		                                                     ClSphConfig &sphConfig,
		                                                     std::vector<uint> &hostGridTable,
		                                                     std::vector<ClSphTraiangle> &hostColliderMesh,
		                                                     ClSphAtoms &atoms,
		                                                     const tvec3<float> minExtent,
		                                                     const tvec3<size_t> extent,
		                                                     std::vector<float3> &hostPosition,
		                                                     std::vector<float3> &hostVelocity,
		                                                     surface::Lattice<float4> &hostLattice
		) {
			auto kernel_copy = watch.start("\t[GPU] kernel_copy");

			const tvec3<size_t> sampleSize(
					hostLattice.xSize(), hostLattice.ySize(), hostLattice.zSize());


			const uint colliderMeshN = static_cast<uint>(hostColliderMesh.size());


			cl::Buffer colliderMesh = colliderMeshN == 0 ?
			                          cl::Buffer(context, CL_MEM_READ_WRITE, 1) :
			                          cl::Buffer(queue, hostColliderMesh.begin(),
			                                     hostColliderMesh.end(), true);


			const uint gridTableN = static_cast<uint>(hostGridTable.size());
			cl::Buffer gridTable(queue, hostGridTable.begin(), hostGridTable.end(), true);

			cl::Buffer zIndex(queue, atoms.zIndex.begin(), atoms.zIndex.end(), true);
			cl::Buffer mass(queue, atoms.mass.begin(), atoms.mass.end(), true);
			cl::Buffer pStar(queue, atoms.pStar.begin(), atoms.pStar.end(), false);
			cl::Buffer position(queue, atoms.position.begin(), atoms.position.end(), false);
			cl::Buffer velocity(queue, atoms.velocity.begin(), atoms.velocity.end(), false);

			cl::Buffer deltaP(context, CL_MEM_READ_WRITE, sizeof(float3) * atoms.size);
			cl::Buffer lambda(context, CL_MEM_READ_WRITE, sizeof(float) * atoms.size);
			cl::Buffer lattice(context, CL_MEM_READ_WRITE, sizeof(float4) * hostLattice.size());

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



			size_t kernelWorkGroupSize = mcSizeKernel
					.getKernel()
					.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);


			const auto maxCube = sampleSize - tvec3<size_t>(1);
			const auto latticeRange = maxCube.x * maxCube.y * maxCube.z;


			size_t workGroupSize = kernelWorkGroupSize;
			size_t nWorkGroup = std::ceil(static_cast<float>(latticeRange) / workGroupSize);


			std::cout << "Samples:" << glm::to_string(maxCube)
			<< " L=" << latticeRange
			<< " HL:" << hostLattice.size()
			<< " WG:" << kernelWorkGroupSize <<
			" nWG:"<< nWorkGroup << " Fl:" << (((float) latticeRange) / workGroupSize)
			<< "\n";
//
			const auto localSum = cl::Local(sizeof(uint) * workGroupSize);

			std::vector<uint> hostPsum(nWorkGroup, 0);

			cl::Buffer pSum(context, CL_MEM_WRITE_ONLY, nWorkGroup * sizeof(uint));


			auto p_sum = watch.start("\t[GPU] mc_psum");





			mcSizeKernel(
					cl::EnqueueArgs(queue,
					            cl::NDRange(nWorkGroup * workGroupSize), cl::NDRange(workGroupSize)),
					mcConfig_,
					clutil::uvec3ToCl(sampleSize),
					lattice,
					localSum,
					pSum
			);


			cl::copy(queue, pSum, hostPsum.begin(), hostPsum.end());
			uint acc = 0;
			for (int j = 0; j < nWorkGroup; ++j) acc += hostPsum[j];

			std::cout << "Acc=" << acc << std::endl;
#ifdef DEBUG
			queue.finish();
#endif
			p_sum();

			auto gpu_mc = watch.start("\t[GPU] gpu_mc");


			std::vector<uint> zero{0};

			cl::Buffer trigCounter(context, zero.begin(), zero.end(), false);

			cl::Buffer outVxs(context, CL_MEM_WRITE_ONLY, sizeof(float3) * acc);
			cl::Buffer outVys(context, CL_MEM_WRITE_ONLY, sizeof(float3) * acc);
			cl::Buffer outVzs(context, CL_MEM_WRITE_ONLY, sizeof(float3) * acc);

			cl::Buffer outNxs(context, CL_MEM_WRITE_ONLY, sizeof(float3) * acc);
			cl::Buffer outNys(context, CL_MEM_WRITE_ONLY, sizeof(float3) * acc);
			cl::Buffer outNzs(context, CL_MEM_WRITE_ONLY, sizeof(float3) * acc);


			mcEvalKernel(
					cl::EnqueueArgs(queue,
					                cl::NDRange(latticeRange)),
					sphConfig_, mcConfig_,
					clutil::vec3ToCl(minExtent), clutil::uvec3ToCl(sampleSize),
					lattice,
					trigCounter,
					acc,
					outVxs, outVys, outVzs,
					outNxs, outNys, outNzs
			);

#ifdef DEBUG
			queue.finish();
#endif

			gpu_mc();

			std::vector<float3> hostOutVxs(acc);
			std::vector<float3> hostOutVys(acc);
			std::vector<float3> hostOutVzs(acc);

			std::vector<float3> hostOutNxs(acc);
			std::vector<float3> hostOutNys(acc);
			std::vector<float3> hostOutNzs(acc);

			cl::copy(queue, outVxs, hostOutVxs.begin(), hostOutVxs.end());
			cl::copy(queue, outVys, hostOutVys.begin(), hostOutVys.end());
			cl::copy(queue, outVzs, hostOutVzs.begin(), hostOutVzs.end());

			cl::copy(queue, outNxs, hostOutNxs.begin(), hostOutNxs.end());
			cl::copy(queue, outNys, hostOutNys.begin(), hostOutNys.end());
			cl::copy(queue, outNzs, hostOutNzs.begin(), hostOutNzs.end());

			std::vector<surface::MeshTriangle<float>> triangles(acc);
#pragma omp parallel for
			for (int i = 0; i < acc; ++i) {
				triangles[i].v0 = clutil::clToVec3<float>(hostOutVxs[i]);
				triangles[i].v1 = clutil::clToVec3<float>(hostOutVys[i]);
				triangles[i].v2 = clutil::clToVec3<float>(hostOutVzs[i]);
				triangles[i].n0 = clutil::clToVec3<float>(hostOutNxs[i]);
				triangles[i].n1 = clutil::clToVec3<float>(hostOutNys[i]);
				triangles[i].n2 = clutil::clToVec3<float>(hostOutNzs[i]);
			}


			std::cout
					<< "LatticeDataN:" << (float) (sizeof(float4) * hostLattice.size()) / 1000000.0
					<< "MB MCGPuN:" << (float) (sizeof(float3) * acc * 6) / 1000000.0 << "MB \n";


			auto kernel_return = watch.start("\t[GPU] kernel_return");
//			cl::copy(queue, lattice, hostLattice.begin(), hostLattice.end());
			cl::copy(queue, position, hostPosition.begin(), hostPosition.end());
			cl::copy(queue, velocity, hostVelocity.begin(), hostVelocity.end());
#ifdef DEBUG
			queue.finish();
#endif
			kernel_return();

			return triangles;
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
			mcConfig.isolevel = 100;
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
						xs.emplace_back(source.tag, fluid::Type::Fluid, 1, pos,
						                config.constantForce);
					}
				}
			}

			xs.erase(std::remove_if(xs.begin(), xs.end(),
			                        [&config](const fluid::Particle<size_t, float> &x) {

				                        for (const fluid::Drain<float> &drain: config.drains) {
					                        // FIXME needs to actually erase at surface, not shperically
					                        if (glm::distance(drain.centre, x.position) < 100) {
						                        return true;
					                        }
				                        }

				                        return false;
			                        }), xs.end());

			sourceDrain();

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

#ifdef DEBUG
			std::cout << "Atoms = " << atomsN
			          << " Extent = " << to_string(minExtent) << " -> " << to_string(extent)
			          << " GridTable = " << gridTableN
			          << std::endl;
#endif

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
			std::vector<surface::MeshTriangle<float>> triangles;
			auto kernel_exec = watch.start("\t[GPU] ===total===");
			try {
				triangles = runKernel(watch, mcConfig, clConfig,
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
//
//			auto march = watch.start("CPU mc");
//
//			std::vector<surface::MeshTriangle<float>> triangles =
//					sampleLattice(100, config.scale,
//					              minExtent, h / mcConfig.sampleResolution, hostLattice);
//
//			march();


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
