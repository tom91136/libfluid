#ifndef LIBFLUID_CL_H
#define LIBFLUID_CL_H

#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION

#include "boost/compute.hpp"
#include "boost/compute/algorithm/copy.hpp"
#include "boost/compute/algorithm/accumulate.hpp"
#include "boost/compute/container/vector.hpp"
#include "boost/compute/types/fundamental.hpp"
#include "sph.h"

namespace compute = boost::compute;
using glm::tvec3;


using compute::float4_;
using compute::float_;
using compute::int_;


class CLOps {

private :
	compute::device device;
	compute::context context;
	compute::command_queue queue;
	compute::kernel nn3k;
public:

	CLOps(const compute::device &device) :
			device(device), context(device), queue(context, device) {

	}

	compute::program make_shnn(const compute::context &context) {
		const char source[] =
				BOOST_COMPUTE_STRINGIZE_SOURCE(
				// @formatter:off
				kernel void shnn(
							float radius,
							int size,
							global const float4 *input,
							int maxSize,
							global int *output) {

				    // TODO
				}
				// @formatter:on
				);
		return compute::program::build_with_source(source, context);
	}

	compute::program make_sph_program(const compute::context &context) {
		const char source[] =
				BOOST_COMPUTE_STRINGIZE_SOURCE(
				// @formatter:off

				static const float VD = 0.49;// Velocity dampening;
				static const float RHO = 6378; // Reference density;
				static const float EPSILON = 0.00000001;
				static const float CFM_EPSILON = 600.0; // CFM propagation;

				static const float C = 0.00001;
				static const float VORTICITY_EPSILON = 0.0005;
				static const float CorrK = 0.0001;
				static const float CorrN = 4.f;

				struct Atom{
					// TODO impl


					N mass;tvec3<N> now;
		N lambda;
		tvec3<N> deltaP;
		tvec3<N> omega;
		tvec3<N> velocity;
				};

				kernel void sph() {
					const int gid = get_global_id(0);
					// TODO impl
				}
				// @formatter:on
				);
		return compute::program::build_with_source(source, context);
	}

	compute::program make_nn3_program(const compute::context &context) {
		const char source[] =
				BOOST_COMPUTE_STRINGIZE_SOURCE(
				// @formatter:off
				kernel void nn3(
							float radius,
							int size,
							global const float4 *input,
							int maxSize,
							global int *output) {

					const int gid = get_global_id(0);

					float4 A = input[gid];
					int j = 0;
					for (int i = 0; i < size; ++i) {
						float4 B = input[i];
						float d = fast_distance(A, B);
						if(d < radius) output[ (j) + gid * maxSize ] +=i;
					}
				}
				// @formatter:on
				);
		return compute::program::build_with_source(source, context);
	}


	void showAtom(Atom atom) {
		std::cout << "P " << atom.id << " " << atom.mass << " " << std::endl;
	}


	inline cl_float3 float3(float x, float y, float z) { return {{x, y, z}}; }

	inline cl_float4 float4(float x, float y, float z, float w) { return {{x, y, z, w}}; }

	inline cl_float3 float3(float x) { return {{x, x, x}}; }

	inline cl_float4 float4(float x) { return {{x, x, x, x}}; }


	void doIt() {

		auto prog = compute::program::create_with_source_file("../sph.cl", context);
		prog.build();
		compute::kernel sph(prog, "sph");

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
		compute::vector<Atom> atoms(ys.begin(), ys.end(), queue);
		compute::vector<uint> neighbours(ns.begin(), ns.end(), queue);

		compute::vector<float> output(ys.size(), -1.f, queue);




		sph.set_arg(0, atoms.get_buffer());
		sph.set_arg(1, neighbours.get_buffer());
		sph.set_arg(2, (uint) ys.size());
		sph.set_arg(3, output.get_buffer());


		queue.enqueue_1d_range_kernel(sph, 0, ys.size(), 0);


		std::vector<float> out(ys.size());

		boost::compute::copy(
				output.begin(), output.end(), out.begin(),
				queue
		);
		std::cout << "vector: [ " << std::endl;
		for (auto a : out) {
			std::cout << a << ",";
		}
		std::cout << "]" << std::endl;


	}


	template<typename N>
	std::vector<std::vector<size_t>> nn3(
			N radius,
			size_t maxN,
			const std::vector<tvec3<N>> &xs) {
		using compute::float4_;

		if (nn3k == nullptr) {
			compute::program program = make_nn3_program(context);
			program.build();
			compute::kernel nn3k(program, "nn3");
			this->nn3k = nn3k;
		}

		auto prog = compute::program::create_with_source_file("../sph.cl", context);
		prog.build();

		std::vector<float4_> ys;

		std::transform(xs.begin(), xs.end(), std::back_inserter(ys),
		               [](const tvec3<N> &v) { return float4_(v.x, v.y, v.z, 0.f); });

		compute::vector<float4_> input(ys.begin(), ys.end(), queue);
		compute::vector<int> output((size_t) maxN * xs.size(), -1, queue);

		std::cout << "n=" << ys.size() << std::endl;
		// XXX must cast to proper type
		nn3k.set_arg(0, (float) radius);
		nn3k.set_arg(1, (int) xs.size());
		nn3k.set_arg(2, input.get_buffer());
		nn3k.set_arg(3, (int) maxN);
		nn3k.set_arg(4, output.get_buffer());

		queue.enqueue_1d_range_kernel(nn3k, 0, xs.size(), 0);


		std::vector<int> out(maxN * xs.size());


		boost::compute::copy(
				output.begin(), output.end(), out.begin(),
				queue
		);
		std::cout << "vector: [ ";
//		for (int a : output) {
//			std::cout << a << std::endl;
//		}
		std::cout << "]" << std::endl;

		// TODO impl
		return std::vector<std::vector<size_t>>();
	}


};


#endif //LIBFLUID_CL_H
