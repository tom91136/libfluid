#ifndef LIBFLUID_CPUSPH_HPP
#define LIBFLUID_CPUSPH_HPP

#include <ostream>
#include <functional>
#include <vector>
#include <numeric>
#include <chrono>
#include <unordered_map>
#include <iostream>
#include <algorithm>
#include "glm/ext.hpp"
#include "glm/glm.hpp"
#include "Octree.hpp"
#include "nanoflann.hpp"
#include <cstdlib>
#include <memory>
#include "clsph.hpp"
#include "fluid.hpp"


namespace cpusph {

	using fluid::Particle;
	using fluid::Atom;
	using fluid::Response;
	using fluid::Ray;

	template<typename T, typename N>
	class GridNN {


		glm::tvec3<N> pMin;
		glm::tvec3<N> pMax;
		glm::tvec3<int> cellMax;
		std::vector<glm::tvec3<N>> offsets = std::vector<glm::tvec3<N>>();

		const std::vector<Atom<T, N>> &xs;

	public:


		float rescale(float x, float a0, float a1, float b0, float b1) {
			return ((x - a0) / (a1 - a0)) * (b1 - b0) + b0;
		}

		int asIndex(glm::tvec3<int> v) {
			return v.x + (v.y * cellMax.x) + v.z * (cellMax.x * cellMax.y);
		}

//		glm::tvec3<int> subscript(Particle<T, N> p) {
//			int i = std::round(rescale(p.position.x,
//			                           pMin.x, pMax.x, 0.0f, cellMax.x - 1));
//			int j = std::round(rescale(p.position.y,
//			                           pMin.y, pMax.y, 0.0f, cellMax.y - 1));
//			int k = std::round(rescale(p.position.z,
//			                           pMin.z, pMax.z, 0.0f, cellMax.z - 1));
//
//			return glm::tvec3<int>(i, j, k);
//		}

		static constexpr auto CELL_SIZE = 0.1;

		struct AtomCompare {
			bool operator()(const glm::tvec3<int> &lhs, const glm::tvec3<int> &rhs) const {
				return memcmp((void *) &lhs, (void *) &rhs, sizeof(glm::tvec3<int>)) > 0;
			}
		};


		typedef std::unordered_multimap<size_t, size_t> SMap;
		SMap M;


		size_t hash(glm::tvec3<N> a) {
			size_t res = 17;
			res = res * 31 + std::hash<int>()(round(a.x / CELL_SIZE));
			res = res * 31 + std::hash<int>()(round(a.y / CELL_SIZE));
			res = res * 31 + std::hash<int>()(round(a.z / CELL_SIZE));
			return res;
		}

		GridNN(const std::vector<Atom<T, N>> &xs) : xs(xs) {




			N H = 0.1f;
			N H2 = H * 2;
			std::vector<N> off = {-H, 0.f, H};
			for (size_t x = 0; x < off.size(); ++x)
				for (size_t y = 0; y < off.size(); ++y)
					for (size_t z = 0; z < off.size(); ++z)
						offsets.push_back(tvec3<N>(off[x], off[y], off[z]));

			for (size_t i = 0; i < xs.size(); ++i) {
				const Atom<T, N> &p = xs[i];

				for (const auto ov : offsets) {
					M.insert({hash(p.now + ov) % xs.size(), p.particle->t});

				}


//				M.insert({hash(p.now), p.particle->t});
			}


		}


		size_t length() {
			return M.size();
		}


		void find(const Atom<T, N> &p, N radius, std::vector<size_t> &acc) {
			auto r2 = radius * radius;
//			for (const auto ov : offsets) {
			auto ret = M.equal_range(hash(p.now));
			for (auto it = ret.first; it != ret.second; ++it) {
				if (glm::distance2(xs[it->second].now, p.now) <= r2) {
					acc.push_back(it->second);
				}
			}
//			}
		}


	};

	template<typename T, typename N>
	struct AtomCloud {
		const std::vector<Atom<T, N>> &pts;
		const N scale;

		AtomCloud(const std::vector<Atom<T, N>> &pts, const N scale) : pts(pts), scale(scale) {}

		inline size_t kdtree_get_point_count() const { return pts.size(); }

		inline N kdtree_get_pt(const size_t idx, const size_t dim) const {

//			if (dim == 0) return pts[idx].now.x;
//			else if (dim == 1) return pts[idx].now.y;
//			else return pts[idx].now.z;

			return pts[idx].now[dim] * scale;
		}

		template<class BBOX>
		bool kdtree_get_bbox(BBOX &) const { return false; }
	};


	template<typename T, typename N>
	class SphSolver {

	public:
		clsph::CLOps<T, N> clo;

		explicit SphSolver(N h = 0.1, N scale = 1) : h(h), scale(scale) {
			clo = clsph::CLOps<T, N>();
			clo.enumeratePlatformToCout();
			clo.prepareProgram();
		}

	public:

		const N h;
		const N scale;

		static constexpr N VD = 0.49;// Velocity dampening;
		static constexpr N RHO = 6378; // Reference density;
		static constexpr N EPSILON = 0.00000001;
		static constexpr N CFM_EPSILON = 600.0; // CFM propagation;

		static constexpr N C = 0.00001;
		static constexpr N VORTICITY_EPSILON = 0.0005;
		static constexpr N CorrK = 0.0001;
		static constexpr N CorrN = 4.f;

	private:


		const N poly6Factor = 315.f / (64.f * glm::pi<N>() * std::pow(h, 9.f));
		const N spikyKernelFactor = -(45.f / (glm::pi<N>() * std::pow(h, 6.f)));
		const N h2 = h * 2;
		const N hp2 = h * h;
		const N CorrDeltaQ = 0.3f * h;
		const N p6DeltaQ = poly6Kernel(CorrDeltaQ);

		const N poly6Kernel(N r) {
			return r <= h ? poly6Factor * std::pow(hp2 - r * r, 3.f) : 0.f;
		}

		// 4.9.2 Spiky Kernel(Desbrun and Gascuel (1996))
		const tvec3<N> spikyKernelGradient(tvec3<N> x, tvec3<N> y, N &r) {
			r = glm::distance(x, y);
			return !(r <= h && r >= EPSILON) ?
			       tvec3<N>(0) :
			       (x - y) * (spikyKernelFactor * (std::pow(h - r, 2.f) / r));
		}

	public:

		void advance(N dt, size_t iteration,
		             std::vector<Particle<T, N>> &xs,
		             const std::function<tvec3<N>(const Particle<T, N> &)> &constForce,
		             const std::vector<std::function<const Response<N>(Ray<N> &)> > &colliders
		) {

			using namespace std::chrono;
			using hrc = high_resolution_clock;


			std::vector<Atom<T, N>> atoms;

			std::transform(xs.begin(), xs.end(), std::back_inserter(atoms),
			               [constForce, dt, this](Particle<T, N> &p) {
				               auto a = Atom<T, N>(&p);
				               a.velocity = constForce(p) * dt + p.velocity;
				               a.mass = p.mass;
				               a.now = (a.velocity * dt) + (p.position / scale);
				               return a;
			               });

			hrc::time_point gnn = hrc::now();
			GridNN<T, N> rtn = GridNN<T, N>(atoms);
			std::cout << "\tGNN done " << std::endl;
			hrc::time_point gnne = hrc::now();
			auto nng = duration_cast<nanoseconds>(gnne - gnn).count();
			std::cout << "\tGNN: " << (nng / 1000000.0) << "ms -> " << rtn.length() << std::endl;


			hrc::time_point nns = hrc::now();

			const N truncation = 30.f;
#ifndef FLANN


			std::vector<tvec3<N>> pts;
			std::transform(atoms.begin(), atoms.end(), std::back_inserter(pts),
						   [truncation](const Atom<T, N> &a) { return a.now * truncation; });

			unibn::Octree<tvec3<N>> octree;
			octree.initialize(pts);
#else

			using namespace nanoflann;
//			std::vector<tvec3<N>> pts;
//			std::transform(atoms.begin(), atoms.end(), std::back_inserter(pts),
//						   [thingy](const Atom<T, N> &a) { return a.now * thingy; });

			const AtomCloud<T, N> cloud = AtomCloud<T, N>(atoms, truncation);
			typedef KDTreeSingleIndexAdaptor<
					L2_Simple_Adaptor<N, AtomCloud<T, N> >,
					AtomCloud<T, N>, 3> kd_tree_t;

			kd_tree_t index(3, cloud, KDTreeSingleIndexAdaptorParams(10));
			index.buildIndex();

			nanoflann::SearchParams params;
			params.sorted = false;
#endif

//			// NN search
#pragma omp parallel for
			for (int i = 0; i < atoms.size(); ++i) {
				Atom<T, N> &a = atoms[i];


#ifndef FLANN
				std::vector<uint32_t> neighbours;
				octree.template radiusNeighbors<unibn::L2Distance<tvec3<N>>>(
						a.now * truncation, (float) (h2 * truncation), neighbours);
				a.neighbours->reserve(neighbours.size());
				a.p6ks->reserve(neighbours.size());
				a.skgs->reserve(neighbours.size());
				for (uint32_t &idx : neighbours)
					a.neighbours->emplace_back(&atoms[idx]);


#else

//				std::vector<size_t> n2;
//				rtn.find(a, h2, n2);
//				a.neighbours->reserve(n2.size());
//				a.p6ks->reserve(n2.size());
//				a.skgs->reserve(n2.size());
//				for (const size_t t : n2)
//					a.neighbours->emplace_back(&atoms[t]);

//				octree.template radiusNeighbors<unibn::L2Distance<tvec3<N>>>(
//						a.now * truncation, (float) (h2 * truncation), n2);


//				std::cout << ">>" << a.particle->t << std::endl;
//				std::cout << "GNN[";
//				for (size_t &idx : n2)
//					std::cout << idx << ' ';
//				std::cout << "]" << std::endl;


//				std::cout << "\tNNC: " <<  neighbours.size() << std::endl;


				std::vector<std::pair<size_t, N> > drain;

				auto origin = a.now * truncation;

				index.radiusSearch(&origin[0], h2 * truncation, drain, params);
				a.neighbours->reserve(drain.size());
				a.p6ks->reserve(drain.size());
				a.skgs->reserve(drain.size());
				for (const std::pair<size_t, N> &p : drain)
					a.neighbours->emplace_back(&atoms[p.first]);

//				std::cout << "OCT[";
//				for (const std::pair<size_t, N> &p : drain)
//					std::cout << p.first << ' ';
//				std::cout << "]" << std::endl;

#endif

			}


			hrc::time_point nne = hrc::now();
			auto nn = duration_cast<nanoseconds>(nne - nns).count();

			std::cout << "\tNN: " << (nn / 1000000.0) << "ms" << std::endl;

			hrc::time_point kerns = hrc::now();


			auto rs = clo.run(atoms, iteration, scale);


			for (int j = 0; j <atoms.size(); ++j) {
				Atom<T, N> &a = atoms[j];
				a.particle->velocity = clutil::glmT(rs[j].velocity);
				a.particle->position = clutil::glmT(rs[j].position);
			}

//
//			for (size_t j = 0; j < iteration; ++j) {
//
//				// solve for lambda
//#pragma omp parallel for
//				for (int i = 0; i < atoms.size(); ++i) {
//					Atom<T, N> &a = atoms[i];
//					// Rho : density of a particle
//					N rho = 0.f;
//					auto norm2V = tvec3<N>(0);
//					int nss = 0;
//					for (size_t l = 0; l < a.neighbours->size(); ++l) {
//						Atom<T, N> *b = (*a.neighbours)[l];
//						N r;
//						tvec3<N> skg = spikyKernelGradient(a.now, b->now, r);
//						N p6k = poly6Kernel(r);
//						rho += b->mass * p6k;
//						norm2V += skg * (1.f / RHO);
//						(*a.skgs)[l] = skg;
//						(*a.p6ks)[l] = p6k;
////						std::cout << "[" << a.particle->t << "]["<< b->particle->t << "]NS:" << nss  <<std::endl;
//
//						nss += b->particle->t;
//					}
//					auto norm2 = glm::length2(norm2V);
//					N C = (rho / RHO - 1.f);
//					a.lambda = -C / (norm2 + CFM_EPSILON);
////					std::cout << "["<< a.particle->t << "]NS:" << nss  << " N2:" << rho <<std::endl;
//
//				}
//
//				// solve for delta p
//#pragma omp parallel for
//				for (int i = 0; i < atoms.size(); ++i) {
//					Atom<T, N> &a = atoms[i];
//					a.deltaP = tvec3<N>(0);
//
//					for (size_t l = 0; l < a.neighbours->size(); ++l) {
//						Atom<T, N> *b = (*a.neighbours)[l];
//						N corr = -CorrK *
//						         std::pow((*a.p6ks)[l] / p6DeltaQ, CorrN);
//						N factor = (a.lambda + b->lambda + corr) / RHO;
//						a.deltaP = (*a.skgs)[l] * factor + a.deltaP;
//					}
//
//					auto current = Response<N>((a.now + a.deltaP) * scale, a.velocity);
//					for (const auto &f : colliders) {
//						Ray<N> ray = Ray<N>(a.particle->position,
//						                    current.getPosition(),
//						                    current.getVelocity());
//						current = f(ray);
//					}
//
//					a.now = current.getPosition() / scale;
//					a.velocity = current.getVelocity();
//				}
//			}
//
//
//			// finalise
//			for (Atom<T, N> &a : atoms) {
//
//
//				auto deltaX = a.now - a.particle->position / scale;
//				a.particle->position = a.now * scale;
//				a.particle->mass = a.mass;
//				a.particle->velocity = (deltaX * (1.f / dt) + a.velocity) * VD;
//			}

//			for (Atom<T, N> &a : atoms) {
//				std::cout << "CPU >> "
//				          << " p=" << glm::to_string(a.particle->position)
//				          << " v=" << glm::to_string(a.particle->velocity)
//						<< " lam=" << a.lambda
//						<< " deltaP=" << glm::to_string(a.deltaP)
//				          << std::endl;
//			}

			hrc::time_point kerne = hrc::now();

			auto kern = duration_cast<nanoseconds>(kerne - kerns).count();
			std::cout << "\tKern: " << (kern / 1000000.0) << "ms" << std::endl;


		}


	};



//namespace unibn {
//	namespace traits {
//		template<typename T>
//		struct access<fluid::Atom<T>, 0> {
//			static float get(const fluid::Atom<T> &p) {
//				return static_cast<float>(p.now.x);
//			}
//		};
//
//		template<typename T>
//		struct access<fluid::Atom<T>, 1> {
//			static float get(const fluid::Atom<T> &p) {
//				return static_cast<float>(p.now.y);
//			}
//		};
//
//		template<typename T>
//		struct access<fluid::Atom<T>, 2> {
//			static float get(const fluid::Atom<T> &p) {
//				return static_cast<float>(p.now.z);
//			}
//		};
//	}
//}

}

#endif //LIBFLUID_CPUSPH_HPP
