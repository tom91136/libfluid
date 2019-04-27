#include "cl_types.h"
#include <type_traits>

#ifndef LIBFLUID_CLUTILS_H
#define LIBFLUID_CLUTILS_HPP


namespace clutil {


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


	bool statDir(const std::string &path) {
		struct stat info{};
		if (stat(path.c_str(), &info) != 0) return false;
		return static_cast<bool>(info.st_mode & S_IFDIR);
	}


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


		if (!statDir(include))
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

	enum BufferType {
		RW = CL_MEM_READ_WRITE,
		RO = CL_MEM_READ_ONLY,
		WO = CL_MEM_WRITE_ONLY,
	};


	template<typename T, BufferType B>
	struct TypedBuffer {

		cl::Buffer actual;
		const size_t length;

	private:
		TypedBuffer(const cl::Context &context, T &t) : actual(
				cl::Buffer(context, B | CL_MEM_COPY_HOST_PTR, sizeof(T), &t)), length(1) {}

	public:
		TypedBuffer(const cl::Context &context, size_t count) : actual(
				cl::Buffer(context, B, sizeof(T) * count)), length(count) {}


		template<typename Iterable>
		TypedBuffer(
				const cl::CommandQueue &queue,
				Iterable &xs,
				bool useHostPtr = false):
				actual(cl::Buffer(queue, xs.begin(), xs.end(), B == BufferType::RO, useHostPtr)),
				length(xs.size()) {
			static_assert(B != BufferType::WO, "BufferType must be RW or RO");
		}

		static TypedBuffer<T, B> ofStruct(const cl::Context &context, T &t) {
			return TypedBuffer<T, B>(context, t);
		}


//		template<typename IteratorType>
//		void drainTo(const cl::CommandQueue &queue,
//		             IteratorType startIterator, IteratorType endIterator) {
//			cl::copy(queue, actual, startIterator, endIterator);
//		}

//		template<typename Iterable, typename = typename std::enable_if<
//				std::is_same<
//						typename std::iterator_traits<Iterable>::value_type,
//						T>::value
//		>>
//		void drainTo(const cl::CommandQueue &queue, Iterable xs) {
//			cl::copy(queue, actual, xs.begin(), xs.end());
//		}

		template<typename Iterable>
		inline void drainTo(const cl::CommandQueue &queue, Iterable &xs) {
			cl::copy(queue, actual, xs.begin(), xs.end());
		}


	};


}

//namespace cl {
//	namespace detail {
//
//		template<typename T>
//		struct ::cl::detail::KernelArgumentHandler<T,
//				typename std::enable_if<std::is_base_of<clutil::TypedBuffer<T, clutil::BufferType::RO>, T>::value>::type> {
//			static size_type size(const T &) { return sizeof(T); }
//			static const T *ptr(const T &value) { return &value; }
//		};
//	}
//}

#endif //LIBFLUID_CLUTILS_H
