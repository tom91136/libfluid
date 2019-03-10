#ifndef LIBFLUID_WRITER_H
#define LIBFLUID_WRITER_H

#include <vector>
#include <nlohmann/json.hpp>


namespace writer {

#define CLS(...) __VA_ARGS__
#define ENTRY(name, CLS, F)  writer::Entry<name, decltype(CLS::F), CLS>( [](const CLS &x) { return x.F;} )
#define ENTRY_T(name, CLS, F)  writer::Entry<a, decltype(CLS::F), CLS>

	template<const char *Name, class Type, class Struct>
	class Entry {

	private:
		Type (*f)(const Struct &);

	public:
		static constexpr auto name = Name;
		static constexpr auto size = sizeof(Type);

		inline explicit Entry(Type (*f)(const Struct &)) : f(f) {}

		template<typename W>
		inline size_t write(W &sink, const Struct &s, const size_t offset) {
			union {
				Type d;
				unsigned char bytes[size];
			} u{f(s)};
			for (size_t i = 0; i < size; i++) sink[offset + i] = u.bytes[i];
			return offset + size;
		}

	};

	template<class ... Args>
	struct pack {
	};

	template<class... Args>
	struct Entries {
		using entries = pack<Args...>;
		const std::tuple<Args...> args;

		inline explicit Entries(Args ...xs) : args(std::tuple<Args...>(xs...)) {}
	};

	template<class... Args>
	inline static const Entries<Args...> makeEntries(Args &&...xs) {
		return Entries<Args...>(xs...);
	}


	using nlohmann::json;

	struct Row {
		const char *name;
		const size_t size;

		Row(const char *name, size_t size) : name(name), size(size) {}
	};

	void to_json(json &j, const Row &p) {
		j = json{{"name", p.name},
		         {"size", p.size}};
	}

	template<typename T>
	inline static size_t writeMeta(std::vector<Row> &xs) {
		xs.push_back(Row(T::name, T::size));
		return T::size;
	}

	template<typename T, class U, class ...Rows>
	inline static size_t writeMeta(std::vector<Row> &xs) {
		return writeMeta < T > (xs) + writeMeta < U, Rows...>(xs);
	}

	template<typename ...Entry>
	inline static std::pair<size_t, json> writeMeta() {
		json j;
		std::vector<Row> xs;
		size_t total = writeMeta < Entry...>(xs);
		j["fields"] = xs;
		return std::pair<size_t, json>(total, j);
	}

	template<typename... Args>
	inline static std::pair<size_t, json> writeMetaPacked_impl(pack<Args...>) {
		return writeMeta<Args...>();
	}

	template<typename E>
	inline static std::pair<size_t, json> writeMetaPacked() {
		return writeMetaPacked_impl(typename E::entries());
	}


	template<typename W, class S, class T>
	inline static size_t write(W &w, const S &s, const size_t offset, T &t) {
		return t.write(w, s, offset);
	}

	template<typename W, class S, class T, class ...Rest>
	inline static size_t write(W &w, const S &s, const size_t offset, T t, Rest...es) {
		size_t next = write(w, s, offset, t);
		return write(w, s, next, es...);
	}

	template<typename W, class S, std::size_t... Is, class ...Rest>
	inline const static size_t writePacked_impl(W &w, const S &s, const size_t offset,
	                                            const std::tuple<Rest...> &xs,
	                                            const std::index_sequence<Is...>) {
		return write(w, s, offset, std::get<Is>(xs)...);
	}

	template<typename W, class S, class ...Rest>
	inline const static size_t
	writePacked(W &w, const S &s, const size_t offset, const Entries<Rest...> &e) {
		return writePacked_impl(w, s, offset, e.args, std::index_sequence_for<Rest...>());
	}


};


#endif //LIBFLUID_WRITER_H
