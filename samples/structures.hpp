#pragma once

#include <chrono>
#include <ostream>

#include "fluid/fluid.hpp"
#include "fluid/surface.hpp"
#include "fluid/mmf.hpp"


namespace miommf {
	mio::mmap_source createSource(const std::string &path) {

		std::error_code error;

		mio::mmap_source source = mio::make_mmap_source(path, error);
		if (error) {
			std::cout << "MMF(" << path << ") failed:" << error.message() << std::endl;
			exit(1);
		} else {
			std::cout << "MMF(" << path << ") size: "
			          << (double) source.size() / (1024 * 1024) << "MB" << std::endl;
		}
		return source;
	}

	mio::mmap_sink createSink(const std::string &path, const size_t length) {
		std::ofstream file(path, std::ios::out | std::ios::trunc);
		std::string s(std::max<size_t>(1024 * 4, length), ' '); // XXX may fail if than 4k
		file << s;

		std::error_code error;
		mio::mmap_sink sink = mio::make_mmap_sink(path, 0, mio::map_entire_file, error);
		if (error) {
			std::cout << "MMF(" << path << ") failed:" << error.message() << std::endl;
			exit(1);
		} else {
			std::cout << "MMF(" << path << ") size: "
			          << (double) sink.size() / (1024 * 1024) << "MB" << std::endl;
		}
		return sink;
	}

	inline bool canRead(mio::mmap_source &source) {
		return source.is_open() && source.is_mapped();
	}

}

namespace strucures {


	static const char id_[] = "id";
	static const char type_[] = "type";
	static const char mass_[] = "mass";
	static const char colour_[] = "colour";
	static const char position_x_[] = "position.x";
	static const char position_y_[] = "position.y";
	static const char position_z_[] = "position.z";
	static const char velocity_x_[] = "velocity.x";
	static const char velocity_y_[] = "velocity.y";
	static const char velocity_z_[] = "velocity.z";

	template<typename T, typename N>
	static inline auto particleDef() {
		return mmf::makeDef(
				DECL_MEMBER(id_, CLS(fluid::Particle<T, N>), id),
				DECL_MEMBER(type_, CLS(fluid::Particle<T, N>), type),
				DECL_MEMBER(mass_, CLS(fluid::Particle<T, N>), mass),
				DECL_MEMBER(colour_, CLS(fluid::Particle<T, N>), colour),
				DECL_MEMBER(position_x_, CLS(fluid::Particle<T, N>), position.x),
				DECL_MEMBER(position_y_, CLS(fluid::Particle<T, N>), position.y),
				DECL_MEMBER(position_z_, CLS(fluid::Particle<T, N>), position.z),
				DECL_MEMBER(velocity_x_, CLS(fluid::Particle<T, N>), velocity.x),
				DECL_MEMBER(velocity_y_, CLS(fluid::Particle<T, N>), velocity.y),
				DECL_MEMBER(velocity_z_, CLS(fluid::Particle<T, N>), velocity.z)
		);
	}

	static const char well_x_[] = "well.x";
	static const char well_y_[] = "well.y";
	static const char well_z_[] = "well.z";
	static const char well_force_[] = "force";


	template<typename N>
	static inline auto wellDef() {
		return mmf::makeDef(
				DECL_MEMBER(well_x_, CLS(fluid::Well<N>), centre.x),
				DECL_MEMBER(well_y_, CLS(fluid::Well<N>), centre.y),
				DECL_MEMBER(well_z_, CLS(fluid::Well<N>), centre.z),
				DECL_MEMBER(well_force_, CLS(fluid::Well<N>), force)
		);
	}


	static const char source_x_[] = "source.x";
	static const char source_y_[] = "source.y";
	static const char source_z_[] = "source.z";
	static const char source_vel_x_[] = "source.vel.x";
	static const char source_vel_y_[] = "source.vel.y";
	static const char source_vel_z_[] = "source.vel.z";
	static const char source_rate_[] = "rate";
	static const char source_tag_[] = "tag";
	static const char source_colour_[] = "colour";

	template<typename N>
	static inline auto sourceDef() {
		return mmf::makeDef(
				DECL_MEMBER(source_x_, CLS(fluid::Source<N>), centre.x),
				DECL_MEMBER(source_y_, CLS(fluid::Source<N>), centre.y),
				DECL_MEMBER(source_z_, CLS(fluid::Source<N>), centre.z),
				DECL_MEMBER(source_vel_x_, CLS(fluid::Source<N>), velocity.x),
				DECL_MEMBER(source_vel_y_, CLS(fluid::Source<N>), velocity.y),
				DECL_MEMBER(source_vel_z_, CLS(fluid::Source<N>), velocity.z),
				DECL_MEMBER(source_rate_, CLS(fluid::Source<N>), rate),
				DECL_MEMBER(source_tag_, CLS(fluid::Source<N>), tag),
				DECL_MEMBER(source_colour_, CLS(fluid::Source<N>), colour)
		);
	}


	static const char drain_x_[] = "drain.x";
	static const char drain_y_[] = "drain.y";
	static const char drain_z_[] = "drain.z";
	static const char drain_width_[] = "width";
	static const char drain_depth_[] = "depth";

	template<typename N>
	static inline auto drainDef() {
		return mmf::makeDef(
				DECL_MEMBER(drain_x_, CLS(fluid::Drain<N>), centre.x),
				DECL_MEMBER(drain_y_, CLS(fluid::Drain<N>), centre.y),
				DECL_MEMBER(drain_z_, CLS(fluid::Drain<N>), centre.z),
				DECL_MEMBER(drain_width_, CLS(fluid::Drain<N>), width),
				DECL_MEMBER(drain_depth_, CLS(fluid::Drain<N>), depth)
		);
	}

	static const char v0_x_[] = "v0.x";
	static const char v0_y_[] = "v0.y";
	static const char v0_z_[] = "v0.z";
	static const char v1_x_[] = "v1.x";
	static const char v1_y_[] = "v1.y";
	static const char v1_z_[] = "v1.z";
	static const char v2_x_[] = "v2.x";
	static const char v2_y_[] = "v2.y";
	static const char v2_z_[] = "v2.z";

	static const char n0_x_[] = "n0.x";
	static const char n0_y_[] = "n0.y";
	static const char n0_z_[] = "n0.z";
	static const char n1_x_[] = "n1.x";
	static const char n1_y_[] = "n1.y";
	static const char n1_z_[] = "n1.z";
	static const char n2_x_[] = "n2.x";
	static const char n2_y_[] = "n2.y";
	static const char n2_z_[] = "n2.z";

	static const char c_x_[] = "c.x";
	static const char c_y_[] = "c.y";
	static const char c_z_[] = "c.z";


	template<typename N>
	static inline auto triangleDef() {
		return mmf::makeDef(
				DECL_MEMBER(v0_x_, CLS(geometry::Triangle<N>), v0.x),
				DECL_MEMBER(v0_y_, CLS(geometry::Triangle<N>), v0.y),
				DECL_MEMBER(v0_z_, CLS(geometry::Triangle<N>), v0.z),
				DECL_MEMBER(v1_x_, CLS(geometry::Triangle<N>), v1.x),
				DECL_MEMBER(v1_y_, CLS(geometry::Triangle<N>), v1.y),
				DECL_MEMBER(v1_z_, CLS(geometry::Triangle<N>), v1.z),
				DECL_MEMBER(v2_x_, CLS(geometry::Triangle<N>), v2.x),
				DECL_MEMBER(v2_y_, CLS(geometry::Triangle<N>), v2.y),
				DECL_MEMBER(v2_z_, CLS(geometry::Triangle<N>), v2.z)
		);
	}

	static const char vec3_x_[] = "vec3.x";
	static const char vec3_y_[] = "vec3.y";
	static const char vec3_z_[] = "vec3.z";

	template<typename N>
	static inline auto vec3Def() {
		return mmf::makeDef(
				DECL_MEMBER(vec3_x_, CLS(glm::tvec3<N>), x),
				DECL_MEMBER(vec3_y_, CLS(glm::tvec3<N>), y),
				DECL_MEMBER(vec3_z_, CLS(glm::tvec3<N>), z)
		);
	}


	template<typename N>
	static inline auto meshTriangleDef() {
		return mmf::makeDef(
				DECL_MEMBER(v0_x_, CLS(geometry::MeshTriangle<N>), v0.x),
				DECL_MEMBER(v0_y_, CLS(geometry::MeshTriangle<N>), v0.y),
				DECL_MEMBER(v0_z_, CLS(geometry::MeshTriangle<N>), v0.z),
				DECL_MEMBER(v1_x_, CLS(geometry::MeshTriangle<N>), v1.x),
				DECL_MEMBER(v1_y_, CLS(geometry::MeshTriangle<N>), v1.y),
				DECL_MEMBER(v1_z_, CLS(geometry::MeshTriangle<N>), v1.z),
				DECL_MEMBER(v2_x_, CLS(geometry::MeshTriangle<N>), v2.x),
				DECL_MEMBER(v2_y_, CLS(geometry::MeshTriangle<N>), v2.y),
				DECL_MEMBER(v2_z_, CLS(geometry::MeshTriangle<N>), v2.z),
				DECL_MEMBER(n0_x_, CLS(geometry::MeshTriangle<N>), n0.x),
				DECL_MEMBER(n0_y_, CLS(geometry::MeshTriangle<N>), n0.y),
				DECL_MEMBER(n0_z_, CLS(geometry::MeshTriangle<N>), n0.z),
				DECL_MEMBER(n1_x_, CLS(geometry::MeshTriangle<N>), n1.x),
				DECL_MEMBER(n1_y_, CLS(geometry::MeshTriangle<N>), n1.y),
				DECL_MEMBER(n1_z_, CLS(geometry::MeshTriangle<N>), n1.z),
				DECL_MEMBER(n2_x_, CLS(geometry::MeshTriangle<N>), n2.x),
				DECL_MEMBER(n2_y_, CLS(geometry::MeshTriangle<N>), n2.y),
				DECL_MEMBER(n2_z_, CLS(geometry::MeshTriangle<N>), n2.z),
				DECL_MEMBER(c_x_, CLS(geometry::MeshTriangle<N>), c.x),
				DECL_MEMBER(c_y_, CLS(geometry::MeshTriangle<N>), c.y),
				DECL_MEMBER(c_z_, CLS(geometry::MeshTriangle<N>), c.z)
		);
	}


	static const char timestamp_[] = "timestamp";
	static const char entries_[] = "entries";
	static const char written_[] = "written";


	struct Header {
		long timestamp;
		size_t entries;
		size_t written;


		Header() = default;
		explicit Header(size_t entries) : timestamp(
				std::chrono::duration_cast<std::chrono::milliseconds>(
						std::chrono::system_clock::now().time_since_epoch()).count()
		), entries(entries), written(0) {}
		friend std::ostream &operator<<(std::ostream &os, const Header &header) {
			return os
					<< "Header(timestamp=" << header.timestamp << ", entries=" << header.entries
					<< ")";
		}
	};

	static inline auto headerDef() {
		return mmf::makeDef(
				DECL_MEMBER(timestamp_, CLS(Header), timestamp),
				DECL_MEMBER(entries_, CLS(Header), entries),
				DECL_MEMBER(written_, CLS(Header), written)
		);
	}

	static const char suspend_[] = "suspend";
	static const char terminate_[] = "terminate";
	static const char solverIter_[] = "solverIter";
	static const char solverStep_[] = "solverStep";
	static const char solverScale_[] = "solverScale";
	static const char surfaceRes_[] = "surfaceRes";
	static const char gravity_[] = "gravity";
	static const char minBound_x_[] = "minBound.x";
	static const char minBound_y_[] = "minBound.y";
	static const char minBound_z_[] = "minBound.z";
	static const char maxBound_x_[] = "maxBound.x";
	static const char maxBound_y_[] = "maxBound.y";
	static const char maxBound_z_[] = "maxBound.z";

	using glm::tvec3;

	template<typename N>
	struct SceneMeta {
		bool suspend;
		bool terminate;
		int solverIter;
		N solverStep;
		N solverScale;
		N surfaceRes;
		N gravity;
		tvec3<N> minBound, maxBound;

		SceneMeta() = default;
		explicit SceneMeta(bool suspend, bool terminate,
		                   int solverIter, N solverStep, N solverScale, N surfaceRes, N gravity,
		                   tvec3<N> minBound, tvec3<N> maxBound) :
				suspend(suspend), terminate(terminate),
				solverIter(solverIter), solverStep(solverStep),
				solverScale(solverScale), surfaceRes(surfaceRes),
				gravity(gravity),
				minBound(minBound), maxBound(maxBound) {}

		friend std::ostream &operator<<(std::ostream &os, const SceneMeta &scene) {
			os << std::boolalpha << "SceneMeta{"
			   << "\n suspend:    " << scene.suspend
			   << "\n terminate:  " << scene.terminate
			   << "\n solverIter: " << scene.solverIter
			   << "\n solverStep: " << scene.solverStep
			   << "\n solverScale:" << scene.solverScale
			   << "\n surfaceRes: " << scene.surfaceRes
			   << "\n gravity:    " << scene.gravity
			   << "\n bounds:     "
			   << " min=" << glm::to_string(scene.minBound)
			   << " max=" << glm::to_string(scene.maxBound)
			   << "\n}";
			return os;
		}
	};

	template<typename N>
	struct Scene {
		SceneMeta<N> meta;
		std::vector<fluid::Well<N>> wells;
		std::vector<fluid::Source<N>> sources;
		std::vector<fluid::Drain<N>> drains;
		explicit Scene(const SceneMeta<N> &meta,
		               const std::vector<fluid::Well<N>> &wells,
		               const std::vector<fluid::Source<N>> &sources,
		               const std::vector<fluid::Drain<N>> &drains) :
				meta(meta), wells(wells), sources(sources), drains(drains) {}
		friend std::ostream &operator<<(std::ostream &os, const Scene &scene) {
			return os << "Scene{"
			          << "\n" << scene.meta
			          << "\n wells  : " << scene.wells.size()
			          << "\n sources: " << scene.sources.size()
			          << "\n drains: " << scene.drains.size()
			          << "\n}";
		}
	};

	template<typename N>
	static inline auto sceneMetaDef() {
		return mmf::makeDef(
				DECL_MEMBER(suspend_, CLS(SceneMeta<N>), suspend),
				DECL_MEMBER(terminate_, CLS(SceneMeta<N>), terminate),
				DECL_MEMBER(solverIter_, CLS(SceneMeta<N>), solverIter),
				DECL_MEMBER(solverStep_, CLS(SceneMeta<N>), solverStep),
				DECL_MEMBER(solverScale_, CLS(SceneMeta<N>), solverScale),
				DECL_MEMBER(surfaceRes_, CLS(SceneMeta<N>), surfaceRes),
				DECL_MEMBER(gravity_, CLS(SceneMeta<N>), gravity),
				DECL_MEMBER(minBound_x_, CLS(SceneMeta<N>), minBound.x),
				DECL_MEMBER(minBound_y_, CLS(SceneMeta<N>), minBound.y),
				DECL_MEMBER(minBound_z_, CLS(SceneMeta<N>), minBound.z),
				DECL_MEMBER(maxBound_x_, CLS(SceneMeta<N>), maxBound.x),
				DECL_MEMBER(maxBound_y_, CLS(SceneMeta<N>), maxBound.y),
				DECL_MEMBER(maxBound_z_, CLS(SceneMeta<N>), maxBound.z)
		);
	}

	template<typename T, typename N>
	void writeParticles(mio::mmap_sink &sink, const std::vector<fluid::Particle<T, N>> &xs) {

		Header header = Header(xs.size());
		size_t offset = mmf::writer::writePacked(sink, header, 0, headerDef());
		for (const fluid::Particle<T, N> &p :  xs) {
			offset = mmf::writer::writePacked(sink, p, offset, particleDef<T, N>());
		}
		header.written = xs.size();
		mmf::writer::writePacked(sink, header, 0, headerDef());
	}

	template<typename N>
	void writeTriangles(mio::mmap_sink &sink, const std::vector<surface::MeshTriangle<N>> &xs) {
		Header header = Header(xs.size());
		size_t offset = mmf::writer::writePacked(sink, header, 0, headerDef());
		for (const surface::MeshTriangle<N> &t :  xs) {
			offset = mmf::writer::writePacked(sink, t, offset, meshTriangleDef<N>());
		}
		header.written = xs.size();
		mmf::writer::writePacked(sink, header, 0, headerDef());
	}


	template<typename N>
	void readSceneMeta(mio::mmap_source &source, SceneMeta<N> &scene) {
		mmf::reader::readPacked(source, scene, 0, sceneMetaDef<N>());
	}


	template<typename N>
	Scene<N> readScene(mio::mmap_source &source) {
		SceneMeta<N> meta{};
		size_t offset = mmf::reader::readPacked(source, meta, 0, sceneMetaDef<N>());

		Header header{};
		offset = mmf::reader::readPacked(source, header, offset, headerDef());

		std::vector<fluid::Well<N>> wells(header.entries);
		for (size_t i = 0; i < header.entries; ++i) {
			offset = mmf::reader::readPacked(source, wells[i], offset, wellDef<N>());
		}

		offset = mmf::reader::readPacked(source, header, offset, headerDef());
		std::vector<fluid::Source<N>> sources(header.entries);
		for (size_t i = 0; i < header.entries; ++i) {
			offset = mmf::reader::readPacked(source, sources[i], offset, sourceDef<N>());
		}

		offset = mmf::reader::readPacked(source, header, offset, headerDef());
		std::vector<fluid::Drain<N>> drains(header.entries);
		for (size_t i = 0; i < header.entries; ++i) {
			offset = mmf::reader::readPacked(source, drains[i], offset, drainDef<N>());
		}

		return Scene<N>(meta, wells, sources, drains);
	}


	std::pair<Header, size_t> readHeader(mio::mmap_source &source) {
		Header header{};
		size_t offset = mmf::reader::readPacked(source, header, 0, headerDef());
		return std::make_pair(header, offset);
	}

	template<typename N>
	fluid::RigidBody<N> readRigidBody(mio::mmap_source &source, std::pair<Header, size_t> partial) {
		std::vector<tvec3<N>> ts(partial.first.entries);
		size_t offset = partial.second;
		for (size_t i = 0; i < partial.first.entries; ++i) {
			offset = mmf::reader::readPacked(source, ts[i], offset, vec3Def<N>());
		}
		return fluid::RigidBody<N>(ts);
	}


	template<typename N>
	fluid::MeshCollider<N> readCollider(mio::mmap_source &source) {
		Header header{};
		size_t offset = mmf::reader::readPacked(source, header, 0, headerDef());
		std::cout << header << "\n";
		std::vector<geometry::Triangle<N>> ts(header.entries);
		for (size_t i = 0; i < header.entries; ++i) {
			offset = mmf::reader::readPacked(source, ts[i], offset, triangleDef<N>());
		}
		return fluid::MeshCollider<N>(ts);
	}


}
