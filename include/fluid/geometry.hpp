#ifndef LIBFLUID_GEOMETRY_HPP
#define LIBFLUID_GEOMETRY_HPP

#include <map>

#include "glm/ext.hpp"
#include "glm/glm.hpp"
#include "glm/gtx/normal.hpp"


namespace geometry {


	using glm::tvec3;


	template<typename N>
	struct Triangle {
		tvec3<N> v0, v1, v2;

		Triangle() = default;

		explicit Triangle(
				const tvec3<N> &v0, const tvec3<N> &v1, const tvec3<N> &v2) :
				v0(v0), v1(v1), v2(v2) {}


		friend std::ostream &operator<<(std::ostream &os, const Triangle &triangle) {
			os << "T("
			   << glm::to_string(triangle.v0) << ","
			   << glm::to_string(triangle.v1) << ","
			   << glm::to_string(triangle.v2)
			   << ")";
			return os;
		}

	};


	template<typename N>
	struct MeshTriangle {
		tvec3<N> v0, v1, v2;
		tvec3<N> n0, n1, n2;
		tvec3<uint32_t> c;

		MeshTriangle() = default;

		explicit MeshTriangle(
				const tvec3<N> &v0, const tvec3<N> &v1, const tvec3<N> &v2, const tvec3<N> &normal
		) : v0(v0), v1(v1), v2(v2), n0(normal), n1(n0), n2(n0), c(tvec3<uint32_t>(0xFFFFFFFF)) {}

		explicit MeshTriangle(
				const tvec3<N> &v0, const tvec3<N> &v1, const tvec3<N> &v2) :
				v0(v0), v1(v1), v2(v2),
				n0(glm::triangleNormal(v0, v1, v2)), n1(n0), n2(n0) {}

		explicit MeshTriangle(const tvec3<N> &v0, const tvec3<N> &v1, const tvec3<N> &v2,
		                      const tvec3<N> &n0, const tvec3<N> &n1, const tvec3<N> &n2) :
				v0(v0), v1(v1), v2(v2),
				n0(n0), n1(n1), n2(n2) {}

		friend std::ostream &operator<<(std::ostream &os, const MeshTriangle &triangle) {
			os << "MT(v=<"
			   << glm::to_string(triangle.v0) << ","
			   << glm::to_string(triangle.v1) << ","
			   << glm::to_string(triangle.v2) << ">"
			   << " n=<"
			   << glm::to_string(triangle.n0) << ", "
			   << glm::to_string(triangle.n1) << ", "
			   << glm::to_string(triangle.n2) << ">"
			   << " c=<0x"
			   << std::hex << triangle.c.x << ", 0x"
			   << std::hex << triangle.c.y << ", 0x"
			   << std::hex << triangle.c.z << ">"
			   << ")";
			return os;
		}

	};


	//https://github.com/opengl-tutorials/ogl/blob/master/common/vboindexer.cpp
	template<typename N>
	struct PackedVertex {
		tvec3<N> position;

		bool operator<(const PackedVertex that) const {
			return memcmp((void *) this, (void *) &that, sizeof(PackedVertex)) > 0;
		};
	};

	template<typename N>
	bool getSimilarVertexIndex_fast(
			PackedVertex<N> &packed,
			std::map<PackedVertex<N>, unsigned short> &VertexToOutIndex,
			unsigned short &result
	) {
		typename std::map<PackedVertex<N>, unsigned short>::iterator it = VertexToOutIndex.find(
				packed);
		if (it == VertexToOutIndex.end()) {
			return false;
		} else {
			result = it->second;
			return true;
		}
	}

	template<typename N>
	void indexVBO(
			std::vector<tvec3<N>> &in_vertices,
			std::vector<unsigned short> &out_indices,
			std::vector<tvec3<N>> &out_vertices
	) {
		std::map<PackedVertex<N>, unsigned short> VertexToOutIndex;
		// For each input vertex
		for (size_t i = 0; i < in_vertices.size(); i++) {
			PackedVertex<N> packed = {in_vertices[i]};
			// Try to find a similar vertex in out_XXXX
			unsigned short index;
			bool found = getSimilarVertexIndex_fast(packed, VertexToOutIndex, index);
			if (found) { // A similar vertex is already in the VBO, use it instead !
				out_indices.push_back(index);
			} else { // If not, it needs to be added in the output data.
				out_vertices.push_back(in_vertices[i]);
				unsigned short newindex = (unsigned short) out_vertices.size() - 1;
				out_indices.push_back(newindex);
				VertexToOutIndex[packed] = newindex;
			}
		}
	}


	template<typename N>
	void indexVBOOne(tvec3<N> point,
	                 std::vector<unsigned short> &out_indices,
	                 std::vector<tvec3<N>> &out_vertices,
	                 std::map<PackedVertex<N>, unsigned short> &VertexToOutIndex
	) {


		PackedVertex<N> packed = {point};
		// Try to find a similar vertex in out_XXXX
		unsigned short index;
		bool found = getSimilarVertexIndex_fast(packed, VertexToOutIndex, index);
		if (found) { // A similar vertex is already in the VBO, use it instead !
			out_indices.push_back(index);
		} else { // If not, it needs to be added in the output data.
			out_vertices.push_back(point);
			unsigned short newindex = (unsigned short) out_vertices.size() - 1;
			out_indices.push_back(newindex);
			VertexToOutIndex[packed] = newindex;
		}
	}


	template<typename N>
	void indexVBO2(
			const std::vector<MeshTriangle<N>> &in_vertices,
			std::vector<unsigned short> &out_indices,
			std::vector<tvec3<N>> &out_vertices
	) {

		std::map<PackedVertex<N>, unsigned short> VertexToOutIndex;
		for (size_t i = 0; i < in_vertices.size(); i++) {
			indexVBOOne(in_vertices[i].v0, out_indices, out_vertices, VertexToOutIndex);
			indexVBOOne(in_vertices[i].v1, out_indices, out_vertices, VertexToOutIndex);
			indexVBOOne(in_vertices[i].v2, out_indices, out_vertices, VertexToOutIndex);
		}
	}

}


#endif //LIBFLUID_GEOMETRY_HPP
