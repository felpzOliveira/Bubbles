/* date = October 6th 2022 19:55 */
#pragma once
#include <gDel3D/GpuDelaunay.h>
#include <util.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <host_mesh.h>

struct DelaunayIndexInfo{
    vec3ui baseTriangle;
    int oposite;
    int counter;
};

typedef iN<2> i2;
typedef iNHasher<2> i2Hasher;
typedef iNIsSame<2> i2IsSame;

typedef std::unordered_map<i2, int, i2Hasher, i2IsSame> DD_edgeMap;
typedef std::unordered_map<i3, int, i3Hasher, i3IsSame> DelaunayTriangleMap;
typedef std::unordered_map<i3, DelaunayIndexInfo, i3Hasher, i3IsSame> DelaunayTriangleIndexedMap;
typedef std::unordered_map<int, Float> DelaunayFloatTriangleMap;
typedef std::unordered_map<int, int> DelaunayVertexMap;
typedef std::unordered_map<uint32_t, std::unordered_set<uint32_t>> DelaunayVertexConnection;

struct DelaunayTriangulation{
    Point3HVec pointVec;
    GDelOutput output;
    DelaunayFloatTriangleMap partRMap;
    DelaunayVertexMap vertexMap;
    size_t pLen;
    std::vector<int> boundary;
    std::vector<vec3i> shrinked;
    std::vector<uint32_t> ids;

    // debug
};

__host__ void
DelaunaySurface(DelaunayTriangulation &triangulation, SphParticleSet3 *sphSet,
                Float spacing, Float mu, Grid3 *domain);

__host__ void
DelaunayGetTriangleMesh(DelaunayTriangulation &triangulation, HostTriangleMesh3 *mesh);

__host__ void
DelaunayWriteBoundary(DelaunayTriangulation &triangulation, SphParticleSet3 *sphSet,
                      const char *path);

__host__ void
DelaunayWritePly(DelaunayTriangulation &triangulation, const char *path);
