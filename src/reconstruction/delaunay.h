/* date = October 6th 2022 19:55 */
#pragma once
#include <gDel3D/GpuDelaunay.h>
#include <util.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <host_mesh.h>
#include <bound_util.h>

struct DelaunayIndexInfo{
    vec3ui baseTriangle;
    int oposite;
    int counter;
};

typedef std::unordered_map<i3, DelaunayIndexInfo, i3Hasher, i3IsSame> DelaunayTriangleIndexedMap;
typedef std::unordered_map<uint32_t, std::unordered_set<uint32_t>> DelaunayVertexConnection;

typedef WorkQueue<vec4ui> DelaunayWorkQueue;

struct DelaunayTriangulation{
    Point3HVec pointVec;
    GDelOutput output;
    size_t pLen;
    std::vector<int> boundary;
    std::vector<vec3i> shrinked;
    std::vector<uint32_t> ids;
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
