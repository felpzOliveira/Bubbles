/* date = October 6th 2022 19:55 */
#pragma once
#include <gDel3D/GpuDelaunay.h>
#include <util.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

typedef std::unordered_set<i3, i3Hasher, i3IsSame> DelaunayTriangleSet;
typedef std::unordered_map<i3, int, i3Hasher, i3IsSame> DelaunayTriangleMap;
typedef std::unordered_map<iN<4>, int, iNHasher<4>, iNIsSame<4>> DelaunayTetraMap;

struct DelaunayTriangulation{
    Point3HVec pointVec;
    GDelOutput output;
    size_t pLen;
    std::vector<i3> shrinked;
    std::vector<uint32_t> ids;
};

__host__ void
DelaunayTriangulate(DelaunayTriangulation &triangulation, SphParticleSet3 *sphSet,
                  Grid3 *domain);

__host__ void
DelaunayShrink(DelaunayTriangulation &triangulation, SphParticleSet3 *sphSet,
             std::vector<int> &boundary);

__host__ void
DelaunayWritePly(DelaunayTriangulation &triangulation, const char *path);
