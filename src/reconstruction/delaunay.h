/* date = October 6th 2022 19:55 */
#pragma once
#include <gDel3D/GpuDelaunay.h>
#include <util.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct DelaunayIndexInfo{
    vec3ui baseTriangle;
    int oposite;
    int counter;
};

typedef std::unordered_map<i3, int, i3Hasher, i3IsSame> DelaunayTriangleMap;
typedef std::unordered_map<i3, DelaunayIndexInfo, i3Hasher, i3IsSame> DelaunayTriangleIndexedMap;
typedef std::unordered_map<int, Float> DelaunayFloatTriangleMap;
typedef std::unordered_map<int, int> DelaunayVertexMap;

struct DelaunayTriangulation{
    Point3HVec pointVec;
    GDelOutput output;
    DelaunayFloatTriangleMap partRMap;
    DelaunayVertexMap vertexMap;
    size_t pLen;
    std::vector<vec3i> shrinked;
    std::vector<uint32_t> ids;
};

__host__ void
DelaunaySurface(DelaunayTriangulation &triangulation, SphParticleSet3 *sphSet,
                Grid3 *domain);

__host__ void
DelaunayWritePly(DelaunayTriangulation &triangulation, const char *path);
