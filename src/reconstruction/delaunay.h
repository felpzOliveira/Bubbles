/* date = October 6th 2022 19:55 */
#pragma once
#include <gDel3D/GpuDelaunay.h>
#include <util.h>
#include <vector>
#include <host_mesh.h>
#include <bound_util.h>
#include <sph_solver.h>
#include <statics.h>

typedef WorkQueue<vec4ui> DelaunayWorkQueue;

struct DelaunayTriangulation{
    Point3HVec pointVec;
    GDelOutput output;
    size_t pLen;
    std::vector<int> boundary;
    std::vector<vec3i> shrinked;
    std::vector<vec3f> shrinkedPs;
    std::vector<uint32_t> ids;
};

void
DelaunaySurface(DelaunayTriangulation &triangulation, SphParticleSet3 *sphSet,
                Float spacing, Float mu, Grid3 *domain, SphSolver3 *solver,
                TimerList &timer);

void DelaunayClassifyNeighbors(ParticleSet3 *pSet, Grid3 *domain, int threshold,
                               Float spacing, Float mu);

void
DelaunayGetTriangleMesh(DelaunayTriangulation &triangulation, HostTriangleMesh3 *mesh);

void
DelaunayWriteBoundary(DelaunayTriangulation &triangulation, SphParticleSet3 *sphSet,
                      const char *path);

