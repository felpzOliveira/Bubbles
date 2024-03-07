/* date = October 6th 2022 19:55 */
#pragma once
#include <gDel3D/GpuDelaunay.h>
#include <util.h>
#include <vector>
#include <host_mesh.h>
#include <bound_util.h>
#include <sph_solver.h>
#include <statics.h>

/*
* NOTE: We have to include the interior particles otherwise we could generate
* incorrect geometry in the interior. While this geometry would not break the surface
* and it makes the algorithm runs faster it will generate a larger error on the
* Hausdorff computation. It also affects the shape of the mesh with regards to the
* weizenbock quality. So it is kinda of a trade-off, do you want faster computation
* with smaller memory footprint even if there are some incorrect geometry in the
* interior of the mesh and possibly some lower quality triangles?
*/
#define DELAUNAY_WITH_INTERIOR

struct DelaunayTriangleInfo{
    vec3ui tri;
    int opp;

    bb_cpu_gpu
    DelaunayTriangleInfo(int){}

    bb_cpu_gpu
    DelaunayTriangleInfo(vec3ui _tri, int op){
        tri = _tri;
        opp = op;
    }
};

typedef WorkQueue<DelaunayTriangleInfo> DelaunayWorkQueue;

struct DelaunayTriangulation{
    Point3HVec pointVec;
    GDelOutput output;
    size_t pLen;
    std::vector<vec3i> shrinked;
    std::vector<vec3f> shrinkedPs;
    std::vector<uint32_t> ids;
};

void
DelaunaySurface(DelaunayTriangulation &triangulation, SphParticleSet3 *sphSet,
                Float spacing, Float mu, Grid3 *domain, SphSolver3 *solver,
                TimerList &timer);

void DelaunaySmooth(DelaunayTriangulation &triangulation, int iterations);

void DelaunayClassifyNeighbors(ParticleSet3 *pSet, Grid3 *domain, int threshold,
                               Float spacing, Float mu);

void
DelaunayGetTriangleMesh(DelaunayTriangulation &triangulation, HostTriangleMesh3 *mesh);

void
DelaunayWriteBoundary(DelaunayTriangulation &triangulation, SphParticleSet3 *sphSet,
                      const char *path);

