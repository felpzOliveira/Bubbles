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
* Use to output 2 files with particles list for interior particles and particles
* that were generated from whatever method gave us the boundary.
*/
#define DELAUNAY_OUTPUT_PARTICLES 0

typedef enum{
    LaplacianSmooth=0,
    TaubinSmooth
}MeshSmoothMethod;

typedef enum{
    GatherSurface=0, // returns the fluid surface, most likely what you want
    GatherConvexHull, // returns the convex hull of the fluid
    GatherEverything, // returns all triangles from the delaunay triangulation (SLOW)
    GatherNone
}DelaunayOutputType;

/* Options for mesh smoothing, we only do simple laplacian and taubin */
struct MeshSmoothOpts{
    MeshSmoothMethod method;
    int iterations;
    Float lambda;
    Float mu;
};

struct DelaunayOptions{
    bool extendBoundary;
    bool withInteriorParticles;
    bool use_1nn_radius;
    bool use_alpha_shapes;
    DelaunayOutputType outType;
    Float mu;
    Float alpha;
    Float spacing;
};

struct DelaunayTriangulation{
    Point3HVec pointVec;
    GDelOutput output;
    size_t pLen;
    std::vector<vec3i> shrinked;
    std::vector<vec3f> shrinkedPs;
    std::vector<uint32_t> ids;
};

/*
* Gets a set of default options for delaunay surface generation.
*/
DelaunayOptions DelaunayDefaultOptions();

/*
* Computes the surface using delaunay triangulation.
*/
void
DelaunaySurface(DelaunayTriangulation &triangulation, SphParticleSet3 *sphSet,
                Grid3 *domain, DelaunayOptions opts, TimerList &timer);

/*
* Smooths the results of the delaunay triangulation.
*/
void DelaunaySmooth(DelaunayTriangulation &triangulation, MeshSmoothOpts opts);

/*
* Fetchs the underlying mesh resulting from the triangulation.
*/
void
DelaunayGetTriangleMesh(DelaunayTriangulation &triangulation, HostTriangleMesh3 *mesh);


/* Some type manipulation */

inline DelaunayOutputType DelaunayOutputTypeFromString(std::string val){
    if(val == "ConvexHull" || val == "chull" || val == "convexhull")
        return GatherConvexHull;
    if(val == "surface" || val == "Surface")
        return GatherSurface;
    if(val == "all" || val == "All" || val == "everything" || val == "Everything")
        return GatherEverything;
    return GatherNone;
}

inline bool DelaunayIsOutputTypeValid(int val){
    return (DelaunayOutputType)val == GatherSurface ||
           (DelaunayOutputType)val == GatherConvexHull ||
           (DelaunayOutputType)val == GatherEverything;
}

inline const char *DelaunayOutputTypeString(DelaunayOutputType type){
    switch(type){
        case GatherSurface: return "Surface";
        case GatherConvexHull: return "ConvexHull";
        case GatherEverything: return "All";
        default: return "(None)";
    }
}
