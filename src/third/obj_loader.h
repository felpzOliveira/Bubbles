#pragma once
#include <cutil.h>
#include <vector>
#include <string.h>
#include <fstream>

/*
* Defines data types to be used for components:
* needs vec3f/Point2f/Point3f/Point3i/Normal3f and
* the definition of ParsedMesh which should be:
* typedef struct{
    *     Point3f *p;
    *     Normal3f *n;
    *     vec3f *s;
    *     Point2f *uv;
    *     Point3i *indices;
    *     int nTriangles, nVertices;
    *     int nUvs, nNormals;
    *     <Optional -- WITH_TRANSFORM>
    *     Transform toWorld;
* }ParsedMesh;
* tangents (vec3f *s) are not actually parsed/supported.
*/
#include <geometry.h>

/*
* Prevent Bubbles from having to include mtl.h/transform.h/shape.h,
* this includes are usefull for ToyTracer where shapes get shifted.
*/
//#define WITH_MTL
//#define WITH_TRANSFORM
//#define WITH_SHAPE

#if defined(WITH_TRANSFORM)
#include <transform.h>
#endif

#if defined(WITH_SHAPE)
#include <shape.h>
#endif

#if defined(WITH_MTL)
#include <mtl.h>
#else
struct MeshMtl{
    std::string file;
    std::string name;
};
#endif

/* utility macros */
#define IS_SPACE(x) (((x) == ' ') || ((x) == '\t'))
#define IS_NEW_LINE(x) (((x) == '\r') || ((x) == '\n') || ((x) == '\0'))
#define IS_DIGIT(x) (static_cast<unsigned int>((x) - '0') < static_cast<unsigned int>(10))

struct MeshProperties{
    int flip_x;
};

typedef enum{
    GPU=0, CPU
}AllocatorType;

typedef void*(*MemoryAllocator)(long size);
typedef void(*MemoryFree)(void *);

/*
* NOTE: Heavily based on tiny_obj_loader. I'm basically splitting the mesh
* into multiple objects, I don't like the way tiny_obj_loader does it,
* makes it very hard to fragment the mesh. I need a API that can actually
* return a series of meshes for a single obj file if needed. The memory 
* returned is GPU ready for the interior pointers of the std::vector object,
* meaning it can be directly passed to CUDA without any memory transfers.
*/

/*
* NOTE: Adding custom memory allocation so that you can choose if the mesh data
* should be in GPU or CPU. The reasoning is simple ToyTracer needs Meshes to be
* ready for GPU usage while Bubbles don't actually use meshes on the GPU, only
* for emittions and colliding with SDFs. If no memory allocator is setup all memory
* allocated are made through cutil.h and are GPU ready.
*/

/*
* Loads a obj file and [optionally] split the objects found into multiple meshes.
* It also gets the list of mtls found, can be passed directly to the MTL API for
* material loading. mtls can be nullptr, in which case only the mesh list is returned.
*/
__host__ std::vector<ParsedMesh*> *LoadObj(const char *path, std::vector<MeshMtl> *mtls,
                                           bool split_mesh=true);

/*
* If you want to simply load the mesh and not split and not get any materials
* and do your own thing, call this. It will be faster and use less memory.
*/
__host__ ParsedMesh *LoadObj(const char *path);

/*
* Duplicates the information of a mesh applying the MeshProperties given,
* not very efficient but usefull.
*/
__host__ ParsedMesh *DuplicateMesh(ParsedMesh *mesh, MeshProperties *props=nullptr);

/*
* Configure function allocators to your desired environment, if no configuration
* is performed Meshes are allocated for GPU usage.
*/
__host__ void SetAllocators(MemoryAllocator memAlloc, MemoryFree memFree);

/*
* Configures the allocators by flag instead. If you don't want to 
* provide allocators use 'is_cpu = 1' so API use defaults CPU allocation or '0'
* for default GPU allocators.
*/
__host__ void UseDefaultAllocatorFor(AllocatorType type);

//////////////////////////////////////////////////////
// Exposing parsing routines for serializer
//////////////////////////////////////////////////////

/*
* Get one line from the input stream and return it in 't'.
*/
__host__ std::istream &GetLine(std::istream &is, std::string &t);

/*
* Parse a single float and move token head.
*/
__host__ Float ParseFloat(const char **token);

/*
* Parse a single vec2f and move token head.
*/
__host__ void ParseV2(vec2f *v, const char **token);

/*
* Parse a single vec3f and move token head.
*/
__host__ void ParseV3(vec3f *v, const char **token);