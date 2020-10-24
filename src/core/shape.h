#pragma once

#include <geometry.h>
#include <transform.h>
#include <grid.h>

/*
* NOTE: Currently the Mesh shape is only supported for particle emitter
*       and its not 100% OK yet.
*/

class SurfaceInteraction2;
class ClosestPointQuery2;
class SurfaceInteraction;
class ClosestPointQuery;

typedef struct{
    Bounds3f bound;
    int handle;
}PrimitiveHandle;

typedef struct Node_t{
    struct Node_t *left, *right;
    PrimitiveHandle *handles;
    int n;
    int is_leaf;
    Bounds3f bound;
}Node;

typedef Node* NodePtr;

typedef enum{
    ShapeSphere2 = 1,
    ShapeRectangle2,
    ShapeSphere,
    ShapeMesh,
    ShapeBox,
}ShapeType;

class Shape2{
    public:
    Transform2 WorldToObject, ObjectToWorld;
    bool reverseOrientation;
    ShapeType type;
    
    // Sphere2
    Float radius;
    
    // Rectangle
    Bounds2f rect;
    
    __bidevice__ Shape2(){}
    __bidevice__ Shape2(const Transform2 &toWorld, bool reverseOrientation=false);
    __bidevice__ void InitSphere2(const Transform2 &toWorld, Float radius,
                                  bool reverseOrientation=false);
    
    __bidevice__ void InitRectangle2(const Transform2 &toWorld, vec2f extension,
                                     bool reverseOrientation=false);
    
    __bidevice__ Bounds2f GetBounds();
    
    __bidevice__ bool Intersect(const Ray2 &ray, SurfaceInteraction2 *isect,
                                Float *tShapeHit) const;
    
    __bidevice__ Float ClosestDistance(const vec2f &point) const;
    
    __bidevice__ void ClosestPoint(const vec2f &point, 
                                   ClosestPointQuery2 *query) const;
    
    __bidevice__ bool IsInside(const vec2f &point) const;
    
    __bidevice__ Float SignedDistance(const vec2f &point) const;
    
    private:
    ///////////////////////
    // SPHERE 2D functions
    ///////////////////////
    __bidevice__ Bounds2f Sphere2GetBounds();
    __bidevice__ bool Sphere2Intersect(const Ray2 &ray, SurfaceInteraction2 *isect,
                                       Float *tShapeHit) const;
    __bidevice__ Float Sphere2ClosestDistance(const vec2f &point) const;
    __bidevice__ void Sphere2ClosestPoint(const vec2f &point, 
                                          ClosestPointQuery2 *query) const;
    
    //////////////////////////
    // RECTANGLE 2D functions
    //////////////////////////
    __bidevice__ Bounds2f Rectangle2GetBounds();
    __bidevice__ bool Rectangle2Intersect(const Ray2 &ray, SurfaceInteraction2 *isec,
                                          Float *tShapeHit) const;
    __bidevice__ Float Rectangle2ClosestDistance(const vec2f &point) const;
    __bidevice__ void Rectangle2ClosestPoint(const vec2f &point, 
                                             ClosestPointQuery2 *query) const;
};

__host__ Shape2 *MakeSphere2(const Transform2 &toWorld, Float radius,
                             bool reverseOrientation = false);
__host__ Shape2 *MakeRectangle2(const Transform2 &toWorld, vec2f extension,
                                bool reverseOrientation = false);
__host__ Node *MakeBVH(ParsedMesh *mesh, int maxDepth);
__bidevice__ bool BVHMeshIntersect(const Ray &r, SurfaceInteraction *isect,
                                   Float *tHit, ParsedMesh *mesh, Node *bvh);
__bidevice__ Float BVHMeshClosestDistance(const vec3f &point, int *closest,
                                          ParsedMesh *mesh, Node *bvh);

class Plane3{
    public:
    vec3f point;
    Normal3f normal;
    
    __bidevice__ Plane3();
    __bidevice__ Plane3(const vec3f &p, const Normal3f &n);
    __bidevice__ void Set(const vec3f &p, const Normal3f &n);
    __bidevice__ vec3f ClosestPoint(const vec3f &p) const;
};

class Shape{
    public:
    Transform WorldToObject, ObjectToWorld;
    bool reverseOrientation;
    ShapeType type;
    
    // Sphere
    Float radius;
    
    // Box
    Float sizex, sizey, sizez;
    
    // Mesh
    ParsedMesh *mesh;
    FieldGrid3f *grid;
    Node *bvh;
    
    __bidevice__ Shape(){}
    __bidevice__ Shape(const Transform &toWorld, bool reverseOrientation=false);
    __bidevice__ void InitSphere(const Transform &toWorld, Float radius,
                                 bool reverseOrientation=false);
    __host__ void InitBox(const Transform &toWorld, Float sizex, Float sizey, Float sizez,
                          bool reverseOrientation=false);
    __host__ void InitMesh(ParsedMesh *mesh, bool reverseOrientation=false, 
                           int maxDepth=22);
    __bidevice__ bool CanSolveSdf(void) const;
    __bidevice__ Bounds3f GetBounds();
    __bidevice__ bool Intersect(const Ray &ray, SurfaceInteraction *isect,
                                Float *tShapeHit) const;
    __bidevice__ Float ClosestDistance(const vec3f &point) const;
    
    __bidevice__ void ClosestPoint(const vec3f &point, 
                                   ClosestPointQuery *query) const;
    
    __bidevice__ bool IsInside(const vec3f &point) const;
    
    __bidevice__ Float SignedDistance(const vec3f &point) const;
    
    private:
    ///////////////////////
    // SPHERE functions
    ///////////////////////
    __bidevice__ Bounds3f SphereGetBounds();
    __bidevice__ bool SphereIntersect(const Ray &ray, SurfaceInteraction *isect,
                                      Float *tShapeHit) const;
    __bidevice__ Float SphereClosestDistance(const vec3f &point) const;
    __bidevice__ void SphereClosestPoint(const vec3f &point, 
                                         ClosestPointQuery *query) const;
    
    ///////////////////////
    // BOX functions
    ///////////////////////
    __bidevice__ Bounds3f BoxGetBounds();
    __bidevice__ bool BoxIntersect(const Ray &ray, SurfaceInteraction *isect,
                                   Float *tShapeHit) const;
    __bidevice__ Float BoxClosestDistance(const vec3f &point) const;
    __bidevice__ void BoxClosestPoint(const vec3f &point, 
                                      ClosestPointQuery *query) const;
    ///////////////////////
    // MESH functions
    ///////////////////////
    __bidevice__ Bounds3f MeshGetBounds();
    __bidevice__ bool MeshIntersect(const Ray &ray, SurfaceInteraction *isect,
                                    Float *tShapeHit) const;
    __bidevice__ Float MeshClosestDistance(const vec3f &point) const;
    __bidevice__ void MeshClosestPoint(const vec3f &point, 
                                       ClosestPointQuery *query) const;
};

__host__ Shape *MakeSphere(const Transform &toWorld, Float radius,
                           bool reverseOrientation = false);
__host__ Shape *MakeBox(const Transform &toWorld, const vec3f &size, 
                        bool reverseOrientation = false);
__host__ Shape *MakeMesh(ParsedMesh *mesh, const Transform &toWorld,
                         bool reverseOrientation = false);

__host__ void GenerateMeshShapeSDF(Shape *shape, Float dx=0.01, Float margin=0.1);

/*
* Expose also symbols for tests
*/
__global__ void CreateShapeSDFGPU(FieldGrid3f *grid, ParsedMesh *mesh, Shape *shape);
__host__ void CreateShapeSDFCPU(FieldGrid3f *grid, ParsedMesh *mesh, Shape *shape);

/*
* Performs Ray Tracing to find if point 'point' is inside the mesh
* described by 'meshShape' using 'bounds' as a clip container.
*/
__bidevice__ bool MeshIsPointInside(const vec3f &point, Shape *meshShape,
                                    const Bounds3f &bounds);