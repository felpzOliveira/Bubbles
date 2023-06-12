#pragma once

#include <geometry.h>
#include <transform.h>
#include <grid.h>

class SurfaceInteraction2;
class ClosestPointQuery2;
class SurfaceInteraction;
class ClosestPointQuery;

struct ParsedMesh{
    Point3f *p;
    Normal3f *n;
    vec3f *s;
    Point2f *uv;
    Point3i *indices;
    int nTriangles, nVertices;
    int nUvs, nNormals;
    AllocatorType allocator;
    char name[256];
    Transform transform;
};

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
    ShapeSDF,
}ShapeType;

class Shape2{
    public:
    Transform2 WorldToObject, ObjectToWorld;
    vec2f linearVelocity;
    Float angularVelocity;
    bool reverseOrientation;
    ShapeType type;

    // Sphere2
    Float radius;

    // Rectangle
    Bounds2f rect;

    // SDF
    FieldGrid2f *grid;

    bb_cpu_gpu Shape2(){}
    bb_cpu_gpu Shape2(const Transform2 &toWorld, bool reverseOrientation=false);
    bb_cpu_gpu void InitSphere2(const Transform2 &toWorld, Float radius,
                                bool reverseOrientation=false);

    bb_cpu_gpu void InitRectangle2(const Transform2 &toWorld, vec2f extension,
                                   bool reverseOrientation=false);

    bb_cpu_gpu Bounds2f GetBounds();

    void Update(const Transform2 &toWorld);

    bb_cpu_gpu void SetVelocities(const vec2f &vel, const Float &angular);

    bb_cpu_gpu bool Intersect(const Ray2 &ray, SurfaceInteraction2 *isect,
                              Float *tShapeHit) const;

    bb_cpu_gpu Float ClosestDistance(const vec2f &point) const;

    bb_cpu_gpu void ClosestPoint(const vec2f &point,
                                 ClosestPointQuery2 *query) const;

    bb_cpu_gpu bool IsInside(const vec2f &point) const;

    bb_cpu_gpu Float SignedDistance(const vec2f &point) const;

    bb_cpu_gpu vec2f VelocityAt(const vec2f &point) const;

    private:
    ///////////////////////
    // SPHERE 2D functions
    ///////////////////////
    bb_cpu_gpu Bounds2f Sphere2GetBounds();
    bb_cpu_gpu bool Sphere2Intersect(const Ray2 &ray, SurfaceInteraction2 *isect,
                                     Float *tShapeHit) const;
    bb_cpu_gpu Float Sphere2ClosestDistance(const vec2f &point) const;
    bb_cpu_gpu void Sphere2ClosestPoint(const vec2f &point,
                                        ClosestPointQuery2 *query) const;

    //////////////////////////
    // RECTANGLE 2D functions
    //////////////////////////
    bb_cpu_gpu Bounds2f Rectangle2GetBounds();
    bb_cpu_gpu bool Rectangle2Intersect(const Ray2 &ray, SurfaceInteraction2 *isec,
                                        Float *tShapeHit) const;
    bb_cpu_gpu Float Rectangle2ClosestDistance(const vec2f &point) const;
    bb_cpu_gpu void Rectangle2ClosestPoint(const vec2f &point,
                                           ClosestPointQuery2 *query) const;

    bb_cpu_gpu void ClosestPointBySDF(const vec2f &point,
                                      ClosestPointQuery2 *query) const;
};

Shape2 *MakeSphere2(const Transform2 &toWorld, Float radius,
                    bool reverseOrientation = false);
Shape2 *MakeRectangle2(const Transform2 &toWorld, vec2f extension,
                       bool reverseOrientation = false);
Node *MakeBVH(ParsedMesh *mesh, int maxDepth);
bb_cpu_gpu bool BVHMeshIntersect(const Ray &r, SurfaceInteraction *isect,
                                 Float *tHit, ParsedMesh *mesh, Node *bvh);
bb_cpu_gpu Float BVHMeshClosestDistance(const vec3f &point, int *closest,
                                        ParsedMesh *mesh, Node *bvh);

class Plane3{
    public:
    vec3f point;
    Normal3f normal;

    bb_cpu_gpu Plane3();
    bb_cpu_gpu Plane3(const vec3f &p, const Normal3f &n);
    bb_cpu_gpu void Set(const vec3f &p, const Normal3f &n);
    bb_cpu_gpu vec3f ClosestPoint(const vec3f &p) const;
};

class Shape{
    public:
    Transform WorldToObject, ObjectToWorld;
    vec3f linearVelocity;
    vec3f angularVelocity;
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

    // Sdf shape
    Bounds3f bounds;

    bb_cpu_gpu Shape(){}
    bb_cpu_gpu Shape(const Transform &toWorld, bool reverseOrientation=false);
    bb_cpu_gpu void InitSphere(const Transform &toWorld, Float radius,
                               bool reverseOrientation=false);
    void InitBox(const Transform &toWorld, Float sizex, Float sizey, Float sizez,
                 bool reverseOrientation=false);
    void InitMesh(ParsedMesh *mesh, bool reverseOrientation=false, int maxDepth=22);

    bb_cpu_gpu bool CanSolveSdf(void) const;
    bb_cpu_gpu Bounds3f GetBounds();
    bb_cpu_gpu bool Intersect(const Ray &ray, SurfaceInteraction *isect,
                              Float *tShapeHit) const;
    bb_cpu_gpu Float ClosestDistance(const vec3f &point) const;

    bb_cpu_gpu void ClosestPoint(const vec3f &point,
                                 ClosestPointQuery *query) const;

    bb_cpu_gpu void SetVelocities(const vec3f &vel, const vec3f &angular);

    bb_cpu_gpu bool IsInside(const vec3f &point) const;

    bb_cpu_gpu Float SignedDistance(const vec3f &point) const;

    void Update(const Transform &toWorld);

    bb_cpu_gpu vec3f VelocityAt(const vec3f &point) const;

    std::string Serialize() const;

    template<typename F>
    void UpdateSDF(F sdf){
        GPUParallelLambda("Shape SDF", grid->total, GPU_LAMBDA(int index){
            vec3ui u = DimensionalIndex(index, grid->resolution, 3);
            vec3f p = grid->GetDataPosition(u);
            Float distance = sdf(p, this);
            grid->SetValueAt(distance, u);
        });
    }

    template<typename F>
    void InitSDFShape(const Bounds3f &maxBounds, Float dx, Float margin, F sdf){
        bounds = maxBounds;
        type = ShapeType::ShapeSDF;
        WorldToObject = Transform();
        ObjectToWorld = Transform();
        vec3f scale(bounds.ExtentOn(0), bounds.ExtentOn(1), bounds.ExtentOn(2));
        bounds.pMin -= margin * scale;
        bounds.pMax += margin * scale;

        Float width = bounds.ExtentOn(0);
        Float height = bounds.ExtentOn(1);
        Float depth = bounds.ExtentOn(2);
        int resolution = (int)std::ceil(width / dx);
        dx = width / (Float)resolution;
        int resolutionY = (int)std::ceil(resolution * height / width);
        int resolutionZ = (int)std::ceil(resolution * depth / width);

        grid = cudaAllocateVx(FieldGrid3f, 1);
        grid->Build(vec3ui(resolution, resolutionY, resolutionZ), vec3f(dx),
                    bounds.pMin, VertexCentered);

        printf("Generating SDF for shape: [%d x %d x %d] ... ",
               resolution, resolutionY, resolutionZ);

        UpdateSDF(sdf);

        grid->MarkFilled();
        printf("OK\n");
    }

    private:
    ///////////////////////
    // SPHERE functions
    ///////////////////////
    bb_cpu_gpu Bounds3f SphereGetBounds();
    bb_cpu_gpu bool SphereIntersect(const Ray &ray, SurfaceInteraction *isect,
                                    Float *tShapeHit) const;
    bb_cpu_gpu Float SphereClosestDistance(const vec3f &point) const;
    bb_cpu_gpu void SphereClosestPoint(const vec3f &point,
                                       ClosestPointQuery *query) const;
    std::string SphereSerialize() const;

    ///////////////////////
    // BOX functions
    ///////////////////////
    bb_cpu_gpu Bounds3f BoxGetBounds();
    bb_cpu_gpu bool BoxIntersect(const Ray &ray, SurfaceInteraction *isect,
                                   Float *tShapeHit) const;
    bb_cpu_gpu Float BoxClosestDistance(const vec3f &point) const;
    bb_cpu_gpu void BoxClosestPoint(const vec3f &point,
                                      ClosestPointQuery *query) const;
    std::string BoxSerialize() const;
    ///////////////////////
    // MESH functions
    ///////////////////////
    bb_cpu_gpu Bounds3f MeshGetBounds();
    bb_cpu_gpu bool MeshIntersect(const Ray &ray, SurfaceInteraction *isect,
                                  Float *tShapeHit) const;
    bb_cpu_gpu Float MeshClosestDistance(const vec3f &point) const;
    bb_cpu_gpu void MeshClosestPoint(const vec3f &point,
                                     ClosestPointQuery *query) const;

    bb_cpu_gpu void ClosestPointBySDF(const vec3f &point,
                                      ClosestPointQuery *query) const;
    std::string MeshSerialize() const;

};

/*
* Wrappers for easy shape constructors.
*/
Shape *MakeSphere(const Transform &toWorld, Float radius, bool reverseOrientation = false);
Shape *MakeBox(const Transform &toWorld, const vec3f &size, bool reverseOrientation = false);
Shape *MakeMesh(ParsedMesh *mesh, const Transform &toWorld, bool reverseOrientation = false);
Shape *MakeMesh(const char *path, const Transform &toWorld, bool reverseOrientation = false);

// Use this for custom SDF functions, the sdf function must have signature given by:
//   GPU_LAMBDA(vec3f p, Shape *shape) -> Float{ .... return signedDistance; };
// because shape SDF initialization is done in GPU code.
template<typename F>
inline Shape *MakeSDFShape(Bounds3f bounds, F sdf, Float dx=0.01, Float margin=0.1){
    Shape *shape = cudaAllocateVx(Shape, 1);
    shape->InitSDFShape(bounds, dx, margin, sdf);
    return shape;
}

// These ones are called by the collider interface when it detects it needs a SDF
// to solve collision, no need to call manually.
void GenerateShapeSDF(Shape *shape, Float dx=0.01, Float margin=0.1);
void GenerateShapeSDF(Shape2 *shape, Float dx=0.01, Float margin=0.1);

/*
* Expose also symbols for tests
*/
bb_kernel void CreateShapeSDFGPU(Shape *shape);
void CreateShapeSDFCPU(Shape *shape);

/*
* Performs Ray Tracing to find if point 'point' is inside the mesh
* described by 'meshShape' using 'bounds' as a clip container.
*/
bb_cpu_gpu bool MeshIsPointInside(const vec3f &point, Shape *meshShape,
                                  const Bounds3f &bounds);

bb_cpu_gpu bool MeshShapeIsPointInside(Shape *meshShape, const vec3f &p,
                                       Float radius, Float offset=0.0f);

// TODO: Is this busted? see surface.cpp build for zhu-bridson and this one,
//       this routine only samples p.y > 0 ??? marching cubes issues?
template<typename F>
FieldGrid3f *CreateSDF(Bounds3f bounds, vec3f spacing, F sdf){
    FieldGrid3f *grid = cudaAllocateVx(FieldGrid3f, 1);
    Float width = bounds.ExtentOn(0);
    Float height = bounds.ExtentOn(1);
    Float depth = bounds.ExtentOn(2);
    int extraOff = 10;
    int rx = (int)std::ceil(width / spacing.x) + extraOff;
    int ry = (int)std::ceil(width / spacing.y) + extraOff;
    int rz = (int)std::ceil(width / spacing.z) + extraOff;

    vec3f p0 = bounds.pMin - spacing * extraOff * 0.5;
    grid->Build(vec3ui(rx, ry, rz), spacing, p0, VertexCentered);

    GPUParallelLambda("CreateSDF", grid->total, GPU_LAMBDA(int index){
        vec3ui u = DimensionalIndex(index, grid->resolution, 3);
        vec3f p = grid->GetDataPosition(u);
        Float distance = sdf(p);
        grid->SetValueAt(distance, u);
    });

    grid->MarkFilled();
    return grid;
}

template<typename F>
FieldGrid3f *CreateSDFByIndex(Bounds3f bounds, vec3f spacing, F sdf){
    FieldGrid3f *grid = cudaAllocateVx(FieldGrid3f, 1);
    Float width = bounds.ExtentOn(0);
    Float height = bounds.ExtentOn(1);
    Float depth = bounds.ExtentOn(2);
    int extraOff = 10;
    int rx = (int)std::ceil(width / spacing.x) + extraOff;
    int ry = (int)std::ceil(width / spacing.y) + extraOff;
    int rz = (int)std::ceil(width / spacing.z) + extraOff;

    vec3f p0 = bounds.pMin - spacing * extraOff * 0.5;
    grid->Build(vec3ui(rx, ry, rz), spacing, p0, VertexCentered);

    GPUParallelLambda("CreateSDF", grid->total, GPU_LAMBDA(int index){
        vec3ui u = DimensionalIndex(index, grid->resolution, 3);
        Float distance = sdf(u);
        grid->SetValueAt(distance, u);
    });

    grid->MarkFilled();
    return grid;
}
