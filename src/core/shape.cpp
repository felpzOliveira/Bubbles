#include <shape.h>
#include <interaction.h>
#include <collider.h>

/*************************************************************/
//                   2 D    S H A P E S                      //
/*************************************************************/
__bidevice__ Shape2::Shape2(const Transform2 &toWorld, bool reverseOrientation)
: ObjectToWorld(toWorld), WorldToObject(Inverse(toWorld)), 
reverseOrientation(reverseOrientation){}

__bidevice__ bool Shape2::IsInside(const vec2f &point) const{
    return reverseOrientation == !(ClosestDistance(point) < 0);
}

__bidevice__ Float Shape2::SignedDistance(const vec2f &point) const{
    Float d = ClosestDistance(point);
    if(IsInside(point)) return -Absf(d);
    return Absf(d);
}

__bidevice__ Bounds2f Shape2::GetBounds(){
    switch(type){
        case ShapeType::ShapeSphere2:{
            return Sphere2GetBounds();
        } break;
        
        case ShapeType::ShapeRectangle2:{
            return Rectangle2GetBounds();
        } break;
        
        default:{
            printf("Unknown shape for Shape2::GetBounds\n");
            return Bounds2f();
        }
    }
}

__bidevice__ bool Shape2::Intersect(const Ray2 &ray, SurfaceInteraction2 *isect,
                                    Float *tShapeHit) const
{
    switch(type){
        case ShapeType::ShapeSphere2:{
            return Sphere2Intersect(ray, isect, tShapeHit);
        } break;
        
        case ShapeType::ShapeRectangle2:{
            return Rectangle2Intersect(ray, isect, tShapeHit);
        } break;
        
        default:{
            printf("Unknown shape for Shape2::Intersect\n");
            return false;
        }
    }
}

__bidevice__ Float Shape2::ClosestDistance(const vec2f &point) const{
    switch(type){
        case ShapeType::ShapeSphere2:{
            return Sphere2ClosestDistance(point);
        } break;
        
        case ShapeType::ShapeRectangle2:{
            return Rectangle2ClosestDistance(point);
        } break;
        
        default:{
            printf("Unknown shape for Shape2::ClosestDistance\n");
            return Infinity;
        }
    }
}

__bidevice__ void Shape2::ClosestPoint(const vec2f &point, 
                                       ClosestPointQuery2 *query) const
{
    switch(type){
        case ShapeType::ShapeSphere2:{
            return Sphere2ClosestPoint(point, query);
        } break;
        
        case ShapeType::ShapeRectangle2:{
            return Rectangle2ClosestPoint(point, query);
        } break;
        
        default:{
            printf("Unknown shape for Shape2::ClosestPoint\n");
        }
    }
}


/*************************************************************/
//                   3 D    S H A P E S                      //
/*************************************************************/
__bidevice__ Shape::Shape(const Transform &toWorld, bool reverseOrientation) :
ObjectToWorld(toWorld), WorldToObject(Inverse(toWorld)),
reverseOrientation(reverseOrientation){}

__bidevice__ bool Shape::CanSolveSdf() const{
    switch(type){
        case ShapeType::ShapeSphere:{
            return true;
        } break;
        
        case ShapeType::ShapeBox:{
            return true;
        } break;
        
        case ShapeType::ShapeMesh:{
            return false;
        } break;
        
        default:{
            printf("Unknown shape for Shape::CanSolveSdf\n");
            return false;
        }
    }
}

__bidevice__ Bounds3f Shape::GetBounds(){
    switch(type){
        case ShapeType::ShapeSphere:{
            return SphereGetBounds();
        } break;
        
        case ShapeType::ShapeBox:{
            return BoxGetBounds();
        } break;
        
        case ShapeType::ShapeMesh:{
            return MeshGetBounds();
        } break;
        
        default:{
            printf("Unknown shape for Shape::GetBounds\n");
            return Bounds3f();
        }
    }
}

__bidevice__ bool Shape::Intersect(const Ray &ray, SurfaceInteraction *isect,
                                   Float *tShapeHit) const
{
    switch(type){
        case ShapeType::ShapeSphere:{
            return SphereIntersect(ray, isect, tShapeHit);
        } break;
        
        case ShapeType::ShapeBox:{
            return BoxIntersect(ray, isect, tShapeHit);
        } break;
        
        case ShapeType::ShapeMesh:{
            return MeshIntersect(ray, isect, tShapeHit);
        } break;
        
        default:{
            printf("Unknown shape for Shape::Intersect\n");
            return false;
        }
    }
}

__bidevice__ Float Shape::ClosestDistance(const vec3f &point) const{
    switch(type){
        case ShapeType::ShapeSphere:{
            return SphereClosestDistance(point);
        } break;
        
        case ShapeType::ShapeBox:{
            return BoxClosestDistance(point);
        } break;
        
        case ShapeType::ShapeMesh:{
            return MeshClosestDistance(point);
        } break;
        
        default:{
            printf("Unknown shape for Shape::ClosestDistance\n");
            return Infinity;
        }
    }
}

__bidevice__ void Shape::ClosestPoint(const vec3f &point, 
                                      ClosestPointQuery *query) const
{
    switch(type){
        case ShapeType::ShapeSphere:{
            SphereClosestPoint(point, query);
        } break;
        
        case ShapeType::ShapeBox:{
            return BoxClosestPoint(point, query);
        } break;
        
        case ShapeType::ShapeMesh:{
            MeshClosestPoint(point, query);
        } break;
        
        default:{
            printf("Unknown shape for Shape::ClosestPoint\n");
        }
    }
}

__bidevice__ bool Shape::IsInside(const vec3f &point) const{
    if(type != ShapeType::ShapeMesh){
        return reverseOrientation == !(ClosestDistance(point) < 0);
    }else{
        // prevent BVH query during simulation, it should use FieldGrid
        return false;
    }
}

__bidevice__ Float Shape::SignedDistance(const vec3f &point) const{
    Float d = ClosestDistance(point);
    if(IsInside(point)) return -Absf(d);
    return Absf(d);
}

/*
* Compute a ray direction from a point and a bounding box being sampled,
* all computations must be performed centered at the origin.
*/
__bidevice__ vec3f GenerateMinimalRayDirection(const vec3f &origin, const Bounds3f &bound){
    vec3f dir(0);
    Float dx = Absf(bound.ExtentOn(0) * 0.5f - origin.x);
    Float dy = Absf(bound.ExtentOn(1) * 0.5f - origin.y);
    Float dz = Absf(bound.ExtentOn(2) * 0.5f - origin.z);
    vec4f ref[] = {
        vec4f(dx, 1, 0, 0),
        vec4f(dy, 0, 1, 0),
        vec4f(dz, 0, 0, 1),
        vec4f(Absf(bound.ExtentOn(0) - dx), -1, 0, 0),
        vec4f(Absf(bound.ExtentOn(1) - dy), 0, -1, 0),
        vec4f(Absf(bound.ExtentOn(2) - dz), 0, 0, -1)
    };
    
    Float mind = Infinity;
    for(int i = 0; i < 6; i++){
        if(mind > ref[i].x){
            dir = vec3f(ref[i].y, ref[i].z, ref[i].w);
            mind = ref[i].x;
        }
    }
    
    return dir;
}

__bidevice__ bool MeshIsPointInside(const vec3f &point, Shape *meshShape,
                                    const Bounds3f &bounds)
{
    int hits = 0;
    bool hit_anything = false;
    vec3f direction = GenerateMinimalRayDirection(point, bounds);
    Ray ray(point, direction);
    
    do{
        SurfaceInteraction isect;
        Float tHit = 0;
        hit_anything = meshShape->Intersect(ray, &isect, &tHit);
        if(hit_anything){
            ray = SpawnRayInDirection(isect, ray.d);
            hits++;
        }
    }while(hit_anything);
    
    return (hits % 2 != 0 && hits > 0);
}

__bidevice__ void SetNodeSDFKernel(FieldGrid3f *grid, ParsedMesh *mesh, 
                                   Shape *shape, int i)
{
    vec3ui u = DimensionalIndex(i, grid->resolution, 3);
    vec3f p = grid->GetVertexPosition(u);
    Float d = shape->ClosestDistance(p);
    bool interior = MeshIsPointInside(p, shape, shape->GetBounds()); 
    Float sd = interior ? -d : d;
    grid->SetValueAt(sd, u);
}

__global__ void CreateShapeSDFGPU(FieldGrid3f *grid, ParsedMesh *mesh, Shape *shape){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < grid->total){
        SetNodeSDFKernel(grid, mesh, shape, i);
    }
}

__host__ void CreateShapeSDFCPU(FieldGrid3f *grid, ParsedMesh *mesh, Shape *shape){
    for(int i = 0; i < grid->total; i++){
        SetNodeSDFKernel(grid, mesh, shape, i);
    }
}

__host__ void GenerateMeshShapeSDF(Shape *shape, Float dx, Float margin){
    if(shape->type != ShapeType::ShapeMesh){
        printf("Warning: Called for SDF generation on non-mesh shape\n");
    }else{
        //TODO: Update grid if already exists
        
        int resolution = 0;
        Bounds3f bounds = shape->GetBounds();
        vec3f scale(bounds.ExtentOn(0), bounds.ExtentOn(1), bounds.ExtentOn(2));
        bounds.pMin -= margin * scale;
        bounds.pMax += margin * scale;
        
        Float width = bounds.ExtentOn(0);
        Float height = bounds.ExtentOn(1);
        Float depth = bounds.ExtentOn(2);
        
        resolution = (int)std::ceil(width / dx);
#if 0
        if(resolution > 200){ // very high resolution reduce
            resolution = 200;
        }
#endif
        
        dx = width / (Float)resolution;
        int resolutionY = (int)std::ceil(resolution * height / width);
        int resolutionZ = (int)std::ceil(resolution * depth / width);
        
        shape->grid = cudaAllocateVx(FieldGrid3f, 1);
        shape->grid->Build(vec3ui(resolution, resolutionY, resolutionZ), vec3f(dx), 
                           bounds.pMin, VertexCentered);
        
        printf("Generating SDF for mesh: [%d x %d x %d] ... ",
               resolution, resolutionY, resolutionZ);
        
        if(shape->mesh->allocator != AllocatorType::GPU){
            static int warned = 0;
            if(warned == 0){
                printf("\nWarning: Requested for SDF with Mesh not bound to GPU.\n");
                printf("         Make sure solvers are set to use CPU.\n");
                warned++;
            }
            CreateShapeSDFCPU(shape->grid, shape->mesh, shape);
        }else{
            GPULaunch(shape->grid->total, CreateShapeSDFGPU, 
                      shape->grid, shape->mesh, shape);
        }
        
        printf("OK\n");
    }
}